import logging
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes, pairwise_iou  #
from fcos.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

from fcos.utils.comm import reduce_sum
from fcos.layers import ml_nms
from .utils import *
from .lovasz import LovaszHinge

from detectron2.layers import interpolate

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores
    
"""


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
              (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


class FCOSOutputs(object):
    def __init__(
            self,
            images,
            locations,
            logits_pred,
            reg_pred,
            ctrness_pred,
            focal_loss_alpha,
            focal_loss_gamma,
            iou_loss,
            center_sample,
            sizes_of_interest,
            strides,
            radius,
            num_classes,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            thresh_with_ctr,
            proposal_embeds,
            pixel_embed,
            margins,
            embed_dim,
            gt_instances=None,
    ):
        self.logits_pred = logits_pred
        self.reg_pred = reg_pred
        self.ctrness_pred = ctrness_pred
        self.locations = locations

        self.proposal_embeds = proposal_embeds
        self.pixel_embed = pixel_embed
        self.margins = margins
        self.embed_dim = embed_dim
        self.mask_loss_func = LovaszHinge(reduction='none')
        self.loss_mask_alpha = 0.5
        self.loss_smooth_alpha = 0.1

        self.gt_instances = gt_instances
        self.num_feature_maps = len(logits_pred)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss = iou_loss
        self.center_sample = center_sample
        self.sizes_of_interest = sizes_of_interest
        self.strides = strides
        self.radius = radius
        self.num_classes = num_classes
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.thresh_with_ctr = thresh_with_ctr
        self.mask_scale_factor = 2

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self):
        num_loc_list = [len(loc) for loc in self.locations]
        self.num_loc_list = num_loc_list

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(self.locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(self.locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, self.gt_instances, loc_to_size_range
        )

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])

        return training_targets

    def get_sample_region(self, gt, strides, num_loc_list, loc_xs, loc_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(loc_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges):
        labels = []
        reg_targets = []
        matched_idxes = []
        im_idxes = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, self.num_loc_list,
                    xs, ys, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            matched_idxes.append(locations_to_gt_inds)
            im_idxes.append(torch.tensor([im_i] * len(labels_per_im)).to(locations_to_gt_inds.device))
        return {"labels": labels, "reg_targets": reg_targets, "matched_idxes": matched_idxes, "im_idxes": im_idxes}

    def losses(self):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self._get_ground_truth()
        labels, reg_targets, matched_idxes, im_idxes = training_targets["labels"], training_targets["reg_targets"], \
                                                       training_targets["matched_idxes"], training_targets["im_idxes"]

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.
        logits_pred = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in self.logits_pred
            ], dim=0, )
        reg_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4)
                for x in self.reg_pred
            ], dim=0, )
        ctrness_pred = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in self.ctrness_pred
            ], dim=0, )

        labels = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in labels
            ], dim=0, )

        reg_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in reg_targets
            ], dim=0, )

        # proposal_embed_pred = cat(
        #     [
        #         # (N, Hi, Wi, dim) -> (N*Hi*Wi, dim)
        #         x.permute(0, 2, 3, 1).reshape(-1, self.embed_dim) for x in self.proposal_embeds
        #     ], dim=0, )
        # margin_pred = cat(
        #     [
        #         # (N, Hi, Wi, 1) -> (N*Hi*Wi, 1)
        #         x.permute(0, 2, 3, 1).reshape(-1, 1) for x in self.margins
        #     ], dim=0, )

        # (N, Hi, Wi, ?) -> (N, Hi*Wi, ?), List
        N = self.proposal_embeds[0].size(0)
        proposal_embed_pred = [x.permute(0, 2, 3, 1).reshape(N, -1, self.embed_dim) for x in self.proposal_embeds]
        margin_pred = [x.permute(0, 2, 3, 1).reshape(N, -1, 1) for x in self.margins]
        # (N, H_0, W_0, 1) -> (N, H_0*W_0, dim)
        # pixel_embed = self.pixel_embed.permute(0, 2, 3, 1).reshape(N, -1, self.embed_dim)

        matched_idxes = cat(
            [
                x.reshape(-1) for x in matched_idxes
            ], dim=0, )
        im_idxes = cat(
            [
                x.reshape(-1) for x in im_idxes
            ], dim=0, )
        return self.fcos_losses(
            labels,
            reg_targets,
            logits_pred,
            reg_pred,
            ctrness_pred,

            proposal_embed_pred,
            margin_pred,

            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            matched_idxes,
        )

    def predict_proposals(self):
        sampled_boxes = []

        bundle = (
            self.locations, self.logits_pred,
            self.reg_pred, self.ctrness_pred,
            self.strides
        )

        for i, (l, o, r, c, s) in enumerate(zip(*bundle)):
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            r = r * s

            proposal_embeds = self.proposal_embeds[i]
            margins = self.margins[i]

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, c, self.image_sizes,
                    proposal_embeds, margins
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        # for embedmask
        boxlists = self.forward_for_mask(boxlists)

        return boxlists

    def compute_mask_prob(self, pixel_embed, proposal_embed, margins):
        '''
        :param pixel_embed: [dim, H, W]
        :param proposal_embed: [n, dim]
        :param margins: [n, 1]
        :return:
        '''
        dim, m_h, m_w = pixel_embed.shape
        margins = margins.view(-1, 1)
        n = margins.size(0)
        # -> (n, H0*W0, dim)
        pixel_embed = pixel_embed.view(dim, m_h * m_w).permute(1, 0).unsqueeze(0).repeat(n, 1, 1)
        # -> (n, 1, dim)
        proposal_embed = proposal_embed.unsqueeze(1)

        mask_var = torch.sum((pixel_embed - proposal_embed) ** 2, dim=2)  # (n, H0*W0)
        mask_prob = torch.exp(-mask_var * margins)

        return mask_prob.view(1, n, m_h, m_w)

    def forward_for_mask(self, boxlists):
        N, dim, m_h, m_w = self.pixel_embed.shape
        o_h = int(m_h * self.strides[0])
        o_w = int(m_w * self.strides[0])
        stride = self.strides[0] / self.mask_scale_factor
        for im in range(N):
            boxlist = boxlists[im]
            input_h, input_w = boxlist.image_size
            proposal_embed = boxlist.proposal_embeds
            if len(proposal_embed) == 0:
                boxlist.mask = torch.tensor([])
                continue

            mask_boxes = boxlist.pred_boxes.tensor / stride
            box_masks = boxes_to_masks(mask_boxes, m_h, m_w)
            margins = boxlist.margins
            mask_prob = self.compute_mask_prob(self.pixel_embed[im], proposal_embed, margins)
            masks = mask_prob * box_masks.float()
            masks = interpolate(masks, size=(o_h, o_w), mode='bilinear', align_corners=False)
            masks = masks[:, :, :input_h, :input_w].permute(1, 0, 2, 3)
            boxlist.pred_masks = masks

        return boxlists

    def forward_for_single_feature_map(
            self, locations, box_cls,
            reg_pred, ctrness, image_sizes,
            proposal_embeds, margins
    ):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness = ctrness.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness = ctrness.reshape(N, -1).sigmoid()

        proposal_embeds = proposal_embeds.view(N, self.embed_dim, H, W).permute(0, 2, 3, 1)
        proposal_embeds = proposal_embeds.reshape(N, H * W, -1)
        margins = margins.view(N, 1, H, W).permute(0, 2, 3, 1)
        margins = margins.reshape(N, -1)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, None]
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        if not self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            proposal_embeds = proposal_embeds[i][per_box_loc]
            margins = margins[i][per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

                proposal_embeds = proposal_embeds[top_k_indices]
                margins = margins[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations

            boxlist.proposal_embeds = proposal_embeds
            boxlist.margins = margins

            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

    def get_pos_proposal_indexes(self, reg_pred, matched_idxes, targets,
                                 sample_pos_iou_th=0.5):
        locations = torch.cat(self.locations, dim=0)
        N = self.pixel_embed.size(0)
        reg_pred = reg_pred.view(N, -1, 4)  # (N*sum[Hi*Wi, 0<i<fpn_layer], 4) -> (N, ..., 4)
        matched_idxes = matched_idxes.view(N, -1)

        pos_indexes_for_targets = []
        for im_i in range(N):
            targets_per_im = targets[im_i]
            regr = reg_pred[im_i]
            pos_indexes_for_targets_per_im = []
            for t_id in range(len(targets[im_i])):
                valid = matched_idxes[im_i] == t_id
                if valid.sum() == 0:
                    pos_indexes_for_targets_per_im.append(valid.new_tensor([]))
                    continue
                valid_location = locations[valid]
                valid_regression = regr[valid]
                detections = torch.stack([  # xyxy
                    valid_location[:, 0] - valid_regression[:, 0],
                    valid_location[:, 1] - valid_regression[:, 1],
                    valid_location[:, 0] + valid_regression[:, 2],
                    valid_location[:, 1] + valid_regression[:, 3],
                ], dim=1)
                target = targets_per_im.gt_boxes[t_id:t_id + 1].tensor

                match_quality_matrix = iou(detections, target)
                pos_labels_per_target = torch.zeros_like(valid)
                iou_in_target = match_quality_matrix[:, 0]
                if iou_in_target.max() > sample_pos_iou_th:
                    pos_in_target = (iou_in_target > sample_pos_iou_th)
                else:
                    pos_in_target = (iou_in_target == iou_in_target.max())
                pos_labels_per_target[valid] = pos_in_target

                pos_indexes_for_targets_per_im.append(pos_labels_per_target.nonzero().squeeze(1))
            pos_indexes_for_targets.append(pos_indexes_for_targets_per_im)

        return pos_indexes_for_targets

    def get_proposal_element(self, features, poses):
        N, _, dim = features[0].shape
        features_flatten = torch.cat(features, dim=1).contiguous()
        pos_features_for_targets = []
        for im in range(N):
            pos_features_for_targets_im = []
            for t_id in range(len(poses[im])):
                if len(poses[im][t_id]) == 0:
                    pos_features_for_targets_im.append(features_flatten.new_tensor([]))
                else:
                    pos_features_for_targets_im.append(features_flatten[im][poses[im][t_id]])  # [num, dim]
            pos_features_for_targets.append(pos_features_for_targets_im)
        return pos_features_for_targets

    def calculate_means(self, features):
        means = []
        for im in range(len(features)):
            means_im = []
            for t_id in range(len(features[im])):
                if len(features[im][t_id]) == 0:
                    means_im.append(features[im][t_id])
                else:
                    means_im.append(features[im][t_id].mean(dim=0).unsqueeze(0))  # [1, dim]
            means.append(means_im)
        return means

    def fcos_losses(
            self,
            labels,
            reg_targets,
            logits_pred,
            _reg_pred,
            ctrness_pred,

            proposal_embed,  # [(N, Hi*Wi, dim)]
            proposal_margin,  # [(N, Ni*Wi, 1)]

            focal_loss_alpha,
            focal_loss_gamma,
            iou_loss,
            matched_idxes,
    ):
        num_classes = logits_pred.size(1)
        labels = labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        class_loss = sigmoid_focal_loss_jit(
            logits_pred,
            class_target,
            alpha=focal_loss_alpha,
            gamma=focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        reg_pred = _reg_pred[pos_inds]
        reg_targets = reg_targets[pos_inds]
        ctrness_pred = ctrness_pred[pos_inds]

        ctrness_targets = compute_ctrness_targets(reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        ctrness_norm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)

        reg_loss = iou_loss(
            reg_pred,
            reg_targets,
            ctrness_targets
        ) / ctrness_norm

        ctrness_loss = F.binary_cross_entropy_with_logits(
            ctrness_pred,
            ctrness_targets,
            reduction="sum"
        ) / num_pos_avg

        # for embed mask
        pos_proposal_labels_for_targets = self.get_pos_proposal_indexes(_reg_pred, matched_idxes, self.gt_instances)

        proposal_embed_for_targets = self.get_proposal_element(proposal_embed, pos_proposal_labels_for_targets)
        proposal_margin_for_targets = self.get_proposal_element(proposal_margin, pos_proposal_labels_for_targets)

        embedding_means = self.calculate_means(proposal_embed_for_targets)
        margin_means = self.calculate_means(proposal_margin_for_targets)

        # smooth loss
        N, dim, m_h, m_w = self.pixel_embed.shape
        smooth_loss = logits_pred.new_tensor(0.)
        for im in range(N):
            target_num = len(proposal_embed_for_targets[im])
            smooth_loss_im = logits_pred.new_tensor(0.0)
            for t_id in range(target_num):
                smooth_loss_im += \
                    torch.sum((proposal_embed_for_targets[im][t_id] - embedding_means[im][t_id]) ** 2) + \
                    torch.sum((proposal_margin_for_targets[im][t_id] - margin_means[im][t_id]) ** 2)
            if target_num > 0:
                smooth_loss += smooth_loss_im / target_num
        smooth_loss = smooth_loss / N * self.loss_smooth_alpha
        # mask loss
        o_h = m_h * self.mask_scale_factor
        o_w = m_w * self.mask_scale_factor
        r_h = int(m_h * self.strides[0])
        r_w = int(m_w * self.strides[0])
        stride = self.strides[0] / self.mask_scale_factor

        targets_masks = [target_im.gt_masks.tensor for target_im in self.gt_instances]
        targets_boxes = [target_im.gt_boxes.tensor for target_im in self.gt_instances]
        masks_t = prepare_masks(o_h, o_w, r_h, r_w, targets_masks)
        pixel_embed = interpolate(input=self.pixel_embed, size=(o_h, o_w), mode="bilinear", align_corners=False)

        mask_loss = logits_pred.new_tensor(0.0)
        for im in range(N):
            mask_loss_im = logits_pred.new_tensor(0.0)
            target_num = len(proposal_embed_for_targets[im])
            for t_id in range(target_num):
                if len(embedding_means[im][t_id]) == 0:
                    continue
                masks_prob = compute_mask_prob(embedding_means[im][t_id],
                                               margin_means[im][t_id],
                                               pixel_embed[im])
                sample_num = len(masks_prob)
                masks_t_id = masks_t[im][t_id]
                boxes_t_id = targets_boxes[im][t_id] / stride
                masks_prob_crop, crop_mask = crop_by_box(masks_prob, boxes_t_id)
                mask_loss_per_target = self.mask_loss_func(masks_prob_crop,
                                                           masks_t_id.unsqueeze(0).expand(sample_num, -1, -1).float(),
                                                           mask=crop_mask, act=True)
                mask_loss_im += mask_loss_per_target.mean()
            if target_num > 0:
                mask_loss += mask_loss_im / target_num
        mask_loss = mask_loss / N * self.loss_mask_alpha

        losses = {
            "loss_fcos_cls": class_loss,
            "loss_fcos_loc": reg_loss,
            "loss_fcos_ctr": ctrness_loss,
            "loss_mask": mask_loss,
            'loss_smooth': smooth_loss,
        }
        return losses, {}
