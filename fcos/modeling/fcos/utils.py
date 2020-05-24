import torch
from detectron2.layers import interpolate


def iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def compute_mask_prob(proposal_embed, proposal_margin, pixel_embed, fix_margin=False):
    m_h, m_w = pixel_embed.shape[-2:]  # [D, H, W]
    obj_num = proposal_embed.shape[0]
    pixel_embed = pixel_embed.permute(1, 2, 0).unsqueeze(0).expand(obj_num, -1, -1, -1)  # [m, H, W, D]
    proposal_embed = proposal_embed.view(obj_num, 1, 1, -1).expand(-1, m_h, m_w, -1)
    if fix_margin:
        init_margin = 1  # todo add to __init__
        proposal_margin = proposal_margin.new_ones(obj_num, m_h, m_w) * init_margin
    else:
        proposal_margin = proposal_margin.view(obj_num, 1, 1).expand(-1, m_h, m_w)
    mask_var = torch.sum((pixel_embed - proposal_embed) ** 2, dim=3)
    mask_prob = torch.exp(-mask_var * proposal_margin)

    return mask_prob


def prepare_masks(o_h, o_w, r_h, r_w, targets_masks):
    masks = []
    for im_i in range(len(targets_masks)):
        mask_t = targets_masks[im_i]
        if len(mask_t) == 0:
            masks.append(mask_t.new_tensor([]))
            continue
        n, h, w = mask_t.shape
        mask = mask_t.new_zeros((n, r_h, r_w))
        mask[:, :h, :w] = mask_t
        resized_mask = interpolate(
            input=mask.float().unsqueeze(0), size=(o_h, o_w), mode="bilinear", align_corners=False,
        )[0].gt(0)

        masks.append(resized_mask)
    return masks


def crop_by_box(masks, box, padding=0.0):
    n, h, w = masks.size()

    b_w = box[2] - box[0]
    b_h = box[3] - box[1]
    x1 = torch.clamp(box[0:1] - b_w * padding - 1, min=0)
    x2 = torch.clamp(box[2:3] + b_w * padding + 1, max=w - 1)
    y1 = torch.clamp(box[1:2] - b_h * padding - 1, min=0)
    y2 = torch.clamp(box[3:4] + b_h * padding + 1, max=h - 1)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(n, h, w)

    masks_left = rows >= x1.expand(n, 1, 1)
    masks_right = rows < x2.expand(n, 1, 1)
    masks_up = cols >= y1.expand(n, 1, 1)
    masks_down = cols < y2.expand(n, 1, 1)

    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float(), crop_mask


def boxes_to_masks(boxes, h, w, padding=0.0):
    n = boxes.shape[0]
    boxes = boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    b_w = x2 - x1
    b_h = y2 - y1
    x1 = torch.clamp(x1 - 1 - b_w * padding, min=0)
    x2 = torch.clamp(x2 + 1 + b_w * padding, max=w)
    y1 = torch.clamp(y1 - 1 - b_h * padding, min=0)
    y2 = torch.clamp(y2 + 1 + b_h * padding, max=h)

    rows = torch.arange(w, device=boxes.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
    cols = torch.arange(h, device=boxes.device, dtype=x1.dtype).view(1, -1, 1).expand(n, h, w)

    masks_left = rows >= x1.view(-1, 1, 1)
    masks_right = rows < x2.view(-1, 1, 1)
    masks_up = cols >= y1.view(-1, 1, 1)
    masks_down = cols < y2.view(-1, 1, 1)

    masks = masks_left * masks_right * masks_up * masks_down

    return masks
