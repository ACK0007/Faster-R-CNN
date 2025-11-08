from torch.utils.data import DataLoader
from COCO import COCODataset, coco_detection_collate
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torch
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import roi_pool


# ---------- helper functions (same as before) ----------
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def encode_boxes(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)

    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)
    return torch.stack([tx, ty, tw, th], dim=1)


def rpn_assign_targets(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    high_thresh: float = 0.7,
    low_thresh: float = 0.3,
    batch_size_per_img: int = 256,
    positive_fraction: float = 0.5,
):
    num_anchors = anchors.shape[0]
    labels = anchors.new_full((num_anchors,), -1, dtype=torch.int64)

    if gt_boxes.numel() == 0:
        labels.fill_(0)
        return labels, anchors.new_zeros((num_anchors, 4))

    ious = box_iou(anchors, gt_boxes)  # [A,G]
    max_iou, max_iou_idx = ious.max(dim=1)

    labels[max_iou >= high_thresh] = 1
    labels[max_iou < low_thresh] = 0

    # ensure every GT has a positive
    max_iou_per_gt, _ = ious.max(dim=0)
    for g, v in enumerate(max_iou_per_gt):
        if v == 0:
            continue
        anchors_for_gt = (ious[:, g] == v).nonzero(as_tuple=False).view(-1)
        labels[anchors_for_gt] = 1

    bbox_targets = anchors.new_zeros((num_anchors, 4))
    pos_mask = labels == 1
    if pos_mask.any():
        matched_gt = gt_boxes[max_iou_idx[pos_mask]]
        bbox_targets[pos_mask] = encode_boxes(matched_gt, anchors[pos_mask])

    # subsample
    num_pos = int(batch_size_per_img * positive_fraction)
    pos_idx = torch.nonzero(labels == 1).view(-1)
    if pos_idx.numel() > num_pos:
        disable = pos_idx[torch.randperm(pos_idx.numel())[: pos_idx.numel() - num_pos]]
        labels[disable] = -1

    num_neg = batch_size_per_img - (labels == 1).sum().item()
    neg_idx = torch.nonzero(labels == 0).view(-1)
    if neg_idx.numel() > num_neg:
        disable = neg_idx[torch.randperm(neg_idx.numel())[: neg_idx.numel() - num_neg]]
        labels[disable] = -1

    return labels, bbox_targets


def roi_assign_targets(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    fg_iou_thresh: float = 0.5,
    bg_iou_thresh: float = 0.1,
    batch_size_per_img: int = 128,
    positive_fraction: float = 0.25,
):
    if gt_boxes.numel() == 0:
        num_bg = min(batch_size_per_img, proposals.size(0))
        idx = torch.randperm(proposals.size(0))[:num_bg]
        return proposals[idx], proposals.new_zeros((num_bg,), dtype=torch.int64), proposals.new_zeros((num_bg, 4))

    ious = box_iou(proposals, gt_boxes)
    max_iou, max_iou_idx = ious.max(dim=1)

    labels = proposals.new_zeros((proposals.size(0),), dtype=torch.int64)
    pos_mask = max_iou >= fg_iou_thresh
    labels[pos_mask] = gt_labels[max_iou_idx[pos_mask]]

    # subsample positives
    num_pos = int(batch_size_per_img * positive_fraction)
    pos_idx = torch.nonzero(pos_mask).view(-1)
    if pos_idx.numel() > num_pos:
        disable = pos_idx[torch.randperm(pos_idx.numel())[: pos_idx.numel() - num_pos]]
        labels[disable] = -1
        pos_mask = labels > 0

    num_pos = pos_mask.sum().item()
    num_neg = batch_size_per_img - num_pos

    neg_idx = torch.nonzero(labels == 0).view(-1)
    if neg_idx.numel() > num_neg:
        disable = neg_idx[torch.randperm(neg_idx.numel())[: neg_idx.numel() - num_neg]]
        labels[disable] = -1

    keep = labels >= 0
    sampled_props = proposals[keep]
    sampled_labels = labels[keep]

    bbox_targets = sampled_props.new_zeros((sampled_props.size(0), 4))
    pos_keep = sampled_labels > 0
    if pos_keep.any():
        matched_gt = gt_boxes[max_iou_idx[keep][pos_keep]]
        bbox_targets[pos_keep] = encode_boxes(matched_gt, sampled_props[pos_keep])

    return sampled_props, sampled_labels, bbox_targets


# ---------- main training function ----------
def train(model: nn.Module, data_source: str, batch_size: int = 1, epochs: int = 10, lr: float = 0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model.to(device)
    model.train()

    dataset = COCODataset(data_source, mode="train", year="2017")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=coco_detection_collate,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        running = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}", total=len(loader))
        for images, targets in pbar:
            # simple version assumes batch_size=1
            assert len(images) == 1
            img = images[0].to(device)
            target = {k: v.to(device) for k, v in targets[0].items()}

            # 1) backbone
            feats_dict = model.backbone(img.unsqueeze(0))
            features = feats_dict["C5"]          # [1, 2048, Hf, Wf]
            _, C, Hf, Wf = features.shape

            # 2) anchors
            N, _, H, W = img.unsqueeze(0).shape
            img_list = ImageList(img.unsqueeze(0), [(H, W)])
            anchors = model.anchor_generator(img_list, [features])[0]  # [A,4]

            # 3) RPN once
            rpn_cls_map, rpn_reg_map = model.RPN(features)  # [1,2A,Hf,Wf], [1,4A,Hf,Wf]
            rpn_cls_map_img = rpn_cls_map[0]
            rpn_reg_map_img = rpn_reg_map[0]

            # 4) reshape for RPN loss
            rpn_cls = rpn_cls_map_img.permute(1, 2, 0).reshape(-1, 2)  # [A,2]
            rpn_reg = rpn_reg_map_img.permute(1, 2, 0).reshape(-1, 4)  # [A,4]

            # 5) RPN targets & losses
            rpn_labels, rpn_box_targets = rpn_assign_targets(anchors, target["boxes"])
            valid = rpn_labels >= 0
            rpn_cls_loss = F.cross_entropy(rpn_cls[valid], rpn_labels[valid])

            pos = rpn_labels == 1
            if pos.any():
                rpn_reg_loss = F.smooth_l1_loss(
                    rpn_reg[pos],
                    rpn_box_targets[pos],
                    reduction="sum"
                ) / pos.sum().clamp(min=1).float()
            else:
                rpn_reg_loss = torch.tensor(0.0, device=device)

            # 6) proposals — ***DETACH*** RPN maps before inference
            proposals, _scores = model._rpn_inference_single(
                rpn_cls_map_img.detach(),   # no grad through proposal path
                rpn_reg_map_img.detach(),
                anchors,
                (H, W),
            )
            proposals = proposals.detach()

            # 7) ROI target assignment
            rois, roi_labels, roi_box_targets = roi_assign_targets(
                proposals, target["boxes"], target["labels"]
            )

            # 8) ROI pooling
            spatial_scale = float(Wf) / float(W)
            pooled = roi_pool(features, [rois], output_size=model.roi_output_size,
                              spatial_scale=spatial_scale)  # [M,C,7,7]
            if pooled.numel() == 0:
                continue

            x = pooled.flatten(1)
            x = model.box_head(x)
            cls_logits = model.cls_score(x)
            bbox_preds = model.bbox_pred(x)

            roi_cls_loss = F.cross_entropy(cls_logits, roi_labels)
            pos_roi = roi_labels > 0
            if pos_roi.any():
                roi_reg_loss = F.smooth_l1_loss(
                    bbox_preds[pos_roi],
                    roi_box_targets[pos_roi],
                    reduction="sum"
                ) / pos_roi.sum().clamp(min=1).float()
            else:
                roi_reg_loss = torch.tensor(0.0, device=device)

            loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

        print(f"epoch {epoch} avg loss: {running / len(loader):.4f}")
