from typing import Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models.detection.image_list import ImageList
from torchvision.ops import roi_pool

from COCO import COCODataset, coco_detection_collate

import wandb


# ---------------------------
# Box / target helper functions
# ---------------------------

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
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
    iou_high: float = 0.7,
    iou_low: float = 0.3,
    batch_size_per_img: int = 256,
    positive_fraction: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    A = anchors.size(0)
    labels = anchors.new_full((A,), -1, dtype=torch.int64)
    bbox_targets = anchors.new_zeros((A, 4))

    if gt_boxes.numel() == 0:
        labels.fill_(0)
        return labels, bbox_targets

    ious = box_iou(anchors, gt_boxes)
    max_iou, max_idx = ious.max(dim=1)
    labels[max_iou >= iou_high] = 1
    labels[max_iou < iou_low] = 0

    max_iou_per_gt, _ = ious.max(dim=0)
    for g, v in enumerate(max_iou_per_gt):
        if v <= 0:
            continue
        best = (ious[:, g] == v).nonzero(as_tuple=False).view(-1)
        labels[best] = 1

    pos_mask = labels == 1
    if pos_mask.any():
        matched_gt = gt_boxes[max_idx[pos_mask]]
        bbox_targets[pos_mask] = encode_boxes(matched_gt, anchors[pos_mask])

    num_pos = int(batch_size_per_img * positive_fraction)
    pos_idx = torch.nonzero(labels == 1).view(-1)
    if pos_idx.numel() > num_pos:
        disable = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[: pos_idx.numel() - num_pos]]
        labels[disable] = -1

    num_neg = batch_size_per_img - (labels == 1).sum().item()
    neg_idx = torch.nonzero(labels == 0).view(-1)
    if neg_idx.numel() > num_neg:
        disable = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[: neg_idx.numel() - num_neg]]
        labels[disable] = -1

    return labels, bbox_targets


def roi_assign_targets(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    fg_iou_thresh: float = 0.5,
    batch_size_per_img: int = 128,
    positive_fraction: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if proposals.numel() == 0:
        return proposals, proposals.new_zeros((0,), dtype=torch.int64), proposals.new_zeros((0, 4))
    if gt_boxes.numel() == 0:
        num_bg = min(batch_size_per_img, proposals.size(0))
        keep = torch.randperm(proposals.size(0), device=proposals.device)[:num_bg]
        return proposals[keep], proposals.new_zeros((num_bg,), dtype=torch.int64), proposals.new_zeros((num_bg, 4))

    ious = box_iou(proposals, gt_boxes)
    max_iou, max_idx = ious.max(dim=1)

    pos_mask = max_iou >= fg_iou_thresh
    labels = proposals.new_zeros((proposals.size(0),), dtype=torch.int64)
    labels[pos_mask] = gt_labels[max_idx[pos_mask]]

    num_pos = int(batch_size_per_img * positive_fraction)
    pos_idx = torch.nonzero(pos_mask).view(-1)
    if pos_idx.numel() > num_pos:
        disable = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[: pos_idx.numel() - num_pos]]
        labels[disable] = -1
        pos_mask = labels > 0

    num_pos = pos_mask.sum().item()
    num_neg = batch_size_per_img - num_pos
    neg_idx = torch.nonzero(labels == 0).view(-1)
    if neg_idx.numel() > num_neg:
        disable = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[: neg_idx.numel() - num_neg]]
        labels[disable] = -1

    keep = labels >= 0
    sampled_props = proposals[keep]
    sampled_labels = labels[keep]

    bbox_targets = sampled_props.new_zeros((sampled_props.size(0), 4))
    pos_keep = sampled_labels > 0
    if pos_keep.any():
        matched_gt = gt_boxes[max_idx[keep][pos_keep]]
        bbox_targets[pos_keep] = encode_boxes(matched_gt, sampled_props[pos_keep])

    return sampled_props, sampled_labels, bbox_targets


# ---------------------------
# Training
# ---------------------------

def train(
    model: nn.Module,
    data_source: str,
    batch_size: int = 1,
    epochs: int = 12,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
    grad_clip_norm: float = 5.0,
    use_wandb: bool = True,
    project: str = "fasterrcnn",
    run_name: str = None,
):
    # Prefer MPS on Apple Silicon, then CUDA, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model.to(device)
    model.train()
    model.backbone.eval()     # BN stability with small batches

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    train_ds = COCODataset(data_source, mode="train", year="2017")
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=coco_detection_collate,
    )

    # ---- NEW: wandb init ----
    if use_wandb:
        wandb.init(
            project=project,
            name=run_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "freeze_backbone": freeze_backbone,
                "grad_clip_norm": grad_clip_norm,
                "num_classes": getattr(train_ds, "num_classes", None),
                "device": str(device),
            },
        )
        # Optional (can be heavy): gradients every N steps
        # wandb.watch(model, log="gradients", log_freq=200)

    global_step = 0

    for epoch in range(epochs):
        running = 0.0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}", total=len(train_dl))

        for images, targets in pbar:
            assert len(images) == 1, "This trainer assumes batch_size=1."
            img = images[0].to(device)
            target = {k: v.to(device) for k, v in targets[0].items()}

            # 1) Backbone
            feats_dict = model.backbone(img.unsqueeze(0))
            features = feats_dict["C5"]
            _, C, Hf, Wf = features.shape

            # 2) Anchors
            H, W = img.shape[-2:]
            img_list = ImageList(img.unsqueeze(0), [(H, W)])
            anchors = model.anchor_generator(img_list, [features])[0]

            # 3) RPN forward (once)
            rpn_cls_map, rpn_reg_map = model.RPN(features)
            rpn_cls_map_img = rpn_cls_map[0]
            rpn_reg_map_img = rpn_reg_map[0]

            # 4) RPN losses
            rpn_cls = rpn_cls_map_img.permute(1, 2, 0).reshape(-1, 2)
            rpn_reg = rpn_reg_map_img.permute(1, 2, 0).reshape(-1, 4)

            rpn_labels, rpn_box_targets = rpn_assign_targets(anchors, target["boxes"])
            rpn_box_targets = rpn_box_targets.clamp(-10.0, 10.0)

            valid = rpn_labels >= 0
            rpn_cls_loss = F.cross_entropy(rpn_cls[valid], rpn_labels[valid]) if valid.any() else torch.tensor(0.0, device=device)

            pos = rpn_labels == 1
            if pos.any():
                pred = rpn_reg[pos].clamp(-10.0, 10.0)
                tgt = rpn_box_targets[pos]
                rpn_reg_loss = F.smooth_l1_loss(pred, tgt, reduction="sum") / pos.sum().clamp(min=1).float()
            else:
                rpn_reg_loss = torch.tensor(0.0, device=device)

            # 5) Proposals (detached)
            proposals, _scores = model._rpn_inference_single(
                rpn_cls_map_img.detach(),
                rpn_reg_map_img.detach(),
                anchors,
                (H, W),
            )
            proposals = proposals.detach()

            # 6) RoI targets
            rois, roi_labels, roi_box_targets = roi_assign_targets(
                proposals, target["boxes"], target["labels"]
            )
            roi_box_targets = roi_box_targets.clamp(-10.0, 10.0)

            # 7) RoI pooling
            spatial_scale = float(Wf) / float(W)
            pooled = roi_pool(features, [rois], output_size=(7, 7), spatial_scale=spatial_scale)
            if pooled.numel() == 0:
                continue

            x = pooled.flatten(1)
            x = model.box_head(x)
            cls_logits = model.cls_score(x)
            bbox_preds = model.bbox_pred(x)

            # 8) RoI losses
            roi_cls_loss = F.cross_entropy(cls_logits, roi_labels) if roi_labels.numel() > 0 else torch.tensor(0.0, device=device)

            pos_roi = roi_labels > 0
            if pos_roi.any():
                pred = bbox_preds[pos_roi].clamp(-10.0, 10.0)
                tgt = roi_box_targets[pos_roi]
                roi_reg_loss = F.smooth_l1_loss(pred, tgt, reduction="sum") / pos_roi.sum().clamp(min=1).float()
            else:
                roi_reg_loss = torch.tensor(0.0, device=device)

            loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            optimizer.step()

            running += loss.item()
            global_step += 1

            # ---- NEW: wandb step logging ----
            if use_wandb:
                wandb.log({
                    "loss/total": float(loss.item()),
                    "loss/rpn_cls": float(rpn_cls_loss.item()),
                    "loss/rpn_reg": float(rpn_reg_loss.item()),
                    "loss/roi_cls": float(roi_cls_loss.item()),
                    "loss/roi_reg": float(roi_reg_loss.item()),
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

            pbar.set_postfix(loss=float(loss.item()))

        avg = running / max(1, len(train_dl))
        print(f"epoch {epoch}: avg_loss={avg:.4f}")

        # ---- NEW: wandb epoch summary + checkpoint ----
        if use_wandb:
            wandb.log({"epoch/avg_loss": avg, "epoch": epoch}, step=global_step)
            ckpt_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({"epoch": epoch + 1, "model": model.state_dict()}, ckpt_path)
            wandb.save(ckpt_path)  # uploads as run file