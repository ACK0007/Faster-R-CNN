import json
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.ops import nms, roi_pool
from torchvision.models.detection.image_list import ImageList

from COCO import COCODataset, coco_detection_collate

import wandb


def decode_boxes_from_proposals(deltas: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    tx, ty, tw, th = deltas.unbind(1)
    cx = tx * pw + px
    cy = ty * ph + py
    w = pw * torch.exp(tw).clamp(max=1e6)
    h = ph * torch.exp(th).clamp(max=1e6)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)


def clip_boxes_to_image(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
    return boxes


def boxes_to_coco_xywh(boxes: torch.Tensor) -> torch.Tensor:
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    x = boxes[:, 0]
    y = boxes[:, 1]
    return torch.stack([x, y, w, h], dim=1)


def postprocess_single_image(
    proposals: torch.Tensor,
    class_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    image_size: tuple,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_img: int = 100,
) -> Dict[str, torch.Tensor]:
    h, w = image_size
    probs = torch.softmax(class_logits, dim=1)  # [R, K+1]
    scores, labels = probs[:, 1:].max(dim=1)
    labels = labels + 1  # 1..K
    boxes = decode_boxes_from_proposals(box_deltas, proposals)
    boxes = clip_boxes_to_image(boxes, h, w)

    keep = scores >= score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.numel() == 0:
        return {"boxes": boxes, "scores": scores, "labels": labels}

    final_boxes, final_scores, final_labels = [], [], []
    for cls in labels.unique():
        m = labels == cls
        keep_idx = nms(boxes[m], scores[m], nms_thresh)
        final_boxes.append(boxes[m][keep_idx])
        final_scores.append(scores[m][keep_idx])
        final_labels.append(torch.full((keep_idx.numel(),), cls, dtype=torch.int64, device=labels.device))

    boxes = torch.cat(final_boxes, dim=0)
    scores = torch.cat(final_scores, dim=0)
    labels = torch.cat(final_labels, dim=0)

    if boxes.size(0) > detections_per_img:
        topk = scores.topk(detections_per_img)
        boxes, scores, labels = boxes[topk.indices], topk.values, labels[topk.indices]
    return {"boxes": boxes, "scores": scores, "labels": labels}


def run_coco_eval(coco_gt, coco_results_json: str):
    from pycocotools.cocoeval import COCOeval
    coco_dt = coco_gt.loadRes(coco_results_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats  # 12-element array


def evaluate(
    model: nn.Module,
    data_source: str,
    year: str = "2017",
    batch_size: int = 1,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_img: int = 100,
    results_out: str = "coco_results.json",
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

    model.to(device)
    model.eval()

    dataset = COCODataset(data_source, mode="val", year=year)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=coco_detection_collate,
    )

    # ---- NEW: wandb init for eval run ----
    if use_wandb:
        wandb.init(project=project, name=run_name, job_type="eval", config={
            "year": year,
            "batch_size": batch_size,
            "score_thresh": score_thresh,
            "nms_thresh": nms_thresh,
            "detections_per_img": detections_per_img,
        })

    all_results = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="evaluating", total=len(dataloader)):
            assert len(images) == 1, "simple eval assumes batch_size=1"
            img = images[0].to(device)
            target = targets[0]
            img_id = int(target["image_id"].item())
            H, W = img.shape[-2:]

            feats = model.backbone(img.unsqueeze(0))["C5"]
            _, C, Hf, Wf = feats.shape

            img_list = ImageList(img.unsqueeze(0), [(H, W)])
            anchors = model.anchor_generator(img_list, [feats])[0]

            rpn_cls_map, rpn_reg_map = model.RPN(feats)
            proposals, _ = model._rpn_inference_single(
                rpn_cls_map[0], rpn_reg_map[0], anchors, (H, W)
            )

            spatial_scale = float(Wf) / float(W)
            pooled = roi_pool(feats, [proposals], output_size=(7, 7), spatial_scale=spatial_scale)
            if pooled.numel() == 0:
                continue

            x = pooled.flatten(1)
            x = model.box_head(x)
            class_logits = model.cls_score(x)
            bbox_deltas = model.bbox_pred(x)

            preds = postprocess_single_image(
                proposals=proposals,
                class_logits=class_logits,
                box_deltas=bbox_deltas,
                image_size=(H, W),
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                detections_per_img=detections_per_img,
            )

            for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
                xywh = boxes_to_coco_xywh(box.unsqueeze(0))[0].tolist()
                coco_cat_id = dataset.contig_to_cat_id[int(label.item())]
                all_results.append({
                    "image_id": img_id,
                    "category_id": coco_cat_id,
                    "bbox": [float(x) for x in xywh],
                    "score": float(score.item()),
                })

            # ---- OPTIONAL: log a few prediction visuals to wandb ----
            if use_wandb and (len(all_results) % 500 == 0):
                # convert to CHW tensor -> HWC numpy
                img_np = (img.cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
                boxes_xywh = boxes_to_coco_xywh(preds["boxes"].cpu()).tolist()
                box_data = []
                for (x, y, w, h), s, lab in zip(boxes_xywh, preds["scores"].cpu().tolist(), preds["labels"].cpu().tolist()):
                    box_data.append({
                        "position": {"minX": x, "minY": y, "maxX": x + w, "maxY": y + h},
                        "class_id": int(lab),
                        "scores": {"score": float(s)},
                    })
                wandb.log({
                    "predictions": wandb.Image(img_np, boxes={
                        "preds": {"box_data": box_data, "class_labels": {}}
                    })
                })

    with open(results_out, "w") as f:
        json.dump(all_results, f)

    stats = run_coco_eval(dataset.coco, results_out)  # array length 12

    # ---- NEW: log COCO metrics to wandb ----
    if use_wandb:
        keys = [
            "AP@[.50:.95]", "AP@0.50", "AP@0.75",
            "AP_small", "AP_medium", "AP_large",
            "AR@1", "AR@10", "AR@100",
            "AR_small", "AR_medium", "AR_large",
        ]
        wandb.log({f"coco/{k}": float(v) for k, v in zip(keys, stats)})
        wandb.save(results_out)