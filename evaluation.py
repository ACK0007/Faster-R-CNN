# evaluation.py

import json
import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.ops import nms

from COCO import COCODataset, coco_detection_collate

# change this to wherever your model lives
# from model import Faster_R_CNN


# ----- small helpers ----- #

def decode_boxes_from_proposals(deltas: torch.Tensor,
                                proposals: torch.Tensor) -> torch.Tensor:
    """
    deltas:    [R, 4]  (tx,ty,tw,th)
    proposals: [R, 4]  (x1,y1,x2,y2) - in image coords
    returns:   [R, 4]  refined boxes in (x1,y1,x2,y2)
    """
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)

    tx, ty, tw, th = deltas.unbind(dim=1)

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
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h - 1)
    return boxes


def boxes_to_coco_xywh(boxes: torch.Tensor) -> torch.Tensor:
    # (x1,y1,x2,y2) → (x,y,w,h)
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
    """
    proposals:   [R,4]
    class_logits:[R, K+1]
    box_deltas:  [R, 4] (class-agnostic)
    returns dict with 'boxes','scores','labels'
    """
    h, w = image_size
    # 1) softmax to get scores
    probs = torch.softmax(class_logits, dim=1)  # [R, K+1]

    # background is column 0
    scores, labels = probs[:, 1:].max(dim=1)  # [R], [R] in 0..K-1
    labels = labels + 1  # shift to 1..K

    # 2) apply box deltas to proposals
    boxes = decode_boxes_from_proposals(box_deltas, proposals)
    boxes = clip_boxes_to_image(boxes, h, w)

    # 3) threshold
    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if boxes.numel() == 0:
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

    # 4) NMS per class
    final_boxes = []
    final_scores = []
    final_labels = []

    unique_labels = labels.unique()
    for cls in unique_labels:
        cls_mask = labels == cls
        boxes_cls = boxes[cls_mask]
        scores_cls = scores[cls_mask]
        keep_idx = nms(boxes_cls, scores_cls, nms_thresh)
        final_boxes.append(boxes_cls[keep_idx])
        final_scores.append(scores_cls[keep_idx])
        final_labels.append(torch.full((keep_idx.numel(),), cls, dtype=torch.int64, device=labels.device))

    boxes = torch.cat(final_boxes, dim=0)
    scores = torch.cat(final_scores, dim=0)
    labels = torch.cat(final_labels, dim=0)

    # 5) keep top-k
    if boxes.size(0) > detections_per_img:
        topk_scores, topk_idx = scores.topk(detections_per_img)
        boxes = boxes[topk_idx]
        labels = labels[topk_idx]
        scores = topk_scores

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


def run_coco_eval(coco_gt, coco_results_json: str):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(coco_results_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def evaluate(
    model: nn.Module,
    data_source: str,
    year: str = "2017",
    batch_size: int = 1,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_img: int = 100,
    results_out: str = "coco_results.json",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    all_results = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="evaluating", total=len(dataloader)):
            # as before, simplest path assumes batch_size=1
            assert len(images) == 1, "this simple eval assumes batch_size=1"
            img = images[0].to(device)
            target = targets[0]
            img_id = int(target["image_id"].item())
            H, W = img.shape[-2:]

            # 1) backbone
            feats_dict = model.backbone(img.unsqueeze(0))
            features = feats_dict["C5"]  # [1,C,Hf,Wf]
            _, C, Hf, Wf = features.shape

            # 2) anchors
            from torchvision.models.detection.image_list import ImageList
            img_list = ImageList(img.unsqueeze(0), [(H, W)])
            anchors = model.anchor_generator(img_list, [features])[0]

            # 3) RPN
            rpn_cls_map, rpn_reg_map = model.RPN(features)
            rpn_cls_map = rpn_cls_map[0]
            rpn_reg_map = rpn_reg_map[0]

            # 4) proposals (inference path — detach, no grad)
            proposals, _ = model._rpn_inference_single(
                rpn_cls_map,
                rpn_reg_map,
                anchors,
                (H, W),
            )

            # 5) ROI pooling
            spatial_scale = float(Wf) / float(W)
            from torchvision.ops import roi_pool
            pooled = roi_pool(
                features,
                [proposals],
                output_size=model.roi_output_size,
                spatial_scale=spatial_scale,
            )  # [R, C, 7, 7]

            if pooled.numel() == 0:
                continue

            x = pooled.flatten(1)
            x = model.box_head(x)
            class_logits = model.cls_score(x)  # [R, K+1]
            bbox_deltas = model.bbox_pred(x)   # [R, 4] (class-agnostic)

            # 6) postprocess into final boxes
            preds = postprocess_single_image(
                proposals=proposals,
                class_logits=class_logits,
                box_deltas=bbox_deltas,
                image_size=(H, W),
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                detections_per_img=detections_per_img,
            )

            # 7) convert to COCO format (x,y,w,h) and map labels back
            # dataset has contig_to_cat_id: {1: coco_id1, 2: coco_id2 ...}
            for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
                box_xywh = boxes_to_coco_xywh(box.unsqueeze(0))[0].tolist()
                coco_cat_id = dataset.contig_to_cat_id[int(label.item())]
                all_results.append(
                    {
                        "image_id": img_id,
                        "category_id": coco_cat_id,
                        "bbox": [float(x) for x in box_xywh],
                        "score": float(score.item()),
                    }
                )

    # write JSON
    with open(results_out, "w") as f:
        json.dump(all_results, f)

    # run COCO eval
    # dataset.coco is the GT pycocotools object because we loaded mode="val"
    run_coco_eval(dataset.coco, results_out)


if __name__ == "__main__":
    # example usage:
    # from model import Faster_R_CNN
    # model = Faster_R_CNN(in_channels=3, num_classes=80)
    # model.load_state_dict(torch.load("ckpt.pth"))
    # evaluate(model, "/path/to/coco2017")
    pass
