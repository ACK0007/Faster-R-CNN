from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.resnet import resnet50
from torchvision.ops import nms, roi_pool
import torch.nn.functional as F
import torch.nn as nn
import torch


class Faster_R_CNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int) -> None:
        super().__init__()
        
        base = resnet50(weights=None)
        self.backbone = create_feature_extractor(base, return_nodes={'layer4': 'C5'})
        self.out_channels = 2048  # resnet50 layer4 channels
        
        
        
        self.RPN = RPN(512,num_anchors)
        
        self.anchor_generator = AnchorGenerator(sizes=((128,256,512)), aspect_ratios=(0.5,1,2))
        
        # Class logits (K + 1 for background); keep raw logits (no Softmax inside)
        self.cls_score = nn.Linear(4096, num_classes + 1)
        # Box regression (class-agnostic here: 4). You can make it 4*num_classes for class-specific.
        self.bbox_pred = nn.Linear(4096, 4)
        
         # ---- RPN proposal settings ----
        self.pre_nms_topk  = 12000
        self.post_nms_topk = 1000
        self.rpn_nms_thresh = 0.7
        self.min_size = 1.0  # discard tiny boxes
        
        self.FC_cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*num_anchors,4096),
            nn.Linear(4096,4096),
            nn.Linear(4096,num_classes+1),
            nn.Softmax(dim=1)
        )
        
        self.FC_reg = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*num_anchors,4096),
            nn.Linear(4096,4096),
            nn.Linear(4096,4)
        )
    
    
    def _compute_stride(self, img_shape, feat_shape):
        # integer stride from image to feature map
        H, W = img_shape
        Hf, Wf = feat_shape
        return (H // Hf, W // Wf)    
    
    def _rpn_inference_single(self, cls_logits, bbox_deltas, anchors, image_size):
        """
        cls_logits: [2A, Hf, Wf]  (for one image)
        bbox_deltas:[4A, Hf, Wf]
        anchors:    [Hf*Wf*A, 4]
        image_size: (H, W)
        returns proposals: [R, 4] and scores: [R]
        """
        AHW = anchors.shape[0]
        # reshape to [AHW, 2] and [AHW, 4]
        cls_logits = cls_logits.permute(1, 2, 0).reshape(-1, 2)    # [Hf,Wf,2A]→[AHW,2]
        bbox_deltas = bbox_deltas.permute(1, 2, 0).reshape(-1, 4)  # [Hf,Wf,4A]→[AHW,4]

        # objectness score = foreground probability
        probs = F.softmax(cls_logits, dim=1)[:, 1]  # [AHW]

        # pre-NMS top-k
        num_topk = min(self.pre_nms_topk, AHW)
        scores, idxs = probs.topk(num_topk, dim=0)
        top_anchors = anchors[idxs]
        top_deltas  = bbox_deltas[idxs]

        # decode + clip
        boxes = decode_boxes(top_deltas, top_anchors)
        H, W = image_size
        boxes = clip_boxes(boxes, H, W)

        # filter tiny boxes
        ws = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        hs = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        keep = (ws >= self.min_size) & (hs >= self.min_size)
        boxes = boxes[keep]
        scores = scores[keep]

        # NMS
        keep_idx = nms(boxes, scores, self.rpn_nms_thresh)
        keep_idx = keep_idx[: self.post_nms_topk]
        return boxes[keep_idx], scores[keep_idx]

        
    def forward(self, images: torch.Tensor):
        """
        images: [N, C, H, W]
        Returns (for inference path):
          - proposals: List[Tensor[R_i, 4]]
          - class_logits: [sum(R_i), K+1]
          - bbox_deltas:  [sum(R_i), 4]
        (Training losses/target assignment omitted in this minimal skeleton.)
        """
        N, _, H, W = images.shape

        # 1) Backbone
        features = self.backbone(images)  # [N, 512, Hf, Wf]
        _, C, Hf, Wf = features.shape

        # 2) Anchors for all images using torchvision AnchorGenerator
        #    (build ImageList so AnchorGenerator can infer strides internally)
        img_list = ImageList(images, [(H, W) for _ in range(N)])
        anchors_per_image = self.anchor_generator(img_list, [features])  # List[N] of [AHW,4]

        # 3) RPN heads
        rpn_cls, rpn_reg = self.RPN(features)  # [N,2A,Hf,Wf], [N,4A,Hf,Wf]

        # 4) RPN inference → proposals per image
        proposals = []
        for i in range(N):
            props, _scores = self._rpn_inference_single(
                rpn_cls[i], rpn_reg[i], anchors_per_image[i], (H, W)
            )
            proposals.append(props)

        # 5) RoIAlign pooled features from backbone feature map
        #    spatial_scale = 1/stride; compute stride from (H,W) and (Hf,Wf)
        sy, sx = self._compute_stride((H, W), (Hf, Wf))
        spatial_scale = 1.0 / float(sx)  # assume square pixels; sx==sy for typical nets

        # roi_align expects a List[Tensor[K_i, 4]] in image coords per image
        pooled_list = []
        for i in range(N):
            if proposals[i].numel() == 0:
                pooled_list.append(torch.zeros((0, C, *self.roi_output_size), device=features.device))
                continue
            # roi_align: input features for the WHOLE batch + per-image boxes tagged with batch indices
            boxes_i = proposals[i]
            # build a single tensor with batch index in front as expected by torchvision <= 0.15 (if needed)
            # newer versions take List[Tensor] directly
            pooled = roi_pool(
                input=features,              # [N, C, Hf, Wf]
                boxes=[boxes_i],             # List[Tensor]
                output_size=(7,7),
                spatial_scale=spatial_scale,
            )  # -> [K_i, C, 7, 7]
            pooled_list.append(pooled)

        pooled_feats = torch.cat(pooled_list, dim=0) if pooled_list else torch.empty(0, C, *self.roi_output_size, device=features.device)
        num_rois = pooled_feats.shape[0]

        if num_rois == 0:
            # nothing proposed → return empties
            return proposals, torch.empty(0, 0, device=features.device), torch.empty(0, 4, device=features.device)

        # 6) Detection heads (per RoI)
        x = pooled_feats.flatten(1)      # [sum(K_i), C*7*7]
        x = self.box_head(x)             # [sum(K_i), hidden]
        class_logits = self.cls_score(x) # [sum(K_i), K+1] (raw logits)
        bbox_deltas  = self.bbox_pred(x) # [sum(K_i), 4]  (class-agnostic here)

        return proposals, class_logits, bbox_deltas
    
    
class RPN(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        
        self.conv_layer = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_conv = nn.Conv2d(512, 2*num_anchors, kernel_size=1, padding=0)
        self.reg_conv = nn.Conv2d(512, 4*num_anchors, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = F.relu(x, inplace=True)
        cls, reg = self.cls_conv(x), self.reg_conv(x)
        return cls, reg
        
def decode_boxes(deltas, anchors):
    """
    deltas:   [M, 4]  (tx, ty, tw, th)
    anchors:  [M, 4]  (x1,y1,x2,y2)
    returns:  [M, 4]  decoded boxes (x1,y1,x2,y2)
    """
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)

    tx, ty, tw, th = deltas.unbind(dim=1)
    px = tx * aw + ax
    py = ty * ah + ay
    pw = aw * torch.exp(tw).clamp(max=1e6)
    ph = ah * torch.exp(th).clamp(max=1e6)

    x1 = px - 0.5 * pw
    y1 = py - 0.5 * ph
    x2 = px + 0.5 * pw
    y2 = py + 0.5 * ph
    return torch.stack([x1, y1, x2, y2], dim=1)


def clip_boxes(boxes, h, w):
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h - 1)
    return boxes
