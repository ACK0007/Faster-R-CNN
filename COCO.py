from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.v2 as T  # torch>=2.1
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
from pycocotools import coco as COCO
from PIL import Image
import zipfile
import torch
import os


class COCODataset(Dataset):
    def __init__(
        self,
        data_source: str,
        mode: str = "train",
        year: str = "2017",
        transforms: Optional[Callable] = None,
        keep_crowd: bool = False,
    ) -> None:
        
        super().__init__()
        assert mode in {"train", "val", "test"}, "mode must be 'train', 'val', or 'test'"

        self.data_source = data_source
        self.mode = mode
        self.year = year
        self.transforms = transforms
        self.keep_crowd = keep_crowd

        # Resolve image dir and annotation file
        img_dir = {
            "train": f"train{year}",
            "val": f"val{year}",
            "test": f"test{year}",
        }[mode]
        self.img_root = os.path.join(self.data_source, img_dir)

        if mode in {"train", "val"}:
            ann_file = os.path.join(
                self.data_source, "annotations", f"instances_{mode}{year}.json"
            )
            if not os.path.isfile(ann_file):
                raise FileNotFoundError(f"Missing annotations file: {ann_file}")
            self.coco = COCO.COCO(ann_file)
            self.ids = list(self.coco.imgs.keys())
            # Build a mapping from COCO category ids -> contiguous [1..K]
            cat_ids = sorted(self.coco.cats.keys())
            self.cat_id_to_contig = {cid: i + 1 for i, cid in enumerate(cat_ids)}
            self.contig_to_cat_id = {v: k for k, v in self.cat_id_to_contig.items()}
            self.num_classes = len(self.cat_id_to_contig) + 1  # +1 for background id 0
        else:
            # test has no labels
            self.coco = None
            # Use filenames in the test folder as image list
            self.ids = sorted([
                int(os.path.splitext(f)[0])
                for f in os.listdir(self.img_root)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            self.cat_id_to_contig = {}
            self.contig_to_cat_id = {}
            self.num_classes = 0  # unknown for test

        if not os.path.isdir(self.img_root):
            raise FileNotFoundError(f"Images folder not found: {self.img_root}")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, img_id: int) -> Image.Image:
        if self.coco is not None:
            img_info = self.coco.loadImgs(img_id)[0]
            path = os.path.join(self.img_root, img_info["file_name"])
        else:
            # test mode: filename is typically {id}.jpg
            path_jpg = os.path.join(self.img_root, f"{img_id:012d}.jpg")
            path_png = os.path.join(self.img_root, f"{img_id:012d}.png")
            path = path_jpg if os.path.exists(path_jpg) else path_png
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        # Always load as RGB
        return Image.open(path).convert("RGB")

    def _build_target(self, img_id: int) -> Dict[str, torch.Tensor]:
        # For test mode: return empty target
        if self.coco is None:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([img_id], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.uint8),
            }

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        area: List[float] = []
        iscrowd: List[int] = []

        for a in anns:
            if (not self.keep_crowd) and a.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_contig[a["category_id"]])
            area.append(a.get("area", float(w * h)))
            iscrowd.append(a.get("iscrowd", 0))

        target: Dict[str, torch.Tensor] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.uint8),
        }
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        img_id = self.ids[index]
        img = self._load_image(img_id)
        target = self._build_target(img_id)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def coco_detection_collate(batch):
    """Collate function for detection models: returns lists instead of stacked tensors."""
    return tuple(zip(*batch))
