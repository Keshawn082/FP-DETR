"""COCO-style dataset and dataloader utilities for FP-DETR."""

import os
import json
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class CocoDetection(Dataset):
    """COCO-format detection dataset.

    Args:
        img_folder:  Path to image directory.
        ann_file:    Path to COCO-format annotation JSON.
        transforms:  Optional callable applied to (image, target) pairs.
    """

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
    ):
        self.img_folder = img_folder
        self.transforms = transforms

        with open(ann_file) as f:
            coco = json.load(f)

        # Build index: image_id → annotations
        self.images = coco["images"]
        ann_by_img = {}
        for ann in coco.get("annotations", []):
            ann_by_img.setdefault(ann["image_id"], []).append(ann)
        self.ann_by_img = ann_by_img

        # Map category id → contiguous class index
        cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
        self.cat2idx = {c["id"]: i for i, c in enumerate(cats)}
        self.num_classes = len(cats)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_info = self.images[idx]
        img_path = os.path.join(self.img_folder, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        anns = self.ann_by_img.get(img_info["id"], [])
        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]  # COCO: (x, y, w, h) in pixels
            # normalise and convert to (cx, cy, w, h)
            cx = (x + bw / 2) / W
            cy = (y + bh / 2) / H
            nw = bw / W
            nh = bh / H
            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat2idx[ann["category_id"]])

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([img_info["id"]]),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
