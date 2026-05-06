"""Image augmentation / transforms for FP-DETR training and evaluation.

Transforms operate on (PIL.Image, target_dict) pairs.  Bounding boxes in
target are expected in normalised (cx, cy, w, h) format.
"""

import random
from typing import Tuple, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class Compose:
    """Chain multiple (image, target) transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Flip image and boxes horizontally with probability *p*."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = TF.hflip(image)
            if "boxes" in target and target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                # cx_new = 1 - cx  (cy, w, h unchanged)
                boxes[:, 0] = 1.0 - boxes[:, 0]
                target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    """Flip image and boxes vertically with probability *p*."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = TF.vflip(image)
            if "boxes" in target and target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, 1] = 1.0 - boxes[:, 1]
                target["boxes"] = boxes
        return image, target


class Resize:
    """Resize image to a fixed size (height, width).

    Boxes are in normalised format, so no rescaling needed.
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (H, W)

    def __call__(self, image, target):
        image = TF.resize(image, self.size)
        return image, target


class ColorJitter:
    """Random colour jitter (brightness, contrast, saturation, hue)."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        import torchvision.transforms as T
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        image = self.jitter(image)
        return image, target


class ToTensor:
    """Convert PIL Image to float tensor and normalise to [0, 1]."""

    def __call__(self, image, target):
        return TF.to_tensor(image), target


class Normalize:
    """Normalise image tensor with ImageNet mean/std."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std:  Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = mean
        self.std  = std

    def __call__(self, image, target):
        image = TF.normalize(image, self.mean, self.std)
        return image, target


# ---------------------------------------------------------------------------
# Pre-built transform sets
# ---------------------------------------------------------------------------

def build_train_transforms(img_size: Tuple[int, int] = (640, 640)) -> Compose:
    return Compose([
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        Resize(img_size),
        ToTensor(),
        Normalize(),
    ])


def build_eval_transforms(img_size: Tuple[int, int] = (640, 640)) -> Compose:
    return Compose([
        Resize(img_size),
        ToTensor(),
        Normalize(),
    ])
