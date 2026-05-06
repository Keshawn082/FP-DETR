"""Evaluation / inference script for FP-DETR.

Usage:
    python tools/eval.py --config configs/fp_detr_r50vd.yaml \\
                         --checkpoint outputs/fp_detr_r50/checkpoint_epoch0119.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.core import load_config
from src.misc import get_logger
from src.nn.model.fp_detr import FPDETR
from src.data.dataset import CocoDetection
from src.data.transforms import build_eval_transforms


logger = get_logger("fp_detr.eval")


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


@torch.no_grad()
def evaluate(model, loader, device, score_threshold: float = 0.3):
    """Run inference and collect predictions."""
    model.eval()
    all_predictions = []

    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)

        pred_logits = outputs["pred_logits"]   # [B, Q, C]
        pred_boxes  = outputs["pred_boxes"]    # [B, Q, 4]

        scores = torch.sigmoid(pred_logits)    # [B, Q, C]
        max_scores, pred_labels = scores.max(dim=-1)  # [B, Q]

        for b in range(images.size(0)):
            keep = max_scores[b] >= score_threshold
            all_predictions.append({
                "image_id": targets[b]["image_id"].item(),
                "scores":   max_scores[b][keep].cpu(),
                "labels":   pred_labels[b][keep].cpu(),
                "boxes":    pred_boxes[b][keep].cpu(),
            })

    logger.info(f"Evaluated {len(all_predictions)} images.")
    return all_predictions


def main():
    parser = argparse.ArgumentParser("FP-DETR Evaluation")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--score_thr",  type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    m   = cfg["model"]
    d   = cfg["data"]

    device   = torch.device(args.device)
    img_size = tuple(d["img_size"])

    val_ds = CocoDetection(
        d["val_img_folder"], d["val_ann_file"],
        transforms=build_eval_transforms(img_size),
    )
    val_loader = DataLoader(
        val_ds, batch_size=d["batch_size"], shuffle=False,
        num_workers=d["num_workers"], collate_fn=collate_fn,
    )

    model = FPDETR(
        num_classes=m["num_classes"],
        num_queries=m["num_queries"],
        hidden_dim=m["hidden_dim"],
        num_heads=m["num_heads"],
        num_encoder_layers=m["num_encoder_layers"],
        num_decoder_layers=m["num_decoder_layers"],
        ffn_dim=m["ffn_dim"],
        dropout=m["dropout"],
        backbone_depth=m["backbone_depth"],
        pretrained_backbone=False,
        num_csp_blocks=m["num_csp_blocks"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    preds = evaluate(model, val_loader, device, args.score_thr)
    logger.info(f"Total detections: {sum(len(p['scores']) for p in preds)}")


if __name__ == "__main__":
    main()
