"""Training script for FP-DETR.

Usage:
    python tools/train.py --config configs/fp_detr_r50vd.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.core import load_config, merge_config
from src.misc import get_logger
from src.nn.model.fp_detr import FPDETR
from src.nn.criterion.mal_criterion import MALCriterion
from src.data.dataset import CocoDetection
from src.data.transforms import build_train_transforms, build_eval_transforms


logger = get_logger("fp_detr.train")


def build_model(cfg: dict) -> FPDETR:
    m = cfg["model"]
    return FPDETR(
        num_classes=m["num_classes"],
        num_queries=m["num_queries"],
        hidden_dim=m["hidden_dim"],
        num_heads=m["num_heads"],
        num_encoder_layers=m["num_encoder_layers"],
        num_decoder_layers=m["num_decoder_layers"],
        ffn_dim=m["ffn_dim"],
        dropout=m["dropout"],
        backbone_depth=m["backbone_depth"],
        pretrained_backbone=m["pretrained_backbone"],
        num_csp_blocks=m["num_csp_blocks"],
    )


def build_criterion(cfg: dict) -> MALCriterion:
    l = cfg["loss"]
    return MALCriterion(
        num_classes=cfg["model"]["num_classes"],
        lambda_cls=l["lambda_cls"],
        lambda_iou=l["lambda_iou"],
        lambda_l1=l["lambda_l1"],
    )


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def warmup_lr_scheduler(optimizer, warmup_epochs: int, base_lr: float):
    """Linear warmup scheduler (applied per epoch)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(0.01, epoch / warmup_epochs)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, criterion, loader, optimizer, device, epoch, print_freq):
    model.train()
    criterion.train()
    total_loss = 0.0
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = losses["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % print_freq == 0:
            logger.info(
                f"Epoch [{epoch}] step [{i+1}/{len(loader)}] "
                f"loss={loss.item():.4f}"
            )
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser("FP-DETR Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume",  default="", help="Checkpoint to resume from")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    d   = cfg["data"]
    t   = cfg["train"]
    o   = cfg["optimizer"]
    s   = cfg["scheduler"]

    os.makedirs(t["output_dir"], exist_ok=True)
    device = torch.device(args.device)

    # Datasets
    img_size = tuple(d["img_size"])
    train_ds = CocoDetection(
        d["train_img_folder"], d["train_ann_file"],
        transforms=build_train_transforms(img_size),
    )
    val_ds = CocoDetection(
        d["val_img_folder"], d["val_ann_file"],
        transforms=build_eval_transforms(img_size),
    )
    train_loader = DataLoader(
        train_ds, batch_size=d["batch_size"], shuffle=True,
        num_workers=d["num_workers"], collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=d["batch_size"], shuffle=False,
        num_workers=d["num_workers"], collate_fn=collate_fn,
    )

    # Model & criterion
    model     = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=o["lr"],
        weight_decay=o["weight_decay"],
        betas=tuple(o["betas"]),
    )

    # Schedulers
    warmup = warmup_lr_scheduler(optimizer, t["warmup_epochs"], o["lr"])
    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=s["milestones"], gamma=s["gamma"]
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, t["epochs"]):
        avg_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch, t["print_freq"],
        )
        logger.info(f"Epoch [{epoch}] avg_loss={avg_loss:.4f}")

        if epoch < t["warmup_epochs"]:
            warmup.step()
        else:
            main_scheduler.step()

        if (epoch + 1) % t["save_freq"] == 0 or epoch == t["epochs"] - 1:
            ckpt_path = os.path.join(t["output_dir"], f"checkpoint_epoch{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
