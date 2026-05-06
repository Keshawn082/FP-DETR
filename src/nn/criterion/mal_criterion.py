"""MAL: Multi-scale Adaptive Loss for FP-DETR.

Combines three loss terms:
  1. **Varifocal classification loss** – focuses on uncertain positive
     predictions and suppresses confident negatives (similar to VarifocalLoss
     used in RT-DETR, but with a scale-adaptive weight).
  2. **Scale-adaptive IoU regression loss** – uses GIoU with a per-prediction
     weight inversely proportional to target box area, so small targets in the
     crowded underground scenes receive stronger gradients.
  3. **L1 regression loss** – lightweight coordinate regression penalty.

The three terms are combined as:
    MAL = λ_cls · L_vfl + λ_iou · w_scale · L_giou + λ_l1 · L_l1

where w_scale ∈ (0, 1] is the scale-adaptive weight derived from box area.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


# ---------------------------------------------------------------------------
# Varifocal Loss
# ---------------------------------------------------------------------------

class VarifocalLoss(nn.Module):
    """Varifocal loss (Zhang et al., 2021).

    For positive samples  (IoU-weighted target q > 0):
        L = -q · ( q·log(p) + (1-p^α) · (1-q)·log(1-p) )
    For negative samples  (q = 0):
        L = -p^γ · log(1 - p)

    Args:
        alpha: Weighting for the negative term in positive samples.
        gamma: Focusing parameter for negative samples.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_score: torch.Tensor,
        gt_score: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_score: [N, C]  – sigmoid logits.
            gt_score:   [N, C]  – soft target (IoU for positives, 0 for negatives).
            weight:     [N]     – optional per-sample weight.
        Returns:
            Scalar loss.
        """
        pred_sigmoid = torch.sigmoid(pred_score)
        # Varifocal per-element loss
        pos_mask = gt_score > 0
        loss_pos = -gt_score * (
            gt_score.log() if False  # use BCE form below
            else (
                gt_score * torch.log(pred_sigmoid + 1e-9)
                + (1 - gt_score) * torch.log(1 - pred_sigmoid + 1e-9)
                - self.alpha * (1 - pred_sigmoid) ** self.gamma * torch.log(1 - pred_sigmoid + 1e-9)
            )
        )
        loss_neg = -(pred_sigmoid ** self.gamma) * torch.log(1 - pred_sigmoid + 1e-9)

        loss = torch.where(pos_mask, loss_pos, loss_neg)
        if weight is not None:
            loss = loss * weight.unsqueeze(-1)
        return loss.sum()


# ---------------------------------------------------------------------------
# GIoU Loss (generalised IoU)
# ---------------------------------------------------------------------------

def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Generalised IoU loss (Rezatofighi et al., 2019).

    Args:
        pred_boxes:   [N, 4] in (cx, cy, w, h) format.
        target_boxes: [N, 4] in (cx, cy, w, h) format.
    Returns:
        [N] per-box GIoU loss values.
    """
    # Convert to (x1, y1, x2, y2)
    def cxcywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    pb = cxcywh_to_xyxy(pred_boxes)
    tb = cxcywh_to_xyxy(target_boxes)

    # Intersection
    inter_x1 = torch.maximum(pb[..., 0], tb[..., 0])
    inter_y1 = torch.maximum(pb[..., 1], tb[..., 1])
    inter_x2 = torch.minimum(pb[..., 2], tb[..., 2])
    inter_y2 = torch.minimum(pb[..., 3], tb[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    pred_area = (pb[..., 2] - pb[..., 0]).clamp(min=0) * (pb[..., 3] - pb[..., 1]).clamp(min=0)
    tgt_area  = (tb[..., 2] - tb[..., 0]).clamp(min=0) * (tb[..., 3] - tb[..., 1]).clamp(min=0)
    union_area = pred_area + tgt_area - inter_area + 1e-9

    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.minimum(pb[..., 0], tb[..., 0])
    enc_y1 = torch.minimum(pb[..., 1], tb[..., 1])
    enc_x2 = torch.maximum(pb[..., 2], tb[..., 2])
    enc_y2 = torch.maximum(pb[..., 3], tb[..., 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-9

    giou = iou - (enc_area - union_area) / enc_area
    return 1.0 - giou   # loss ∈ [0, 2]


# ---------------------------------------------------------------------------
# Scale-adaptive weight
# ---------------------------------------------------------------------------

def scale_adaptive_weight(target_boxes: torch.Tensor) -> torch.Tensor:
    """Compute per-box scale weight: smaller boxes → larger weight.

    w = 2 – (w_norm * h_norm)^0.5  ∈ (1, 2]  for boxes with area ∈ (0, 1].

    This gives small targets (common in underground scenes viewed from above)
    stronger supervision signal.

    Args:
        target_boxes: [N, 4] in (cx, cy, w, h) normalised format.
    Returns:
        [N] weight tensor.
    """
    _, _, bw, bh = target_boxes.unbind(-1)
    area = (bw * bh).clamp(min=0, max=1)
    weight = 2.0 - area
    return weight


# ---------------------------------------------------------------------------
# Hungarian / bipartite matching (simple OTA-style assignment)
# ---------------------------------------------------------------------------

def _hungarian_match(
    pred_scores: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> tuple:
    """Greedy cost-based assignment (OTA-simplified) for one image.

    Cost = λ_cls · cost_cls + λ_iou · cost_iou + λ_l1 · cost_l1

    Returns:
        pred_idx: matched prediction indices [M]
        gt_idx:   matched ground-truth  indices [M]
    """
    num_pred = pred_scores.size(0)
    num_gt   = gt_boxes.size(0)
    if num_gt == 0 or num_pred == 0:
        return (
            torch.zeros(0, dtype=torch.long, device=pred_scores.device),
            torch.zeros(0, dtype=torch.long, device=gt_scores.device
                        if hasattr(gt_scores, 'device') else pred_scores.device),
        )

    # Classification cost: focal-style
    alpha, gamma = 0.25, 2.0
    neg_cost = -(1 - pred_scores + 1e-8).log() * (1 - alpha) * (pred_scores ** gamma)
    pos_cost = -(pred_scores + 1e-8).log() * alpha * ((1 - pred_scores) ** gamma)
    # [num_pred, num_gt]
    cost_cls = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]

    # IoU cost
    # expand for pairwise: [num_pred, num_gt]
    pb_exp = pred_boxes.unsqueeze(1).expand(-1, num_gt, -1).reshape(-1, 4)
    gb_exp = gt_boxes.unsqueeze(0).expand(num_pred, -1, -1).reshape(-1, 4)
    cost_iou = giou_loss(pb_exp, gb_exp).reshape(num_pred, num_gt)

    # L1 cost
    cost_l1 = (pred_boxes.unsqueeze(1) - gt_boxes.unsqueeze(0)).abs().sum(-1)

    cost_matrix = 2.0 * cost_cls + 3.0 * cost_iou + 1.0 * cost_l1  # [P, G]

    # Greedy column-wise min (one pred per gt)
    min_vals, pred_idx = cost_matrix.min(dim=0)  # [G]
    gt_idx = torch.arange(num_gt, device=pred_boxes.device)
    return pred_idx, gt_idx


# ---------------------------------------------------------------------------
# MAL Criterion
# ---------------------------------------------------------------------------

class MALCriterion(nn.Module):
    """Multi-scale Adaptive Loss for FP-DETR.

    Applies a set-prediction matching (greedy OTA-style) per image and
    computes the three loss terms with scale-adaptive weighting.

    Args:
        num_classes:  Number of detection categories.
        lambda_cls:   Weight for the varifocal classification loss.
        lambda_iou:   Weight for the scale-adaptive GIoU loss.
        lambda_l1:    Weight for the L1 coordinate loss.
        alpha:        Varifocal alpha.
        gamma:        Varifocal gamma.
    """

    def __init__(
        self,
        num_classes: int,
        lambda_cls: float = 1.0,
        lambda_iou: float = 2.0,
        lambda_l1:  float = 5.0,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls  = lambda_cls
        self.lambda_iou  = lambda_iou
        self.lambda_l1   = lambda_l1
        self.vfl = VarifocalLoss(alpha=alpha, gamma=gamma)

    # ------------------------------------------------------------------
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: dict with
                "pred_logits": [B, num_queries, num_classes]
                "pred_boxes":  [B, num_queries, 4]  (cx,cy,w,h, normalised)
            targets: list of per-image dicts:
                "labels":  [num_gt]  long
                "boxes":   [num_gt, 4]  (cx,cy,w,h, normalised)
        Returns:
            dict of scalar loss tensors.
        """
        pred_logits = outputs["pred_logits"]   # [B, Q, C]
        pred_boxes  = outputs["pred_boxes"]    # [B, Q, 4]
        B = pred_logits.size(0)

        loss_cls_total = pred_logits.new_zeros(1)
        loss_iou_total = pred_logits.new_zeros(1)
        loss_l1_total  = pred_logits.new_zeros(1)
        num_boxes = 0

        for b in range(B):
            gt_labels = targets[b]["labels"]   # [G]
            gt_boxes  = targets[b]["boxes"]    # [G, 4]
            num_gt = gt_labels.size(0)

            scores_b = torch.sigmoid(pred_logits[b])  # [Q, C]
            boxes_b  = pred_boxes[b]                  # [Q, 4]

            if num_gt == 0:
                # no gt → only background loss
                gt_scores = torch.zeros_like(scores_b)
                loss_cls_total = loss_cls_total + self.vfl(
                    pred_logits[b], gt_scores
                ) / max(1, B)
                continue

            # --- matching ---
            pred_idx, gt_idx = _hungarian_match(
                scores_b.detach(),
                boxes_b.detach(),
                gt_labels,
                gt_boxes,
            )

            # --- build targets for matched pairs ---
            matched_pred_boxes = boxes_b[pred_idx]           # [M, 4]
            matched_gt_boxes   = gt_boxes[gt_idx]            # [M, 4]
            matched_gt_labels  = gt_labels[gt_idx]           # [M]

            # IoU between matched pairs (for soft target in vfl)
            with torch.no_grad():
                iou_scores = 1.0 - giou_loss(
                    matched_pred_boxes, matched_gt_boxes
                ).clamp(0, 1)  # [M]

            # Build soft gt_score matrix [Q, C]
            gt_score_mat = torch.zeros_like(scores_b)
            gt_score_mat[pred_idx, matched_gt_labels] = iou_scores

            # Scale-adaptive weight [M]
            sw = scale_adaptive_weight(matched_gt_boxes)

            # --- losses ---
            loss_cls_total = loss_cls_total + self.vfl(pred_logits[b], gt_score_mat)

            loss_giou = (giou_loss(matched_pred_boxes, matched_gt_boxes) * sw).sum()
            loss_iou_total = loss_iou_total + loss_giou

            loss_l1 = (F.l1_loss(
                matched_pred_boxes, matched_gt_boxes, reduction="none"
            ).sum(-1) * sw).sum()
            loss_l1_total = loss_l1_total + loss_l1

            num_boxes += num_gt

        num_boxes = max(1, num_boxes)
        losses = {
            "loss_cls": self.lambda_cls * loss_cls_total / num_boxes,
            "loss_iou": self.lambda_iou * loss_iou_total / num_boxes,
            "loss_l1":  self.lambda_l1  * loss_l1_total  / num_boxes,
        }
        losses["loss_total"] = sum(losses.values())
        return losses
