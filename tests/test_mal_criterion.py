"""Tests for MAL (Multi-scale Adaptive Loss) criterion."""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nn.criterion.mal_criterion import (
    VarifocalLoss,
    giou_loss,
    scale_adaptive_weight,
    MALCriterion,
)


class TestVarifocalLoss:
    def test_output_is_scalar(self):
        vfl = VarifocalLoss()
        pred = torch.randn(10, 5)
        gt   = torch.rand(10, 5)
        loss = vfl(pred, gt)
        assert loss.ndim == 0

    def test_loss_non_negative(self):
        vfl = VarifocalLoss()
        pred = torch.randn(20, 3)
        gt   = torch.zeros(20, 3)
        loss = vfl(pred, gt)
        assert loss.item() >= 0.0

    def test_zero_gt_score(self):
        """All-negative gt should yield a finite positive loss."""
        vfl = VarifocalLoss()
        pred = torch.zeros(5, 2)
        gt   = torch.zeros(5, 2)
        loss = vfl(pred, gt)
        assert torch.isfinite(loss)

    def test_perfect_prediction_lower_loss(self):
        """Higher-confidence correct predictions should give lower loss."""
        vfl = VarifocalLoss()
        # Perfect: predicted high scores for positive slots
        gt   = torch.zeros(4, 1)
        gt[0, 0] = 1.0   # one positive
        pred_good = torch.tensor([[5.0], [-5.0], [-5.0], [-5.0]])
        pred_bad  = torch.tensor([[-5.0], [5.0], [5.0], [5.0]])
        loss_good = vfl(pred_good, gt).item()
        loss_bad  = vfl(pred_bad,  gt).item()
        assert loss_good < loss_bad

    def test_gradient_flows(self):
        vfl = VarifocalLoss()
        pred = torch.randn(6, 3, requires_grad=True)
        gt   = torch.rand(6, 3)
        vfl(pred, gt).backward()
        assert pred.grad is not None


class TestGiouLoss:
    def test_identical_boxes_zero_loss(self):
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        loss  = giou_loss(boxes, boxes)
        assert loss.abs().item() < 1e-5

    def test_non_overlapping_boxes(self):
        pred = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        tgt  = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
        loss = giou_loss(pred, tgt)
        # GIoU loss in [0, 2]
        assert 0.0 <= loss.item() <= 2.0

    def test_output_shape(self):
        pred = torch.rand(8, 4)
        tgt  = torch.rand(8, 4)
        loss = giou_loss(pred, tgt)
        assert loss.shape == (8,)

    def test_batch_mode(self):
        pred = torch.rand(16, 4)
        tgt  = torch.rand(16, 4)
        loss = giou_loss(pred, tgt)
        assert loss.numel() == 16

    def test_gradient_flows(self):
        pred = torch.rand(4, 4, requires_grad=True)
        tgt  = torch.rand(4, 4)
        giou_loss(pred, tgt).sum().backward()
        assert pred.grad is not None


class TestScaleAdaptiveWeight:
    def test_small_boxes_larger_weight(self):
        small = torch.tensor([[0.5, 0.5, 0.05, 0.05]])  # area ≈ 0.0025
        large = torch.tensor([[0.5, 0.5, 0.5,  0.5]])   # area = 0.25
        w_small = scale_adaptive_weight(small)
        w_large = scale_adaptive_weight(large)
        assert w_small.item() > w_large.item()

    def test_weight_range(self):
        boxes = torch.rand(20, 4).clamp(0.01, 0.99)
        w = scale_adaptive_weight(boxes)
        assert (w >= 1.0).all()
        assert (w <= 2.0).all()

    def test_output_shape(self):
        boxes = torch.rand(10, 4)
        w = scale_adaptive_weight(boxes)
        assert w.shape == (10,)


class TestMALCriterion:
    def _make_outputs(self, B=2, Q=50, C=1):
        return {
            "pred_logits": torch.randn(B, Q, C),
            "pred_boxes":  torch.rand(B, Q, 4),
        }

    def _make_targets(self, B=2, num_gt=3, C=1):
        targets = []
        for _ in range(B):
            targets.append({
                "labels": torch.randint(0, C, (num_gt,)),
                "boxes":  torch.rand(num_gt, 4),
            })
        return targets

    def test_returns_expected_keys(self):
        criterion = MALCriterion(num_classes=1)
        losses = criterion(self._make_outputs(), self._make_targets())
        for key in ("loss_cls", "loss_iou", "loss_l1", "loss_total"):
            assert key in losses, f"Missing key: {key}"

    def test_all_losses_finite(self):
        criterion = MALCriterion(num_classes=2)
        outputs = self._make_outputs(C=2)
        targets = self._make_targets(C=2)
        losses  = criterion(outputs, targets)
        for k, v in losses.items():
            assert torch.isfinite(v), f"Loss {k} is not finite: {v}"

    def test_total_is_sum(self):
        criterion = MALCriterion(num_classes=1)
        losses = criterion(self._make_outputs(), self._make_targets())
        expected = losses["loss_cls"] + losses["loss_iou"] + losses["loss_l1"]
        assert torch.allclose(losses["loss_total"], expected, atol=1e-5)

    def test_empty_targets(self):
        """Images with no ground-truth boxes should still yield finite losses."""
        criterion = MALCriterion(num_classes=1)
        outputs = self._make_outputs(B=1)
        targets = [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}]
        losses  = criterion(outputs, targets)
        for k, v in losses.items():
            assert torch.isfinite(v), f"Loss {k} is not finite: {v}"

    def test_gradient_flows(self):
        criterion = MALCriterion(num_classes=1)
        pred_logits = torch.randn(2, 50, 1, requires_grad=True)
        pred_boxes  = torch.rand(2, 50, 4, requires_grad=True)
        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        targets = self._make_targets()
        losses  = criterion(outputs, targets)
        losses["loss_total"].backward()
        assert pred_logits.grad is not None
        assert pred_boxes.grad is not None

    def test_non_negative_losses(self):
        criterion = MALCriterion(num_classes=1)
        losses = criterion(self._make_outputs(), self._make_targets())
        for k, v in losses.items():
            assert v.item() >= 0.0, f"Loss {k} is negative: {v.item()}"

    @pytest.mark.parametrize("num_classes", [1, 3, 10])
    def test_multi_class(self, num_classes):
        criterion = MALCriterion(num_classes=num_classes)
        outputs = self._make_outputs(C=num_classes)
        targets = self._make_targets(C=num_classes)
        losses  = criterion(outputs, targets)
        assert torch.isfinite(losses["loss_total"])
