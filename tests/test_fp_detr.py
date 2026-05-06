"""Integration tests for the full FP-DETR model."""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nn.model.fp_detr import FPDETR, MLP
from src.nn.model.hybrid_encoder import HybridEncoder
from src.nn.criterion.mal_criterion import MALCriterion


# Use a small model to keep tests fast
SMALL_CFG = dict(
    num_classes=1,
    num_queries=10,
    hidden_dim=32,
    num_heads=4,
    num_encoder_layers=1,
    num_decoder_layers=2,
    ffn_dim=64,
    dropout=0.0,
    backbone_depth=18,
    pretrained_backbone=False,
    num_csp_blocks=1,
)


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(in_dim=64, hidden_dim=128, out_dim=4, num_layers=3)
        x = torch.randn(5, 64)
        assert mlp(x).shape == (5, 4)

    def test_gradient_flow(self):
        mlp = MLP(in_dim=32, hidden_dim=64, out_dim=8, num_layers=2)
        x = torch.randn(3, 32, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None


class TestHybridEncoder:
    def test_output_shapes(self):
        in_channels = [128, 256, 512]
        enc = HybridEncoder(in_channels=in_channels, hidden_dim=32,
                            num_blocks=1, num_heads=4, num_encoder_layers=1,
                            ffn_dim=64)
        feats = [
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20),
            torch.randn(2, 512, 10, 10),
        ]
        outs = enc(feats)
        assert len(outs) == len(feats)
        assert outs[0].shape == (2, 32, 40, 40)
        assert outs[1].shape == (2, 32, 20, 20)
        assert outs[2].shape == (2, 32, 10, 10)

    def test_gradient_flow(self):
        in_channels = [128, 256, 512]
        enc = HybridEncoder(in_channels=in_channels, hidden_dim=32,
                            num_blocks=1, num_heads=4, num_encoder_layers=1,
                            ffn_dim=64)
        feats = [
            torch.randn(1, c, 8, 8, requires_grad=True)
            for c in in_channels
        ]
        outs = enc(feats)
        sum(o.sum() for o in outs).backward()
        for f in feats:
            assert f.grad is not None


class TestFPDETR:
    def _make_model(self):
        return FPDETR(**SMALL_CFG)

    def test_forward_output_keys(self):
        model = self._make_model()
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert "pred_logits" in out
        assert "pred_boxes"  in out

    def test_pred_logits_shape(self):
        model = self._make_model()
        B = 2
        x = torch.randn(B, 3, 64, 64)
        out = model(x)
        assert out["pred_logits"].shape == (B, SMALL_CFG["num_queries"], SMALL_CFG["num_classes"])

    def test_pred_boxes_shape(self):
        model = self._make_model()
        B = 2
        x = torch.randn(B, 3, 64, 64)
        out = model(x)
        assert out["pred_boxes"].shape == (B, SMALL_CFG["num_queries"], 4)

    def test_pred_boxes_in_0_1(self):
        """Sigmoid-activated box predictions should lie in [0, 1]."""
        model = self._make_model()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        boxes = out["pred_boxes"]
        assert (boxes >= 0.0).all()
        assert (boxes <= 1.0).all()

    def test_gradient_flow_through_full_model(self):
        model = self._make_model()
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        out["pred_logits"].sum().backward()
        assert x.grad is not None

    def test_no_nan_in_output(self):
        model = self._make_model()
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in {k}"
            assert not torch.isinf(v).any(), f"Inf in {k}"

    def test_eval_mode_deterministic(self):
        """In eval mode with same input the output should be identical."""
        model = self._make_model()
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1["pred_logits"], out2["pred_logits"])
        assert torch.allclose(out1["pred_boxes"],  out2["pred_boxes"])

    def test_batch_size_one(self):
        model = self._make_model()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out["pred_logits"].shape[0] == 1

    def test_batch_size_four(self):
        model = self._make_model()
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out["pred_logits"].shape[0] == 4

    def test_parameter_count_with_r18(self):
        """R-18 backbone model should have a reasonable parameter count."""
        model = self._make_model()
        n_params = sum(p.numel() for p in model.parameters())
        # Sanity: should be < 100M with our small config
        assert n_params < 100_000_000, f"Model has {n_params:,} parameters (unexpectedly large)"


class TestFPDETRWithLoss:
    def _make_targets(self, B, Q, device="cpu"):
        targets = []
        for _ in range(B):
            num_gt = 3
            targets.append({
                "labels": torch.zeros(num_gt, dtype=torch.long, device=device),
                "boxes":  torch.rand(num_gt, 4, device=device),
            })
        return targets

    def test_end_to_end_loss(self):
        model     = FPDETR(**SMALL_CFG)
        criterion = MALCriterion(num_classes=SMALL_CFG["num_classes"])
        x         = torch.randn(2, 3, 64, 64)
        targets   = self._make_targets(2, SMALL_CFG["num_queries"])
        outputs   = model(x)
        losses    = criterion(outputs, targets)
        assert torch.isfinite(losses["loss_total"])

    def test_backward_through_loss(self):
        model     = FPDETR(**SMALL_CFG)
        criterion = MALCriterion(num_classes=SMALL_CFG["num_classes"])
        x         = torch.randn(2, 3, 64, 64)
        targets   = self._make_targets(2, SMALL_CFG["num_queries"])
        outputs   = model(x)
        losses    = criterion(outputs, targets)
        losses["loss_total"].backward()
        # At least some backbone parameters should have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed during backward pass"
