"""Tests for POTE (Polarity-aware Linear Attention) components."""

import pytest
import torch
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nn.model.pote import (
    PolarityLinearAttention,
    POTEAttention,
    POTELayer,
    POTEEncoder,
    _elu_feature_map,
)


class TestEluFeatureMap:
    def test_strictly_positive(self):
        x = torch.randn(4, 10, 32)
        out = _elu_feature_map(x)
        assert (out > 0).all(), "ELU+1 feature map must be strictly positive"

    def test_shape_preserved(self):
        x = torch.randn(2, 5, 16)
        assert _elu_feature_map(x).shape == x.shape


class TestPolarityLinearAttention:
    def test_output_shape(self):
        attn = PolarityLinearAttention(embed_dim=32)
        q = torch.randn(2, 10, 32)
        k = torch.randn(2, 15, 32)
        v = torch.randn(2, 15, 32)
        out = attn(q, k, v)
        assert out.shape == q.shape

    def test_self_attention_shape(self):
        attn = PolarityLinearAttention(embed_dim=64)
        x = torch.randn(4, 20, 64)
        out = attn(x, x, x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        attn = PolarityLinearAttention(embed_dim=16)
        q = torch.randn(1, 5, 16, requires_grad=True)
        k = torch.randn(1, 5, 16)
        v = torch.randn(1, 5, 16)
        attn(q, k, v).sum().backward()
        assert q.grad is not None

    def test_no_nan_or_inf(self):
        attn = PolarityLinearAttention(embed_dim=32)
        q = torch.randn(2, 50, 32)
        k = torch.randn(2, 50, 32)
        v = torch.randn(2, 50, 32)
        out = attn(q, k, v)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_polarity_distinguishes_sign(self):
        """Positive and negative inputs should produce different outputs."""
        attn = PolarityLinearAttention(embed_dim=16)
        x_pos = torch.ones(1, 4, 16)
        x_neg = -torch.ones(1, 4, 16)
        out_pos = attn(x_pos, x_pos, x_pos)
        out_neg = attn(x_neg, x_neg, x_neg)
        assert not torch.allclose(out_pos, out_neg), (
            "Polarity attention must differentiate positive/negative inputs"
        )


class TestPOTEAttention:
    def test_output_shape(self):
        pote = POTEAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 20, 64)
        out = pote(x, x, x)
        assert out.shape == x.shape

    def test_cross_attention_shape(self):
        pote = POTEAttention(embed_dim=128, num_heads=8)
        q = torch.randn(2, 10, 128)
        kv = torch.randn(2, 50, 128)
        out = pote(q, kv, kv)
        assert out.shape == q.shape

    def test_gradient_flow(self):
        pote = POTEAttention(embed_dim=32, num_heads=4)
        x = torch.randn(1, 8, 32, requires_grad=True)
        pote(x, x, x).sum().backward()
        assert x.grad is not None

    def test_embed_dim_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            POTEAttention(embed_dim=33, num_heads=8)

    def test_dropout_no_nan(self):
        pote = POTEAttention(embed_dim=32, num_heads=4, dropout=0.1)
        pote.train()
        x = torch.randn(2, 16, 32)
        out = pote(x, x, x)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("heads", [1, 2, 4, 8])
    def test_various_heads(self, heads):
        pote = POTEAttention(embed_dim=64, num_heads=heads)
        x = torch.randn(1, 12, 64)
        out = pote(x, x, x)
        assert out.shape == x.shape


class TestPOTELayer:
    def test_output_shape(self):
        layer = POTELayer(embed_dim=64, num_heads=4, ffn_dim=256)
        x = torch.randn(2, 15, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        layer = POTELayer(embed_dim=32, num_heads=4, ffn_dim=128)
        x = torch.randn(1, 10, 32, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None

    def test_pre_norm_behaviour(self):
        """Output of layer should not be identical to input (transformation occurs)."""
        torch.manual_seed(42)
        layer = POTELayer(embed_dim=32, num_heads=4, ffn_dim=128)
        x = torch.randn(1, 10, 32)
        out = layer(x)
        assert not torch.allclose(x, out, atol=1e-5)


class TestPOTEEncoder:
    def test_single_layer_output_shape(self):
        enc = POTEEncoder(embed_dim=64, num_heads=4, num_layers=1, ffn_dim=256)
        x = torch.randn(2, 100, 64)
        out = enc(x)
        assert out.shape == x.shape

    def test_multi_layer_output_shape(self):
        enc = POTEEncoder(embed_dim=128, num_heads=8, num_layers=3, ffn_dim=512)
        x = torch.randn(2, 64, 128)
        out = enc(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        enc = POTEEncoder(embed_dim=32, num_heads=4, num_layers=2, ffn_dim=128)
        x = torch.randn(1, 25, 32, requires_grad=True)
        enc(x).sum().backward()
        assert x.grad is not None

    def test_no_nan_large_sequence(self):
        enc = POTEEncoder(embed_dim=64, num_heads=4, num_layers=1, ffn_dim=256)
        x = torch.randn(1, 1600, 64)   # 40×40 feature map
        out = enc(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_normalised(self):
        """The final LayerNorm should keep output in a reasonable range."""
        enc = POTEEncoder(embed_dim=32, num_heads=4, num_layers=1, ffn_dim=128)
        enc.eval()
        x = torch.randn(2, 50, 32) * 10   # large input
        out = enc(x)
        # After LayerNorm, std should be roughly 1 per token
        std = out.std(dim=-1)
        assert (std < 5.0).all(), "LayerNorm output std unexpectedly large"
