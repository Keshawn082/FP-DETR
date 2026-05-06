"""Tests for CSP-FFCM (CSP Fast Fourier Convolution Module)."""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nn.model.csp_ffcm import (
    ConvBNAct,
    Bottleneck,
    SpectralTransform,
    FFCBlock,
    CSPFFCM,
)


class TestConvBNAct:
    def test_output_shape(self):
        m = ConvBNAct(32, 64, kernel_size=3, padding=1)
        x = torch.randn(2, 32, 16, 16)
        out = m(x)
        assert out.shape == (2, 64, 16, 16)

    def test_no_act(self):
        m = ConvBNAct(16, 16, act=False)
        x = torch.randn(1, 16, 8, 8)
        out = m(x)
        assert out.shape == (1, 16, 8, 8)

    def test_stride(self):
        m = ConvBNAct(32, 64, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 32, 32, 32)
        out = m(x)
        assert out.shape == (2, 64, 16, 16)


class TestBottleneck:
    def test_shortcut_preserves_shape(self):
        m = Bottleneck(64, shortcut=True)
        x = torch.randn(2, 64, 16, 16)
        out = m(x)
        assert out.shape == x.shape

    def test_no_shortcut(self):
        m = Bottleneck(64, shortcut=False)
        x = torch.randn(1, 64, 8, 8)
        out = m(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        m = Bottleneck(32, shortcut=True)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None


class TestSpectralTransform:
    def test_output_shape_square(self):
        m = SpectralTransform(32, 32)
        x = torch.randn(2, 32, 16, 16)
        out = m(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_output_shape_non_square(self):
        m = SpectralTransform(16, 16)
        x = torch.randn(2, 16, 20, 30)
        out = m(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        m = SpectralTransform(16, 16)
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None

    @pytest.mark.parametrize("batch", [1, 4])
    def test_various_batch_sizes(self, batch):
        m = SpectralTransform(8, 8)
        x = torch.randn(batch, 8, 12, 12)
        assert m(x).shape == x.shape


class TestFFCBlock:
    def test_output_shape(self):
        m = FFCBlock(64)
        x = torch.randn(2, 64, 16, 16)
        out = m(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        m = FFCBlock(32)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestCSPFFCM:
    def test_same_channels(self):
        """in_channels == out_channels – identity residual."""
        m = CSPFFCM(64, 64, num_blocks=2)
        x = torch.randn(2, 64, 20, 20)
        out = m(x)
        assert out.shape == (2, 64, 20, 20)

    def test_different_channels(self):
        """in_channels != out_channels – projected residual."""
        m = CSPFFCM(128, 256, num_blocks=1)
        x = torch.randn(2, 128, 16, 16)
        out = m(x)
        assert out.shape == (2, 256, 16, 16)

    def test_gradient_flow(self):
        m = CSPFFCM(32, 64, num_blocks=1)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_num_blocks_zero(self):
        """Works with 0 CSP blocks (only FFC path active)."""
        m = CSPFFCM(32, 32, num_blocks=0)
        x = torch.randn(1, 32, 8, 8)
        out = m(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("in_c,out_c,blocks", [
        (64, 64, 1),
        (128, 64, 2),
        (256, 128, 3),
    ])
    def test_parametric(self, in_c, out_c, blocks):
        m = CSPFFCM(in_c, out_c, num_blocks=blocks)
        x = torch.randn(1, in_c, 16, 16)
        out = m(x)
        assert out.shape == (1, out_c, 16, 16)

    def test_frequency_features_used(self):
        """Output should differ from a pure spatial-only residual."""
        torch.manual_seed(0)
        m = CSPFFCM(32, 32, num_blocks=1)
        m.eval()
        x1 = torch.randn(1, 32, 16, 16)
        # perturb in frequency: add high-frequency component
        x2 = x1.clone()
        x2[:, 0, ::2, ::2] += 0.5
        out1 = m(x1)
        out2 = m(x2)
        assert not torch.allclose(out1, out2), (
            "Outputs are identical – frequency branch may not be contributing"
        )
