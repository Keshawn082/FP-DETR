"""CSP-FFCM: Cross Stage Partial Fast Fourier Convolution Module.

Combines a Cross Stage Partial (CSP) bottleneck with Fast Fourier Convolution
(FFC) to fuse spatial and frequency-domain features. This joint representation
improves robustness under dust, uneven lighting, and partial occlusions
commonly found in underground mining environments.

Architecture:
    input  ──► split ──► CSP branch  ──►─┐
                └──────► FFC branch  ──►─┤─► concat ──► 1×1 proj ──► output
                                          │
          (residual shortcut added after projection)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Conv2d → BatchNorm2d → Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard CSP-style bottleneck (1×1 → 3×3 → 1×1)."""

    def __init__(self, channels: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = int(channels * expansion)
        self.cv1 = ConvBNAct(channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden, hidden, 3, padding=1)
        self.cv3 = ConvBNAct(hidden, channels, 1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv3(self.cv2(self.cv1(x)))
        return x + out if self.shortcut else out


# ---------------------------------------------------------------------------
# Fast Fourier Convolution (FFC)
# ---------------------------------------------------------------------------

class SpectralTransform(nn.Module):
    """Process feature maps in the frequency domain via 2-D real FFT.

    Steps:
      1. rFFT2  → complex spectrum  [B, C, H, W//2+1]
      2. Separate real/imag → cat  → 2-channel processing
      3. 1×1 convolution on spectrum
      4. Merge back → irFFT2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # operates on real+imag concatenated → 2*in_channels → out_channels
        self.conv_real = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_real = nn.BatchNorm2d(out_channels)
        self.bn_imag = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # rfft2 → [B, C, H, W//2+1] complex
        x_freq = torch.fft.rfft2(x, norm="ortho")
        x_real = x_freq.real  # [B, C, H, W//2+1]
        x_imag = x_freq.imag

        # convolve real and imaginary parts independently (1×1 on freq grid)
        out_real = F.relu(self.bn_real(self.conv_real(x_real)), inplace=True)
        out_imag = F.relu(self.bn_imag(self.conv_imag(x_imag)), inplace=True)

        # reconstruct complex and irfft
        x_out_freq = torch.complex(out_real, out_imag)
        x_out = torch.fft.irfft2(x_out_freq, s=(H, W), norm="ortho")
        return x_out


class FFCBlock(nn.Module):
    """Fast Fourier Convolution block.

    Splits channels into a local (spatial-conv) path and a global
    (spectral-transform) path, then concatenates results.
    """

    def __init__(self, channels: int, ratio_global: float = 0.5):
        super().__init__()
        g = max(1, int(channels * ratio_global))
        l = channels - g  # local channels

        self.local_conv = ConvBNAct(l, l, 3, padding=1)
        self.global_spec = SpectralTransform(g, g)
        # cross-path interaction
        self.l2g = nn.Conv2d(l, g, 1, bias=False)
        self.g2l = nn.Conv2d(g, l, 1, bias=False)
        self.local_channels = l
        self.global_channels = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l = x[:, : self.local_channels]
        x_g = x[:, self.local_channels :]

        # cross-path signals
        out_l = self.local_conv(x_l) + self.g2l(x_g)
        out_g = self.global_spec(x_g) + F.relu(self.l2g(x_l), inplace=False)
        return torch.cat([out_l, out_g], dim=1)


# ---------------------------------------------------------------------------
# CSP-FFCM
# ---------------------------------------------------------------------------

class CSPFFCM(nn.Module):
    """CSP Fast Fourier Convolution Module.

    The input feature map is split along the channel axis into two streams:
      - **CSP stream**: passes through ``num_blocks`` Bottleneck layers.
      - **FFC stream**: passes through one ``FFCBlock`` for frequency-domain
        global context.

    Both streams are concatenated and projected to ``out_channels`` via a 1×1
    convolution.  A residual connection (with optional projection) bridges the
    module input to the output.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        num_blocks:   Number of CSP Bottleneck blocks (default 3).
        expansion:    Channel expansion ratio inside bottlenecks (default 0.5).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)

        # Initial projections for each stream
        self.stem = ConvBNAct(in_channels, out_channels, 1)
        self.proj_csp = ConvBNAct(out_channels, hidden, 1)
        self.proj_ffc = ConvBNAct(out_channels, hidden, 1)

        # CSP stream: stack of bottleneck blocks
        self.csp_blocks = nn.Sequential(
            *[Bottleneck(hidden, shortcut=True, expansion=1.0)
              for _ in range(num_blocks)]
        )

        # FFC stream
        self.ffc_block = FFCBlock(hidden, ratio_global=0.5)

        # Fusion
        self.fusion = ConvBNAct(hidden * 2, out_channels, 1)

        # Residual (identity or 1×1 projection)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else ConvBNAct(in_channels, out_channels, 1, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)

        feat = self.stem(x)
        csp_out = self.csp_blocks(self.proj_csp(feat))
        ffc_out = self.ffc_block(self.proj_ffc(feat))

        out = self.fusion(torch.cat([csp_out, ffc_out], dim=1))
        return out + identity
