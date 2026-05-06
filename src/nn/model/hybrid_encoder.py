"""Hybrid encoder for FP-DETR.

The encoder sits between the backbone (which provides multi-scale feature
maps) and the DETR decoder.  It consists of:

  1. **Channel projection**: a 1×1 ConvBNAct per scale to project to a
     common ``hidden_dim``.
  2. **CSP-FFCM blocks**: applied at each scale to fuse spatial and
     frequency features.
  3. **Feature pyramid fusion**: top-down + bottom-up FPN-style paths.
  4. **POTE encoder**: flattens the fused features and applies the
     polarity-aware linear attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .csp_ffcm import CSPFFCM, ConvBNAct
from .pote import POTEEncoder


class HybridEncoder(nn.Module):
    """Multi-scale hybrid encoder with CSP-FFCM and POTE.

    Args:
        in_channels:    List of input channel counts from the backbone
                        (e.g. [512, 1024, 2048] for ResNet-50).
        hidden_dim:     Common channel dimension after projection.
        num_blocks:     CSP-FFCM bottleneck blocks per scale.
        num_heads:      Attention heads in the POTE encoder.
        num_encoder_layers: Number of stacked POTE encoder layers.
        ffn_dim:        FFN inner dimension in POTE layers.
        dropout:        Dropout probability.
    """

    def __init__(
        self,
        in_channels: list,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        num_heads: int = 8,
        num_encoder_layers: int = 1,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = len(in_channels)

        # 1. Channel projection per scale
        self.proj_layers = nn.ModuleList([
            ConvBNAct(c, hidden_dim, 1) for c in in_channels
        ])

        # 2. CSP-FFCM at each scale
        self.csp_ffcm_layers = nn.ModuleList([
            CSPFFCM(hidden_dim, hidden_dim, num_blocks=num_blocks)
            for _ in in_channels
        ])

        # 3a. Top-down path (high res → low res context propagation)
        self.td_convs = nn.ModuleList([
            ConvBNAct(hidden_dim, hidden_dim, 3, padding=1)
            for _ in range(self.num_scales - 1)
        ])
        # 3b. Bottom-up path (low res context → high res detail)
        self.bu_convs = nn.ModuleList([
            ConvBNAct(hidden_dim, hidden_dim, 3, stride=2, padding=1)
            for _ in range(self.num_scales - 1)
        ])

        # 4. POTE encoder on the highest-resolution scale only
        #    (other scales are encoded by smaller POTE or skipped for speed)
        self.pote_encoder = POTEEncoder(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, features: list) -> list:
        """
        Args:
            features: list of [B, C_i, H_i, W_i] tensors from backbone,
                      ordered from finest to coarsest scale.
        Returns:
            out_features: list of [B, hidden_dim, H_i, W_i] tensors,
                          refined by the encoder.
        """
        # 1. Project channels
        feats = [proj(f) for proj, f in zip(self.proj_layers, features)]

        # 2. Apply CSP-FFCM at each scale
        feats = [csp(f) for csp, f in zip(self.csp_ffcm_layers, feats)]

        # 3a. Top-down: from coarse to fine
        for i in range(self.num_scales - 1, 0, -1):
            upsampled = F.interpolate(
                feats[i], size=feats[i - 1].shape[-2:], mode="nearest"
            )
            feats[i - 1] = self.td_convs[i - 1](feats[i - 1] + upsampled)

        # 3b. Bottom-up: from fine to coarse
        for i in range(self.num_scales - 1):
            downsampled = self.bu_convs[i](feats[i])
            # Ensure spatial size matches in case of odd-dimension rounding
            if downsampled.shape[-2:] != feats[i + 1].shape[-2:]:
                downsampled = F.interpolate(
                    downsampled, size=feats[i + 1].shape[-2:], mode="nearest"
                )
            feats[i + 1] = feats[i + 1] + downsampled

        # 4. Apply POTE on the finest scale feature map
        B, C, H, W = feats[0].shape
        tokens = feats[0].flatten(2).transpose(1, 2)   # [B, H*W, C]
        tokens = self.pote_encoder(tokens)              # [B, H*W, C]
        feats[0] = tokens.transpose(1, 2).view(B, C, H, W)

        return feats
