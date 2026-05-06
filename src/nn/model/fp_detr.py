"""FP-DETR: Full model definition.

Architecture overview:
    Image → PResNet backbone → HybridEncoder (CSP-FFCM + POTE) →
    Transformer Decoder → Detection head (class logits + box regression)

The decoder is a standard DETR-style transformer decoder with learnable
object queries.  Prediction heads are lightweight MLPs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.presnet import PResNet
from .hybrid_encoder import HybridEncoder
from .pote import POTEAttention


# ---------------------------------------------------------------------------
# Helper MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple fully-connected network (used as prediction head)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """One DETR-style decoder layer using POTE cross-attention.

    Self-attention on queries + POTE cross-attention with encoder memory +
    FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn   = POTEAttention(embed_dim, num_heads, dropout)
        self.cross_attn  = POTEAttention(embed_dim, num_heads, dropout)

        act = nn.GELU()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), act,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tgt:    [B, num_queries, embed_dim]
            memory: [B, N_enc, embed_dim]
        Returns:
            tgt:    [B, num_queries, embed_dim]
        """
        # Self-attention
        t = self.norm1(tgt)
        tgt = tgt + self.drop(self.self_attn(t, t, t))
        # Cross-attention with encoder memory
        t = self.norm2(tgt)
        tgt = tgt + self.drop(self.cross_attn(t, memory, memory))
        # FFN
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt


# ---------------------------------------------------------------------------
# Full FP-DETR model
# ---------------------------------------------------------------------------

class FPDETR(nn.Module):
    """FP-DETR: Real-time surrounding personnel detection for LHD vehicles.

    Key components:
    - PResNet backbone (depth 18 or 50).
    - HybridEncoder with CSP-FFCM (spatial-frequency fusion) and POTE
      (polarity-aware linear attention).
    - Standard DETR transformer decoder with POTE cross-attention.
    - Lightweight MLP prediction heads.

    Args:
        num_classes:      Number of detection categories (e.g. 1 for "person").
        num_queries:      Number of learnable object queries (default 300).
        hidden_dim:       Transformer/encoder embedding dimension (default 256).
        num_heads:        Attention heads (default 8).
        num_encoder_layers: POTE encoder depth (default 1).
        num_decoder_layers: Decoder depth (default 6).
        ffn_dim:          Feed-forward inner dimension (default 1024).
        dropout:          Dropout probability (default 0.0).
        backbone_depth:   ResNet depth — 18 or 50 (default 50).
        pretrained_backbone: Load ImageNet backbone weights (default True).
        num_csp_blocks:   CSP-FFCM bottleneck blocks per scale (default 3).
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_queries: int = 300,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        backbone_depth: int = 50,
        pretrained_backbone: bool = True,
        num_csp_blocks: int = 3,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.num_queries  = num_queries
        self.hidden_dim   = hidden_dim

        # Backbone
        self.backbone = PResNet(
            depth=backbone_depth,
            pretrained=pretrained_backbone,
            freeze_at=0,
            return_idx=(1, 2, 3),
        )
        in_channels = self.backbone.out_channels  # e.g. [512, 1024, 2048] for R-50

        # Hybrid encoder
        self.encoder = HybridEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=num_csp_blocks,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # Decoder queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head  = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Initialise prediction head weights."""
        nn.init.constant_(self.class_head.bias, -math.log((1 - 0.01) / 0.01))
        nn.init.xavier_uniform_(self.bbox_head.net[0].weight)
        nn.init.xavier_uniform_(self.bbox_head.net[2].weight)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, 3, H, W] input image batch.
        Returns:
            dict with keys:
                "pred_logits": [B, num_queries, num_classes]
                "pred_boxes":  [B, num_queries, 4]  (cx,cy,w,h, normalised)
        """
        B = x.size(0)

        # Backbone
        features = self.backbone(x)   # list of [B, C_i, H_i, W_i]

        # Hybrid encoder
        enc_feats = self.encoder(features)  # list of [B, hidden_dim, H_i, W_i]

        # Flatten all scales and concatenate as encoder memory
        memory_tokens = []
        for feat in enc_feats:
            tokens = feat.flatten(2).transpose(1, 2)  # [B, H_i*W_i, hidden_dim]
            memory_tokens.append(tokens)
        memory = torch.cat(memory_tokens, dim=1)  # [B, sum(H_i*W_i), hidden_dim]

        # Decoder
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)
        tgt = self.decoder_norm(tgt)

        # Prediction
        pred_logits = self.class_head(tgt)                     # [B, Q, num_classes]
        pred_boxes  = self.bbox_head(tgt).sigmoid()            # [B, Q, 4]

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
