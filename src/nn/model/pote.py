"""POTE: Polarity-aware linear attention for the FP-DETR transformer encoder.

Standard softmax attention has O(N²) complexity, which is prohibitive for
dense feature maps in real-time detection.  POTE replaces the softmax with a
kernel-based *linear* attention that decomposes the attention matrix into a
product of two lower-rank matrices.

**Polarity decomposition**
Each query/key feature is split into a *positive* (active) and a *negative*
(suppressive) polarity.  The positive polarity attends to salient foreground
tokens (e.g., personnel near LHD vehicles), while the negative polarity helps
suppress background clutter (dust, rock walls).

Formally, for a query q and key k:
    φ(q) = [elu(q)+1,  elu(-q)+1]   (positive, negative half)
    φ(k) = [elu(k)+1,  elu(-k)+1]

Linear attention:
    Attn(Q,K,V) = φ(Q) · (φ(K)ᵀ · V) / (φ(Q) · φ(K)ᵀ · 1)

This keeps the O(N·d) complexity while the polarity split doubles the
effective feature capacity without extra parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 kernel: maps to strictly positive values (needed for linear attn)."""
    return F.elu(x) + 1.0


# ---------------------------------------------------------------------------
# Polarity-aware linear attention
# ---------------------------------------------------------------------------

class PolarityLinearAttention(nn.Module):
    """Single-head polarity-aware linear attention.

    The *polarity decomposition* concatenates the positive and negative ELU
    feature maps, doubling the feature dimension used in the kernel trick.

    Args:
        embed_dim: Dimensionality of query/key/value projections.
        eps:       Numerical stability constant for the normaliser.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q: [B, N_q, D]
            k: [B, N_k, D]
            v: [B, N_k, D]
        Returns:
            out: [B, N_q, D]
        """
        # Polarity feature maps – doubles effective dimension
        phi_q = torch.cat([_elu_feature_map(q), _elu_feature_map(-q)], dim=-1)  # [B, N_q, 2D]
        phi_k = torch.cat([_elu_feature_map(k), _elu_feature_map(-k)], dim=-1)  # [B, N_k, 2D]

        # Linear attention: compute K^T V first → O(N·d²)
        kv = torch.einsum("bnd,bnm->bdm", phi_k, v)   # [B, 2D, D_v]
        # Numerator
        num = torch.einsum("bnd,bdm->bnm", phi_q, kv)  # [B, N_q, D_v]
        # Denominator (normalisation)
        k_sum = phi_k.sum(dim=1)                        # [B, 2D]
        denom = torch.einsum("bnd,bd->bn", phi_q, k_sum).unsqueeze(-1)  # [B, N_q, 1]

        return num / (denom + self.eps)


# ---------------------------------------------------------------------------
# Multi-head POTE
# ---------------------------------------------------------------------------

class POTEAttention(nn.Module):
    """Multi-head POlarity-aware Transformer Encoder attention module.

    Replaces the standard multi-head self/cross attention in the RT-DETR
    encoder with efficient polarity-aware linear attention.

    Args:
        embed_dim:   Total embedding dimension.
        num_heads:   Number of attention heads.
        dropout:     Attention dropout probability.
        eps:         Numerical stability constant.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn = PolarityLinearAttention(self.head_dim, eps=eps)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, D] → [B*H, N, D/H]"""
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B * self.num_heads, N, self.head_dim)

    def _merge_heads(self, x: torch.Tensor, B: int) -> torch.Tensor:
        """[B*H, N, D/H] → [B, N, D]"""
        BH, N, dh = x.shape
        x = x.view(B, self.num_heads, N, dh)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, N, self.embed_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:  [B, N_q, embed_dim]
            key:    [B, N_k, embed_dim]
            value:  [B, N_k, embed_dim]
        Returns:
            out:    [B, N_q, embed_dim]
        """
        B = query.size(0)
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        attn_out = self.attn(q, k, v)            # [B*H, N_q, head_dim]
        attn_out = self.dropout(attn_out)
        out = self._merge_heads(attn_out, B)     # [B, N_q, embed_dim]
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer encoder layer (full POTE layer)
# ---------------------------------------------------------------------------

class POTELayer(nn.Module):
    """A single POTE Transformer encoder layer.

    Follows the standard pre-norm residual design:
        x = x + POTE-Attn(norm(x))
        x = x + FFN(norm(x))

    Args:
        embed_dim:     Model dimension.
        num_heads:     Number of attention heads.
        ffn_dim:       Inner dimension of the feed-forward network.
        dropout:       Dropout applied after attention and FFN.
        act:           Activation used in the FFN (default: GELU).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
        act: str = "gelu",
    ):
        super().__init__()
        self.self_attn = POTEAttention(embed_dim, num_heads, dropout)

        # Feed-forward network
        act_fn = nn.GELU() if act == "gelu" else nn.ReLU(inplace=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, embed_dim]
        Returns:
            x: [B, N, embed_dim]
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)
        x = x + self.drop(self.self_attn(normed, normed, normed))
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Stacked POTE encoder
# ---------------------------------------------------------------------------

class POTEEncoder(nn.Module):
    """Stack of ``num_layers`` POTE encoder layers.

    Args:
        embed_dim: Model dimension.
        num_heads: Number of attention heads per layer.
        num_layers: Number of stacked POTE layers.
        ffn_dim:   Feed-forward inner dimension.
        dropout:   Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [POTELayer(embed_dim, num_heads, ffn_dim, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, embed_dim]   (flattened spatial tokens)
        Returns:
            x: [B, N, embed_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
