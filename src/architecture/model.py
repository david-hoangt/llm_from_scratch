"""
A dense decoder-only LLM built from the components of this series.

Assembles: RMSNorm + QK-Norm (post 4), RoPE (post 3), Grouped-Query Attention
(post 2), SwiGLU FFN (post 5) into the architecture of Qwen3-0.6B, so that
published weights can be loaded into it directly.

Conventions here deliberately match the HuggingFace reference implementation:
  - RoPE rotates SPLIT HALVES (dim i pairs with i + head_dim/2), not interleaved pairs
  - RMSNorm computes in float32 and casts back before applying the weight
  - QK-Norm normalizes over head_dim, and runs BEFORE RoPE
  - Grouped-Query Attention repeats each KV head for CONSECUTIVE query heads
Any one of these being wrong loads without error and generates nonsense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    """Qwen3-0.6B. Every field is read straight from the published config.json."""

    d_model: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8            # GQA: 2 query heads per KV head
    head_dim: int = 128            # decoupled: 16 * 128 = 2048 != d_model
    d_ff: int = 3072               # 3.0 * d_model
    vocab_size: int = 151936
    norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_embeddings: bool = True

    @property
    def n_rep(self) -> int:
        """How many query heads share one KV head."""
        return self.n_heads // self.n_kv_heads


class RMSNorm(nn.Module):
    """
    y = gamma * x / sqrt(mean(x^2) + eps)

    Args:
        dim: Size of the normalized dimension (d_model, or head_dim for QK-Norm).
        eps: Numerical stability constant.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)                                  # compute in fp32
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def build_rope_cache(cfg: Config, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cos/sin for positions [0, seq_len).

    Returns two tensors of shape [seq_len, head_dim]. The frequencies are
    duplicated (cat([f, f])) because the split-half rotation pairs dimension i
    with dimension i + head_dim/2, so both halves need the same angle.
    """
    half = cfg.head_dim // 2
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()           # [T]
    freqs = torch.outer(pos, inv_freq)                           # [T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)                      # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split-half rotation: (x1, x2) -> (-x2, x1). NOT the interleaved variant."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q, k: [B, H, T, head_dim]; cos, sin: [T, head_dim]."""
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    [B, n_kv, T, D] -> [B, n_kv * n_rep, T, D]

    Each KV head is repeated for CONSECUTIVE query heads: KV head 0 serves query
    heads 0 and 1, KV head 1 serves 2 and 3, and so on. Using a plain tile here
    instead pairs every query head with the wrong keys.
    """
    if n_rep == 1:
        return x
    B, n_kv, T, D = x.shape
    return x[:, :, None].expand(B, n_kv, n_rep, T, D).reshape(B, n_kv * n_rep, T, D)


class KVCache:
    """Per-layer key/value store for incremental decoding."""

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)               # grow along time
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def __len__(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class GroupedQueryAttention(nn.Module):
    """
    GQA with QK-Norm and RoPE, no projection biases.

    Args:
        cfg: Model configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        q_out = cfg.n_heads * cfg.head_dim                       # 2048
        kv_out = cfg.n_kv_heads * cfg.head_dim                   # 1024

        self.q_proj = nn.Linear(cfg.d_model, q_out, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_out, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, cfg.d_model, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)        # per head, not per stream
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, cache: KVCache | None = None) -> torch.Tensor:
        """x: [B, T, d_model] -> [B, T, d_model]."""
        B, T, _ = x.shape
        cfg = self.cfg

        q = self.q_proj(x).view(B, T, cfg.n_heads, cfg.head_dim)     # [B, T, 16, 128]
        k = self.k_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim)  # [B, T,  8, 128]
        v = self.v_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim)

        q = self.q_norm(q).transpose(1, 2)                       # QK-Norm before RoPE
        k = self.k_norm(k).transpose(1, 2)                       # [B, H, T, 128]
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v)

        k = repeat_kv(k, cfg.n_rep)                              # 8 -> 16 heads
        v = repeat_kv(v, cfg.n_rep)

        # Causal only when several new positions arrive at once (prefill).
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    """y = (SiLU(x W_gate) * (x W_up)) W_down, no biases."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """One pre-norm block: attention sublayer, then FFN sublayer."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.input_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.post_attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), cos, sin, cache)
        x = x + self.ffn(self.post_attn_norm(x))
        return x


class LanguageModel(nn.Module):
    """Full decoder-only model: embedding, N blocks, final norm, LM head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)           # Pre-Norm needs this
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight                 # one matrix, two uses

    def forward(self, ids: torch.Tensor, caches: list[KVCache] | None = None) -> torch.Tensor:
        """ids: [B, T] -> logits [B, T, vocab_size]."""
        B, T = ids.shape
        x = self.embed(ids)                                      # [B, T, d_model]

        past = len(caches[0]) if caches else 0                   # positions already cached
        cos, sin = build_rope_cache(self.cfg, past + T, ids.device, x.dtype)
        cos, sin = cos[past:], sin[past:]                        # only the new positions

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches else None)

        return self.head(self.norm(x))

    def n_params(self) -> dict[str, int]:
        """Parameter breakdown, counting tied weights once."""
        cfg = self.cfg
        attn = cfg.d_model * cfg.n_heads * cfg.head_dim * 2 + 2 * cfg.d_model * cfg.n_kv_heads * cfg.head_dim
        ffn = 3 * cfg.d_model * cfg.d_ff
        norms = 2 * cfg.d_model + 2 * cfg.head_dim
        embed = cfg.vocab_size * cfg.d_model
        return {
            "attention": cfg.n_layers * attn,
            "ffn": cfg.n_layers * ffn,
            "norms": cfg.n_layers * norms + cfg.d_model,
            "embedding": embed,
            "total": cfg.n_layers * (attn + ffn + norms) + embed + cfg.d_model,
        }
