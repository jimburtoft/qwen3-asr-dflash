# SPDX-License-Identifier: Apache-2.0
"""DFlash draft model for Qwen3-ASR (training-side reference implementation).

Written from the DFlash paper (arXiv:2602.06036, Sec. 4 and App. A.3) and the
public loading format of vLLM's ``qwen3_dflash.py``. Parameter names match the
release checkpoint layout (``fc``, ``hidden_norm``, ``layers.{i}.self_attn.q_proj`` ...,
``norm``), so ``state_dict()`` of :class:`DFlashDraft` is the release state dict.

Structure (per draft layer i)::

    H_t = RMSNorm_hidden(W_c [h^(l1); ...; h^(l5)])        # once, shared by all layers
    Q_i = q_norm(W_q LN(h_d))
    K_i = k_norm([W_k H_t ; W_k LN(h_d)]),  V_i = [W_v H_t ; W_v LN(h_d)]
    RoPE on Q_i and K_i with absolute target positions
    h_d += W_o Attn(Q_i, K_i, V_i, mask);  h_d += MLP(LN(h_d))

Target features bypass the draft's input norm, Q, O and MLP (paper A.3). The token
embedding and LM head belong to the frozen target and are passed in, not owned.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import DraftConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    """Neox-style (rotate-half) RoPE tables for integer ``positions`` of any shape."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq  # [..., hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, hd]; cos/sin: [B, S, hd]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


class DFlashAttention(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        H = cfg.hidden_size
        self.q_proj = nn.Linear(H, self.nh * self.hd, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(H, self.nkv * self.hd, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(H, self.nkv * self.hd, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.nh * self.hd, H, bias=cfg.attention_bias)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)

    def forward(self, h, ctx, rope_q, rope_ctx, mask):
        """h: [B, Sq, H] draft block states (already input-normed).
        ctx: [B, Sc, H] fused target features (after fc + hidden_norm).
        rope_q / rope_ctx: (cos, sin) of shape [B, S, hd].
        mask: bool [B, Sq, Sc + Sq] (True = may attend) or None for full attention.
        """
        B, Sq, _ = h.shape
        Sc = ctx.shape[1]
        q = self.q_norm(self.q_proj(h).view(B, Sq, self.nh, self.hd)).transpose(1, 2)
        k_c = self.k_norm(self.k_proj(ctx).view(B, Sc, self.nkv, self.hd)).transpose(1, 2)
        k_q = self.k_norm(self.k_proj(h).view(B, Sq, self.nkv, self.hd)).transpose(1, 2)
        v_c = self.v_proj(ctx).view(B, Sc, self.nkv, self.hd).transpose(1, 2)
        v_q = self.v_proj(h).view(B, Sq, self.nkv, self.hd).transpose(1, 2)
        q = apply_rope(q, *rope_q)
        k_q = apply_rope(k_q, *rope_q)
        k_c = apply_rope(k_c, *rope_ctx)
        k = torch.cat([k_c, k_q], dim=2)
        v = torch.cat([v_c, v_q], dim=2)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        attn_mask = None if mask is None else mask.unsqueeze(1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(self.hd))
        if mask is not None:
            # Rows with no visible key (padding queries) give NaN in SDPA; zero them.
            o = torch.where(mask.any(-1)[:, None, :, None], o, torch.zeros_like(o))
        o = o.transpose(1, 2).reshape(B, Sq, self.nh * self.hd)
        return self.o_proj(o)


class DFlashMLP(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        H, I = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashLayer(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = DFlashAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = DFlashMLP(cfg)

    def forward(self, h, ctx, rope_q, rope_ctx, mask):
        h = h + self.self_attn(self.input_layernorm(h), ctx, rope_q, rope_ctx, mask)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class DFlashDraft(nn.Module):
    """Trainable part of the draft. ``state_dict()`` == release checkpoint tensors."""

    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.fc = nn.Linear(cfg.num_target_features * cfg.target_hidden_size, H, bias=False)
        self.hidden_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.layers = nn.ModuleList([DFlashLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(H, cfg.rms_norm_eps)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.02):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(self, noise_emb, target_feats, q_pos, ctx_pos, mask):
        """
        noise_emb:    [B, Sq, H]  target embeddings of the block tokens (anchor + masks)
        target_feats: [B, Sc, n_feat * H] concatenated target hidden states
        q_pos:        [B, Sq] absolute positions of block tokens
        ctx_pos:      [B, Sc] absolute positions of context tokens
        mask:         bool [B, Sq, Sc + Sq] or None
        returns final hidden states [B, Sq, H] (feed to the target LM head)
        """
        cfg = self.cfg
        ctx = self.hidden_norm(self.fc(target_feats))
        dt = noise_emb.dtype
        rope_q = rope_cos_sin(q_pos, cfg.head_dim, cfg.rope_theta, dt)
        rope_c = rope_cos_sin(ctx_pos, cfg.head_dim, cfg.rope_theta, dt)
        h = noise_emb
        for layer in self.layers:
            h = layer(h, ctx, rope_q, rope_c, mask)
        return self.norm(h)


class FrozenTargetHead(nn.Module):
    """Frozen token embedding + LM head taken from the target (not part of the release)."""

    def __init__(self, embed_weight: torch.Tensor, lm_head_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("embed", embed_weight, persistent=False)
        self.register_buffer("lm_head", lm_head_weight, persistent=False)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.embed)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.lm_head.t().to(h.dtype)
