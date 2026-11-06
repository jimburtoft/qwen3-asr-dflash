# SPDX-License-Identifier: Apache-2.0
"""Static-shape Qwen3-ASR-1.7B target (plain PyTorch: Trainium, CUDA or CPU).

Re-implements the audio encoder and the 28-layer Qwen3 text decoder with fixed-shape tensors so
that ``torch.compile`` (``backend="neuron"`` on Trainium) sees one graph per bucket. Weights are
copied from the Hugging Face ``Qwen3ASRForConditionalGeneration`` (transformers >= 5.13).

Audio encoder (HF ``Qwen3ASREncoder``):

* log-mel [128, T] (T a multiple of 100) is cut into 100-frame chunks; each chunk goes through
  three stride-2 convs independently (13 tokens per full chunk) and gets sinusoid positions 0..12.
* Attention is bidirectional inside windows of 8 chunks (``n_window_infer`` 800 frames = 104
  tokens), never across windows. Only the post-conv positions of real frames are kept; the
  clip's last chunk may be partial, so the valid tokens of a window are always a prefix.
* ``encode_windows(mel[W,128,800], n_valid[W]) -> [W, 104, 2048]`` runs the encoder + projector
  for W windows; ``n_valid`` tokens per window are real, the rest is padding (masked as keys).

Text decoder: Qwen3 (QK-norm, GQA 16/8, head_dim 128, rope_theta 1e6). Qwen3-ASR declares
interleaved mRoPE, but every token (audio included) gets the same index in all three sections,
which is exactly 1-D RoPE; positions are 0..L-1.

Hidden-state convention: feature for layer ``l`` = output of decoder layer ``l`` (residual
stream, pre final norm) == HF ``hidden_states[l + 1]``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

AUDIO_TOKEN_ID = 151676
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
IM_START_ID = 151644
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
ASR_TEXT_ID = 151704
EOS_IDS = (IM_END_ID, ENDOFTEXT_ID)

CHUNK_FRAMES = 100          # 2 * n_window
TOKENS_PER_CHUNK = 13
CHUNKS_PER_WINDOW = 8       # n_window_infer // CHUNK_FRAMES
WINDOW_FRAMES = CHUNK_FRAMES * CHUNKS_PER_WINDOW
TOKENS_PER_WINDOW = TOKENS_PER_CHUNK * CHUNKS_PER_WINDOW


def post_cnn_len(frames: torch.Tensor | int):
    """Tokens produced by three (k=3, s=2, p=1) convs; 0 stays 0."""
    t = torch.as_tensor(frames)
    for _ in range(3):
        t = torch.where(t > 0, (t - 1) // 2 + 1, torch.zeros_like(t))
    return t


def audio_token_count(n_frames: int) -> int:
    full, rem = divmod(int(n_frames), CHUNK_FRAMES)
    return full * TOKENS_PER_CHUNK + int(post_cnn_len(rem))


def window_layout(n_frames: int) -> tuple[int, list[int]]:
    """(number of windows, valid tokens per window) for a clip of ``n_frames`` real mel frames."""
    n_tok = audio_token_count(n_frames)
    W = max(1, math.ceil(n_frames / WINDOW_FRAMES))
    valid = [min(TOKENS_PER_WINDOW, n_tok - w * TOKENS_PER_WINDOW) for w in range(W)]
    return W, valid


def _rope_tables(max_pos: int, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    f = torch.outer(t, inv)
    emb = torch.cat([f, f], dim=-1)
    return emb.cos(), emb.sin()


def _rot_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> torch.Tensor:
    log_inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-log_inc * torch.arange(channels // 2).float())
    t = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([torch.sin(t), torch.cos(t)], dim=1)


class EncoderLayer(nn.Module):
    def __init__(self, D, ffn, nh):
        super().__init__()
        self.D, self.nh, self.hd = D, nh, D // nh
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.wqkv = nn.Linear(D, 3 * D)
        self.wo = nn.Linear(D, D)
        self.fc1 = nn.Linear(D, ffn)
        self.fc2 = nn.Linear(ffn, D)

    def forward(self, x, mask):
        """x [W, S, D]; mask additive [W, 1, 1, S] (0 = real key, -inf-ish = padding)."""
        W, S, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.wqkv(h).view(W, S, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.hd ** -0.5)
        x = x + self.wo(o.transpose(1, 2).reshape(W, S, self.D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class AudioEncoder(nn.Module):
    """HF ``Qwen3ASREncoder`` + ``Qwen3ASRMultiModalProjector`` on whole 8-chunk windows."""

    def __init__(self, hf_enc, hf_proj, cfg):
        super().__init__()
        D = cfg.d_model
        self.D = D
        self.conv1, self.conv2, self.conv3 = hf_enc.conv2d1, hf_enc.conv2d2, hf_enc.conv2d3
        self.conv_out = hf_enc.conv_out
        self.ln_post = hf_enc.ln_post
        self.proj1, self.proj2 = hf_proj.linear_1, hf_proj.linear_2
        self.register_buffer("pos", _sinusoids(TOKENS_PER_CHUNK, D), persistent=False)
        self.layers = nn.ModuleList()
        for L in hf_enc.layers:
            a = L.self_attn
            e = EncoderLayer(D, L.fc1.out_features, a.num_heads)
            with torch.no_grad():
                e.ln1.load_state_dict(L.self_attn_layer_norm.state_dict())
                e.ln2.load_state_dict(L.final_layer_norm.state_dict())
                e.wqkv.weight.copy_(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0))
                e.wqkv.bias.copy_(torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], 0))
                e.wo.load_state_dict(a.out_proj.state_dict())
                e.fc1.load_state_dict(L.fc1.state_dict())
                e.fc2.load_state_dict(L.fc2.state_dict())
            self.layers.append(e)

    def forward(self, mel: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        """mel [W, 128, 800]; n_valid [W] long. Returns [W, 104, out_dim]."""
        W, M, _ = mel.shape
        x = mel.view(W, M, CHUNKS_PER_WINDOW, CHUNK_FRAMES).permute(0, 2, 1, 3)
        x = x.reshape(W * CHUNKS_PER_WINDOW, 1, M, CHUNK_FRAMES)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        n, c, f, t = x.shape                                        # t == 13
        x = self.conv_out(x.permute(0, 3, 1, 2).reshape(n, t, c * f)) + self.pos.to(x.dtype)
        x = x.view(W, CHUNKS_PER_WINDOW * t, self.D)
        keep = torch.arange(x.shape[1], device=x.device)[None, :] < n_valid[:, None]   # [W, S]
        mask = torch.zeros(keep.shape, dtype=x.dtype, device=x.device)
        mask = mask.masked_fill(~keep, torch.finfo(x.dtype).min)[:, None, None, :]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_post(x)
        return self.proj2(F.gelu(self.proj1(x)))


class DecoderLayer(nn.Module):
    def __init__(self, H, I, nh, nkv, hd, eps):
        super().__init__()
        self.nh, self.nkv, self.hd, self.eps, self.H, self.I = nh, nkv, hd, eps, H, I
        self.ln1 = nn.Parameter(torch.ones(H), requires_grad=False)
        self.ln2 = nn.Parameter(torch.ones(H), requires_grad=False)
        self.qn = nn.Parameter(torch.ones(hd), requires_grad=False)
        self.kn = nn.Parameter(torch.ones(hd), requires_grad=False)
        self.wqkv = nn.Parameter(torch.empty((nh + 2 * nkv) * hd, H), requires_grad=False)
        self.wo = nn.Parameter(torch.empty(H, nh * hd), requires_grad=False)
        self.wgu = nn.Parameter(torch.empty(2 * I, H), requires_grad=False)
        self.wd = nn.Parameter(torch.empty(H, I), requires_grad=False)

    def _qkv(self, x, cos, sin):
        B, S, _ = x.shape
        h = torch.rms_norm(x, (self.H,), self.ln1, self.eps)
        q, k, v = F.linear(h, self.wqkv).split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], -1)
        q = torch.rms_norm(q.view(B, S, self.nh, self.hd), (self.hd,), self.qn, self.eps).transpose(1, 2)
        k = torch.rms_norm(k.view(B, S, self.nkv, self.hd), (self.hd,), self.kn, self.eps).transpose(1, 2)
        v = v.view(B, S, self.nkv, self.hd).transpose(1, 2)
        c, s = cos[:, None], sin[:, None]
        return q * c + _rot_half(q) * s, k * c + _rot_half(k) * s, v

    def _mlp(self, x):
        h = torch.rms_norm(x, (self.H,), self.ln2, self.eps)
        g, u = F.linear(h, self.wgu).chunk(2, dim=-1)
        return x + F.linear(F.silu(g) * u, self.wd)

    def _attn_out(self, x, q, k, v, mask):
        B, S = x.shape[:2]
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / math.sqrt(self.hd))
        return x + F.linear(o.transpose(1, 2).reshape(B, S, self.nh * self.hd), self.wo)

    def forward_full(self, x, cos, sin, mask):
        q, k, v = self._qkv(x, cos, sin)
        return self._mlp(self._attn_out(x, q, k, v, mask))


class Qwen3ASRTarget(nn.Module):
    def __init__(self, hf_model, max_pos: int = 8192):
        super().__init__()
        cfg = hf_model.config
        tc = cfg.text_config
        rope = getattr(tc, "rope_theta", None) or (getattr(tc, "rope_parameters", None) or {}).get("rope_theta")
        assert float(rope) == 1e6, f"target rope_theta {rope} (silent-config-drop check)"
        self.H, self.nl = tc.hidden_size, tc.num_hidden_layers
        self.nh, self.nkv = tc.num_attention_heads, tc.num_key_value_heads
        self.hd = getattr(tc, "head_dim", None) or self.H // self.nh
        self.eps = tc.rms_norm_eps
        base = hf_model.model
        self.audio = AudioEncoder(base.audio_tower, base.multi_modal_projector, cfg.audio_config)
        dec = base.language_model
        self.embed = nn.Parameter(dec.embed_tokens.weight.detach().clone(), requires_grad=False)
        self.norm = nn.Parameter(dec.norm.weight.detach().clone(), requires_grad=False)
        self.lm_head = nn.Parameter(hf_model.lm_head.weight.detach().clone(), requires_grad=False)
        self.layers = nn.ModuleList()
        for L in dec.layers:
            a, m = L.self_attn, L.mlp
            d = DecoderLayer(self.H, m.gate_proj.weight.shape[0], self.nh, self.nkv, self.hd, self.eps)
            with torch.no_grad():
                d.ln1.copy_(L.input_layernorm.weight)
                d.ln2.copy_(L.post_attention_layernorm.weight)
                d.qn.copy_(a.q_norm.weight)
                d.kn.copy_(a.k_norm.weight)
                d.wqkv.copy_(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0))
                d.wo.copy_(a.o_proj.weight)
                d.wgu.copy_(torch.cat([m.gate_proj.weight, m.up_proj.weight], 0))
                d.wd.copy_(m.down_proj.weight)
            self.layers.append(d)
        cos, sin = _rope_tables(max_pos, self.hd, float(rope))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(self.embed.dtype)

    # ---------------- audio ----------------
    @staticmethod
    def windows_from_mel(mel: torch.Tensor, n_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        """mel [128, T>=n_frames] (HF feature extractor output) -> ([W,128,800], n_valid [W])."""
        W, valid = window_layout(n_frames)
        need = W * WINDOW_FRAMES
        m = mel[:, :need]
        if m.shape[1] < need:
            m = F.pad(m, (0, need - m.shape[1]))
        # frames past the clip's last 100-frame chunk never reach a kept token; zero them so the
        # result does not depend on how far the caller padded
        n_chunk_frames = math.ceil(n_frames / CHUNK_FRAMES) * CHUNK_FRAMES
        m = m.clone()
        m[:, n_chunk_frames:] = 0
        m = m.view(m.shape[0], W, WINDOW_FRAMES).transpose(0, 1).contiguous()
        return m, torch.tensor(valid, dtype=torch.long)

    def encode_windows(self, mel_w: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
        return self.audio(mel_w, n_valid)

    @staticmethod
    def gather_audio(enc: torch.Tensor, n_valid: list[int]) -> torch.Tensor:
        """[W,104,H] + valid counts -> [sum(valid), H] in clip order."""
        return torch.cat([enc[w, :n] for w, n in enumerate(n_valid)], dim=0)

    @staticmethod
    def audio_span(ids_row: torch.Tensor) -> tuple[int, int]:
        pos = (ids_row == AUDIO_TOKEN_ID).nonzero().flatten()
        s, e = int(pos[0]), int(pos[-1]) + 1
        assert e - s == pos.numel(), "audio tokens not contiguous"
        return s, e

    def build_embeds(self, ids: torch.Tensor, audio: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        """ids [B,L] (all rows share the audio span), audio [B, e-s, H]."""
        s, e = span
        emb = F.embedding(ids, self.embed)
        return torch.cat([emb[:, :s], audio.to(emb.dtype), emb[:, e:]], dim=1)

    # ---------------- full (teacher-forced) ----------------
    def _cos_sin(self, S, dtype):
        return self.cos[:S].to(dtype)[None], self.sin[:S].to(dtype)[None]

    def forward_full(self, x: torch.Tensor, layer_ids: list[int], key_valid: torch.Tensor | None = None,
                     want_logits: bool = False):
        """x [B,S,H] embeddings. Returns (feats [B,S,len(layer_ids)*H], logits or None)."""
        B, S, _ = x.shape
        cos, sin = self._cos_sin(S, x.dtype)
        mask = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()[None, None]
        if key_valid is not None:
            mask = mask & key_valid[:, None, None, :]
            mask = mask | torch.eye(S, dtype=torch.bool, device=x.device)[None, None]
        want = set(layer_ids)
        last = self.nl - 1 if want_logits else max(layer_ids)
        outs = {}
        for i, layer in enumerate(self.layers):
            if i > last:
                break
            x = layer.forward_full(x, cos, sin, mask)
            if i in want:
                outs[i] = x
        feats = torch.cat([outs[i] for i in layer_ids], dim=-1)
        logits = None
        if want_logits:
            logits = F.linear(torch.rms_norm(x, (self.H,), self.norm, self.eps), self.lm_head)
        return feats, logits

    # ---------------- static KV-cache greedy decoding ----------------
    def alloc_cache(self, B: int, S: int, dtype, device):
        self.kc = [torch.zeros(B, self.nkv, S, self.hd, dtype=dtype, device=device) for _ in self.layers]
        self.vc = [torch.zeros(B, self.nkv, S, self.hd, dtype=dtype, device=device) for _ in self.layers]
        self.S_cache = S

    def prefill_chunk(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Chunked prefill against the static cache. x [B,C,H] at positions ``pos`` [C].
        Returns argmax at every chunk position [B, C]."""
        cos = F.embedding(pos, self.cos).to(x.dtype)[None]
        sin = F.embedding(pos, self.sin).to(x.dtype)[None]
        ar = torch.arange(self.S_cache, device=x.device)
        mask = (ar[None, :] <= pos[:, None])[None, None]
        h = x
        for i, layer in enumerate(self.layers):
            q, k, v = layer._qkv(h, cos, sin)
            self.kc[i].index_copy_(2, pos, k)
            self.vc[i].index_copy_(2, pos, v)
            h = layer._mlp(layer._attn_out(h, q, self.kc[i], self.vc[i], mask))
        hn = torch.rms_norm(h, (self.H,), self.norm, self.eps)
        return F.linear(hn, self.lm_head).argmax(-1)

    def feats_chunk(self, x: torch.Tensor, pos: torch.Tensor, layer_ids: list[int]) -> torch.Tensor:
        """Teacher-forced features for one chunk against the static cache (long-form)."""
        cos = F.embedding(pos, self.cos).to(x.dtype)[None]
        sin = F.embedding(pos, self.sin).to(x.dtype)[None]
        ar = torch.arange(self.S_cache, device=x.device)
        mask = (ar[None, :] <= pos[:, None])[None, None]
        h = x
        want = set(layer_ids)
        outs = {}
        for i, layer in enumerate(self.layers):
            if i > max(layer_ids):
                break
            q, k, v = layer._qkv(h, cos, sin)
            self.kc[i].index_copy_(2, pos, k)
            self.vc[i].index_copy_(2, pos, v)
            h = layer._mlp(layer._attn_out(h, q, self.kc[i], self.vc[i], mask))
            if i in want:
                outs[i] = h
        return torch.cat([outs[i] for i in layer_ids], dim=-1)

    def prefill_rows(self, x: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        """Batched prefill of right-padded prompts of different lengths (one graph per padded length).

        x [B,Pm,H] embeddings at positions 0..Pm-1; ``last`` [B] = P_b - 1 (index of each row's last
        real prompt token). Causal attention needs no padding mask: a real query at p only sees keys
        <= p < P_b. Padded positions write K/V that later ``decode_rows`` steps overwrite before they
        are visible. Returns argmax at each row's last real position [B]."""
        B, Pm, _ = x.shape
        pos = torch.arange(Pm, device=x.device)
        cos = F.embedding(pos, self.cos).to(x.dtype)[None]
        sin = F.embedding(pos, self.sin).to(x.dtype)[None]
        ar = torch.arange(self.S_cache, device=x.device)
        mask = (ar[None, :] <= pos[:, None])[None, None]
        h = x
        for i, layer in enumerate(self.layers):
            q, k, v = layer._qkv(h, cos, sin)
            self.kc[i].index_copy_(2, pos, k)
            self.vc[i].index_copy_(2, pos, v)
            h = layer._mlp(layer._attn_out(h, q, self.kc[i], self.vc[i], mask))
        idx = last.view(B, 1, 1).expand(B, 1, self.H)
        hl = torch.gather(h, 1, idx)
        hn = torch.rms_norm(hl, (self.H,), self.norm, self.eps)
        return F.linear(hn, self.lm_head)[:, 0].argmax(-1)

    def decode_rows(self, tok: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """tok [B] long, pos [B] long (per-row positions). Writes K/V at (b, pos_b); row b attends to
        cache positions <= pos_b. Returns argmax [B]."""
        B = tok.shape[0]
        x = F.embedding(tok.view(-1, 1), self.embed)
        cos = F.embedding(pos.view(B, 1), self.cos).to(x.dtype)       # [B,1,hd]
        sin = F.embedding(pos.view(B, 1), self.sin).to(x.dtype)
        ar = torch.arange(self.S_cache, device=x.device)
        mask = (ar[None, :] <= pos[:, None])[:, None, None, :]       # [B,1,1,S]
        onehot = (ar[None, :] == pos[:, None]).to(x.dtype)[:, None, :, None]   # [B,1,S,1]
        for i, layer in enumerate(self.layers):
            q, k, v = layer._qkv(x, cos, sin)                         # k,v [B,nkv,1,hd]
            # per-row scatter without data-dependent indexing: blend the new K/V in at pos_b
            self.kc[i].copy_(self.kc[i] * (1 - onehot) + k * onehot)
            self.vc[i].copy_(self.vc[i] * (1 - onehot) + v * onehot)
            x = layer._mlp(layer._attn_out(x, q, self.kc[i], self.vc[i], mask))
        h = torch.rms_norm(x, (self.H,), self.norm, self.eps)
        return F.linear(h, self.lm_head)[:, 0].argmax(-1)

    # ---- left-padded batches: every row's prompt ends at the same position, so decode writes
    # one shared cache position per step (index_copy, no per-row scatter). Row b has ``pad[b]``
    # leading pad tokens that are masked as keys. Positions are shifted by -pad[b] per row in RoPE
    # (rotary attention depends only on relative positions, so the shift is exact in math).
    def prefill_left(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """x [B,Pm,H] left-padded embeddings; pad [B]. Returns argmax at the last position [B]."""
        B, Pm, _ = x.shape
        ar = torch.arange(Pm, device=x.device)
        rp = (ar[None, :] - pad[:, None]).clamp(min=0)                  # per-row rope positions
        cos = F.embedding(rp, self.cos).to(x.dtype)
        sin = F.embedding(rp, self.sin).to(x.dtype)
        keys = torch.arange(self.S_cache, device=x.device)
        mask = (keys[None, None, :] <= ar[None, :, None]) & (keys[None, None, :] >= pad[:, None, None])
        mask = (mask | (keys[None, None, :] == ar[None, :, None]))[:, None]   # pads see themselves (no NaN)
        h = x
        for i, layer in enumerate(self.layers):
            q, k, v = layer._qkv(h, cos, sin)
            self.kc[i].index_copy_(2, ar, k)
            self.vc[i].index_copy_(2, ar, v)
            h = layer._mlp(layer._attn_out(h, q, self.kc[i], self.vc[i], mask))
        hn = torch.rms_norm(h[:, -1:], (self.H,), self.norm, self.eps)
        return F.linear(hn, self.lm_head)[:, 0].argmax(-1)

    def decode_left(self, tok: torch.Tensor, pos: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """tok [B]; pos [1] shared cache position; pad [B]. Returns argmax [B]."""
        B = tok.shape[0]
        x = F.embedding(tok.view(-1, 1), self.embed)
        rp = (pos - pad).view(B, 1)
        cos = F.embedding(rp, self.cos).to(x.dtype)
        sin = F.embedding(rp, self.sin).to(x.dtype)
        keys = torch.arange(self.S_cache, device=x.device)
        mask = ((keys[None, :] <= pos) & (keys[None, :] >= pad[:, None]))[:, None, None, :]
        for i, layer in enumerate(self.layers):
            q, k, v = layer._qkv(x, cos, sin)
            self.kc[i].index_copy_(2, pos, k)
            self.vc[i].index_copy_(2, pos, v)
            x = layer._mlp(layer._attn_out(x, q, self.kc[i], self.vc[i], mask))
        h = torch.rms_norm(x, (self.H,), self.norm, self.eps)
        return F.linear(h, self.lm_head)[:, 0].argmax(-1)

    def decode(self, tok: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """tok [B] long, pos [1] long (same position for every row). Returns argmax [B]."""
        x = F.embedding(tok.view(-1, 1), self.embed)
        cos = F.embedding(pos, self.cos).to(x.dtype)[None]
        sin = F.embedding(pos, self.sin).to(x.dtype)[None]
        ar = torch.arange(self.S_cache, device=x.device)
        mask = (ar <= pos)[None, None, None, :]
        for i, layer in enumerate(self.layers):
            q, k, v = layer._qkv(x, cos, sin)
            self.kc[i].index_copy_(2, pos, k)
            self.vc[i].index_copy_(2, pos, v)
            x = layer._mlp(layer._attn_out(x, q, self.kc[i], self.vc[i], mask))
        h = torch.rms_norm(x, (self.H,), self.norm, self.eps)
        return F.linear(h, self.lm_head)[:, 0].argmax(-1)


# ---------------- prompt ----------------
def build_prompt_ids(tokenizer, n_audio: int, language: str | None = None, context: str = "") -> list[int]:
    """Chat-template prompt of the public Qwen3-ASR chat template (+ optional forced language).

    ``<|im_start|>system\\n{context}<|im_end|>\\n<|im_start|>user\\n<|audio_start|>`` +
    ``<|audio_pad|> * n_audio`` + ``<|audio_end|><|im_end|>\\n<|im_start|>assistant\\n`` and, if
    ``language`` is given, ``language {language}<asr_text>``. With ``language=None`` the target
    writes ``language X<asr_text>`` itself (auto language ID, the serving default).
    """
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    ids = [IM_START_ID] + enc("system\n" + context) + [IM_END_ID] + enc("\n")
    ids += [IM_START_ID] + enc("user\n") + [AUDIO_START_ID] + [AUDIO_TOKEN_ID] * n_audio
    ids += [AUDIO_END_ID, IM_END_ID] + enc("\n") + [IM_START_ID] + enc("assistant\n")
    if language is not None:
        ids += enc(f"language {language}") + [ASR_TEXT_ID]
    return ids


def load_hf_qwen3_asr(dtype=torch.bfloat16):
    from transformers import Qwen3ASRForConditionalGeneration
    from .config import TARGET_ID, TARGET_REVISION
    m = Qwen3ASRForConditionalGeneration.from_pretrained(TARGET_ID, revision=TARGET_REVISION, dtype=dtype,
                                                         attn_implementation="eager")
    return m.eval()


def load_processor():
    from transformers import AutoProcessor
    from .config import TARGET_ID, TARGET_REVISION
    return AutoProcessor.from_pretrained(TARGET_ID, revision=TARGET_REVISION)
