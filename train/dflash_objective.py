# SPDX-License-Identifier: Apache-2.0
"""DFlash training objective (paper arXiv:2602.06036, Sec. 4.2 and App. A.1).

For one sequence ``x = prompt + response`` of length ``L`` (prompt = audio + instruction
tokens, response = the target's own greedy transcript, ending with EOS):

* Sample anchor positions ``a`` uniformly from the response, ``P <= a <= L - 2``
  (``P`` = prompt length) so each block has at least one label. At inference the
  anchor is the clean token the target just produced (bonus token).
* Block for anchor ``a``: tokens ``[x_a, MASK, ..., MASK]`` (``block_size`` tokens) at
  absolute positions ``a .. a + bs - 1``. Labels for block slot ``k = 1 .. bs - 1`` are
  ``x_{a+k}``; slots past the end of the sequence are ignored.
* Attention: block queries see target context features at positions ``j < a`` and
  every token of their own block (bidirectional); nothing from other blocks. This is
  exactly the inference-time view: the draft KV cache holds the target features of all
  verified tokens before the anchor, and the block is decoded non-causally.
* Loss: cross-entropy with weight ``w_k = exp(-(k - 1) / gamma)`` for slot ``k``
  (gamma = 7 for block 16, 5 for block 10, 4 for block 8), averaged over valid slots
  by total weight.

All blocks of a batch are packed into one ``[B, n_blocks * bs]`` query sequence with a
dense boolean mask ``[B, Sq, Sc + Sq]`` (context keys first, then block keys).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

PAPER_GAMMA = {16: 7.0, 10: 5.0, 8: 4.0}


def position_weights(block_size: int, gamma: float | None = None) -> torch.Tensor:
    """[block_size] weights; slot 0 (anchor) gets 0, slot k >= 1 gets exp(-(k-1)/gamma)."""
    if gamma is None:
        gamma = PAPER_GAMMA.get(block_size, block_size / 2.0)
    k = torch.arange(block_size, dtype=torch.float32)
    w = torch.exp(-(k - 1) / gamma)
    w[0] = 0.0
    return w


def sample_anchors(prompt_len: int, seq_len: int, n_blocks: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Up to ``n_blocks`` distinct anchors in ``[prompt_len, seq_len - 2]``, sorted."""
    lo, hi = prompt_len, seq_len - 2
    n_cand = hi - lo + 1
    if n_cand <= 0:
        return torch.empty(0, dtype=torch.long)
    n = min(n_blocks, n_cand)
    perm = torch.randperm(n_cand, generator=generator)[:n]
    return (perm + lo).sort().values


@dataclass
class PackedBatch:
    block_ids: torch.Tensor      # [B, Sq] long: anchor token then mask tokens
    q_pos: torch.Tensor          # [B, Sq] long
    labels: torch.Tensor         # [B, Sq] long, -100 = ignore
    weights: torch.Tensor        # [B, Sq] float, 0 = ignore
    ctx_pos: torch.Tensor        # [B, Sc] long
    mask: torch.Tensor           # [B, Sq, Sc + Sq] bool
    n_blocks: int
    block_size: int


def build_mask(anchors: torch.Tensor, n_valid_blocks: torch.Tensor, ctx_len: torch.Tensor,
               Sc: int, n_blocks: int, block_size: int) -> torch.Tensor:
    """Vectorised DFlash training mask.

    anchors: [B, n_blocks] (padding entries arbitrary), n_valid_blocks: [B], ctx_len: [B]
    (number of real context tokens). Returns bool [B, Sq, Sc + Sq].
    """
    B = anchors.shape[0]
    Sq = n_blocks * block_size
    dev = anchors.device
    q_block = torch.arange(Sq, device=dev) // block_size                     # [Sq]
    a_q = anchors[:, q_block]                                                 # [B, Sq]
    j = torch.arange(Sc, device=dev)
    ctx_vis = (j[None, None, :] < a_q[:, :, None]) & (j[None, None, :] < ctx_len[:, None, None])
    blk_valid = q_block[None, :] < n_valid_blocks[:, None]                    # [B, Sq]
    ctx_vis = ctx_vis & blk_valid[:, :, None]
    same_blk = q_block[:, None] == q_block[None, :]                           # [Sq, Sq]
    blk_vis = same_blk[None].expand(B, Sq, Sq)
    return torch.cat([ctx_vis, blk_vis], dim=-1)


def pack_batch(seqs: list[torch.Tensor], prompt_lens: list[int], n_blocks: int, block_size: int,
               mask_token_id: int, generator: torch.Generator | None = None,
               anchors_override: list[torch.Tensor] | None = None) -> PackedBatch:
    """seqs: list of [L_b] token ids (prompt + response). Context = all L_b positions."""
    B = len(seqs)
    Sc = max(int(s.numel()) for s in seqs)
    Sq = n_blocks * block_size
    block_ids = torch.full((B, Sq), mask_token_id, dtype=torch.long)
    q_pos = torch.zeros((B, Sq), dtype=torch.long)
    labels = torch.full((B, Sq), -100, dtype=torch.long)
    weights = torch.zeros((B, Sq), dtype=torch.float32)
    anchors_pad = torch.zeros((B, n_blocks), dtype=torch.long)
    n_valid = torch.zeros(B, dtype=torch.long)
    ctx_len = torch.tensor([int(s.numel()) for s in seqs], dtype=torch.long)
    w_slot = position_weights(block_size)
    k = torch.arange(block_size)
    for b, (s, P) in enumerate(zip(seqs, prompt_lens)):
        L = int(s.numel())
        anc = anchors_override[b] if anchors_override is not None else sample_anchors(P, L, n_blocks, generator)
        n = int(anc.numel())
        n_valid[b] = n
        anchors_pad[b, :n] = anc
        for i, a in enumerate(anc.tolist()):
            sl = slice(i * block_size, (i + 1) * block_size)
            block_ids[b, sl.start] = s[a]
            pos = a + k
            q_pos[b, sl] = pos
            inside = (pos < L) & (k >= 1)
            lab = torch.full((block_size,), -100, dtype=torch.long)
            lab[inside] = s[pos[inside]]
            labels[b, sl] = lab
            weights[b, sl] = torch.where(inside, w_slot, torch.zeros_like(w_slot))
        # padded blocks keep q_pos 0, labels -100, weight 0
    mask = build_mask(anchors_pad, n_valid, ctx_len, Sc, n_blocks, block_size)
    ctx_pos = torch.arange(Sc).unsqueeze(0).expand(B, Sc).clone()
    return PackedBatch(block_ids, q_pos, labels, weights, ctx_pos, mask, n_blocks, block_size)


def dflash_loss(logits: torch.Tensor, batch: PackedBatch) -> torch.Tensor:
    """Weighted CE; logits [B, Sq, V]."""
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V).float(), batch.labels.reshape(-1).clamp(min=0), reduction="none")
    w = batch.weights.reshape(-1).to(ce.device)
    return (ce * w).sum() / w.sum().clamp(min=1e-6)


@torch.no_grad()
def acceptance_stats(pred: torch.Tensor, batch: PackedBatch) -> dict:
    """Greedy DFlash verify rule per block: accepted = number of leading slots k>=1 whose
    prediction equals the label (the target's greedy token); tau = accepted + 1 (bonus).
    Blocks truncated by the sequence end are scored over their valid slots only.
    pred: [B, Sq] argmax ids.
    """
    bs = batch.block_size
    B = pred.shape[0]
    p = pred.view(B, batch.n_blocks, bs)[..., 1:]
    lab = batch.labels.view(B, batch.n_blocks, bs)[..., 1:].to(p.device)
    valid = lab != -100
    hit = (p == lab) & valid
    acc = hit.long().cumprod(-1).sum(-1)                      # [B, n_blocks]
    blk_valid = valid.any(-1)
    full = valid.all(-1)                                       # blocks with all bs-1 slots inside the sequence
    n = blk_valid.sum().clamp(min=1)
    tau = ((acc + 1).float() * blk_valid).sum() / n
    nf = full.sum().clamp(min=1)
    tau_full = ((acc + 1).float() * full).sum() / nf
    top1 = hit.sum().float() / valid.sum().clamp(min=1)
    return {"tau": tau.item(), "tau_full_blocks": tau_full.item(), "top1": top1.item(),
            "n_blocks": int(blk_valid.sum()), "n_full_blocks": int(full.sum())}


def num_blocks_for(response_len: int, max_blocks: int) -> int:
    return max(0, min(max_blocks, response_len - 1))


__all__ = [
    "PAPER_GAMMA", "position_weights", "sample_anchors", "PackedBatch", "build_mask",
    "pack_batch", "dflash_loss", "acceptance_stats", "num_blocks_for",
]
