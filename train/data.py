# SPDX-License-Identifier: Apache-2.0
"""Shared data helpers for DFlash training on Qwen3-ASR labels.

* ``LabelData``: labelled short clips (target greedy responses + prompt ids) joined with their audio.
* ``MelPrep``: HF feature extractor -> fixed 8-chunk encoder windows for one clip.
* ``build_cpu_batch``: pack a batch of clips into right-padded prompt+response ids, the audio scatter
  index into the encoder output, and DFlash training blocks, all at fixed shapes:
  context ``Sc = ctx_len`` tokens, ``n_win`` encoder windows.
* ``eval_items``: exhaustive-anchor held-out evaluation items.

Qwen3-ASR prompt length = 18 + 13 tokens per second of audio, so every row has its own prompt
length ``P_b``; DFlash anchors are drawn from that row's response.
"""

from __future__ import annotations

import glob
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DraftConfig  # noqa: E402
from model.qwen3_asr_target import (  # noqa: E402
    TOKENS_PER_WINDOW, WINDOW_FRAMES, Qwen3ASRTarget, load_processor,
)
from train.dflash_objective import pack_batch  # noqa: E402

EXCLUDE = ROOT / "data" / "filters" / "train_exclude_eval_speakers.txt"

# Fixed shapes for the compiled producer graphs (short clips <= 30 s: <= 4 windows, prompt <= 408).
CTX_LEN = 640          # prompt (<= 408) + response (<= r_max 224) + slack, padded
WIN_PER_CLIP = 4


def _frac_hash(clip_id: str) -> float:
    import hashlib
    return int(hashlib.sha1(clip_id.encode()).hexdigest()[:8], 16) / 2**32


class LabelData:
    """Labelled clips (finished, 2 <= n_resp <= r_max, prompt + response <= ctx_len) with audio in RAM.

    Rank r owns the label files whose (sorted) index % world == r and loads only their audio.
    """

    def __init__(self, label_glob: str, audio_root: str, r_max: int, rank: int, world: int, seed: int,
                 load_audio: bool = True, subset_frac: float = 1.0, ctx_len: int = CTX_LEN):
        files = sorted(glob.glob(label_glob))
        assert files, f"no label files match {label_glob}"
        files = files[rank::world]
        excl = set(EXCLUDE.read_text().split()) if (EXCLUDE.exists() and "/train/" in label_glob) else set()
        self.n_excluded = 0
        self.rows = []
        for f in files:
            t = pq.read_table(f)
            if t.num_rows == 0:
                continue
            cols = ["clip_id", "dataset", "config", "split", "duration_s", "prompt_len", "prompt_ids",
                    "response_ids", "n_resp", "finished"]
            t = t.select([c for c in cols if c in t.column_names]).to_pylist()
            shard = Path(f).name.replace(".labels.parquet", "").replace("__", "/") + ".parquet"
            n0 = len(t)
            t = [r for r in t if r["clip_id"] not in excl]
            self.n_excluded += n0 - len(t)
            keep = {r["clip_id"]: r for r in t if r["finished"] and 2 <= r["n_resp"] <= r_max
                    and r["prompt_len"] + r["n_resp"] <= ctx_len
                    and (subset_frac >= 1.0 or _frac_hash(r["clip_id"]) < subset_frac)}
            if load_audio and keep:
                a = pq.read_table(Path(audio_root) / shard, columns=["clip_id", "audio_pcm16"])
                for cid, pcm in zip(a.column("clip_id").to_pylist(), a.column("audio_pcm16").to_pylist()):
                    if cid in keep:
                        keep[cid]["audio_pcm16"] = pcm
            self.rows.extend(keep.values())
        self.rank, self.world, self.seed = rank, world, seed

    def __len__(self):
        return len(self.rows)

    def epoch_batches(self, epoch: int, batch: int, n_batches: int):
        idx = list(range(len(self.rows)))
        random.Random(self.seed * 7919 + 1000 * epoch + self.rank).shuffle(idx)
        for k in range(n_batches):
            yield [self.rows[i] for i in idx[k * batch:(k + 1) * batch]]


class MelPrep:
    def __init__(self):
        self.proc = load_processor()
        self.fe = self.proc.feature_extractor

    def mel(self, pcm16: bytes):
        """-> (windows [W,128,800], valid tokens per window [W])."""
        a = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        x = self.fe(a, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        nf = int(x["attention_mask"][0].sum())
        return Qwen3ASRTarget.windows_from_mel(x["input_features"][0], nf)


def pack_windows(mels, n_win: int):
    """List of (windows, n_valid) -> (mel [n_win,128,800], n_valid [n_win], first window index per clip)."""
    mel = torch.zeros(n_win, 128, WINDOW_FRAMES)
    nv = torch.zeros(n_win, dtype=torch.long)
    starts, w = [], 0
    for mw, v in mels:
        W = mw.shape[0]
        assert w + W <= n_win, f"{w + W} windows > {n_win}"
        mel[w:w + W] = mw
        nv[w:w + W] = v
        starts.append(w)
        w += W
    return mel, nv, starts


def build_cpu_batch(rows, mels, r_max, n_blocks, cfg: DraftConfig, gen, anchors_override=None,
                    ctx_len: int = CTX_LEN):
    """Returns (ids [B,Sc], aud_idx [B,Sc] (-1 = text), mel [B*4,128,800], n_valid, pb)."""
    B = len(rows)
    Sc = ctx_len
    n_win = B * WIN_PER_CLIP
    mel, nv, starts = pack_windows(mels, n_win)
    ids = torch.full((B, Sc), cfg.pad_token_id, dtype=torch.long)
    aud_idx = torch.full((B, Sc), -1, dtype=torch.long)
    seqs, P = [], []
    for b, r in enumerate(rows):
        prompt = torch.tensor(r["prompt_ids"], dtype=torch.long)
        resp = torch.tensor(r["response_ids"], dtype=torch.long)
        p, L = prompt.numel(), prompt.numel() + resp.numel()
        assert L <= Sc
        ids[b, :p] = prompt
        ids[b, p:L] = resp
        s, e = Qwen3ASRTarget.audio_span(prompt)
        v = mels[b][1].tolist()
        flat = torch.cat([torch.arange(n) + (starts[b] + k) * TOKENS_PER_WINDOW for k, n in enumerate(v)])
        assert flat.numel() == e - s, (flat.numel(), e - s)
        aud_idx[b, s:e] = flat
        seqs.append(ids[b, :L])
        P.append(p)
    pb = pack_batch(seqs, P, n_blocks, cfg.block_size, cfg.mask_token_id, generator=gen,
                    anchors_override=anchors_override)
    sc_now = pb.mask.shape[-1] - pb.block_ids.shape[1]
    if sc_now < Sc:
        Sq = pb.block_ids.shape[1]
        m = torch.zeros(B, Sq, Sc + Sq, dtype=torch.bool)
        m[:, :, :sc_now] = pb.mask[:, :, :sc_now]
        m[:, :, Sc:] = pb.mask[:, :, sc_now:]
        pb.mask = m
        pb.ctx_pos = torch.arange(Sc).unsqueeze(0).expand(B, Sc).clone()
    return ids, aud_idx, mel, nv, pb


def eval_items(rows, n_blocks, per_set: int, seed: int = 0):
    """(set index, row, anchors): every response position is an anchor, chunked by n_blocks.
    At most ``per_set`` clips per (dataset, config, split)."""
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r.get("config", ""), r["split"]), []).append(r)
    items = []
    names = sorted(by)
    for si, k in enumerate(names):
        rs = by[k]
        random.Random(seed).shuffle(rs)
        for r in rs[:per_set]:
            P = r["prompt_len"]
            a = torch.arange(P, P + r["n_resp"] - 1)
            for c in range(0, a.numel(), n_blocks):
                items.append((si, r, a[c:c + n_blocks]))
    return names, items
