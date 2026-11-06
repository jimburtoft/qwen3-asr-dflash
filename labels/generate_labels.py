# SPDX-License-Identifier: Apache-2.0
"""Greedy transcript labels for the corpus with the static-shape Qwen3-ASR target (Trainium, CUDA or CPU).

One process per accelerator core / device (on Trainium ``NEURON_RT_VISIBLE_CORES=<i>``); each takes
every ``world``-th shard. Qwen3-ASR's prompt grows with the audio (13 audio tokens per second), so
clips are sorted by duration and batched into prompt-length buckets (one compiled graph per bucket):

    mel (CPU, HF feature extractor) -> encode_windows ([B*W,128,800]) -> right-padded prompt embeds
    -> LEFT-padded prompt embeds -> prefill_left (B x P_bucket) -> up to MAX_NEW decode_left steps
    (one shared cache position per step; pads masked, RoPE positions shifted per row)

The prompt is the public chat template with automatic language ID (the target writes
``language X<asr_text>`` itself), i.e. what a server sends by default. Rows that never stop within
MAX_NEW are kept but flagged ``finished=False`` (the trainer drops them). Output: one parquet per
input shard under ``--out`` with ``prompt_ids`` + ``response_ids``.

Usage (Trainium, inside the torch-neuronx venv):
    NEURON_RT_VISIBLE_CORES=0 python labels/generate_labels.py --rank 0 --world 4 \
        --manifest <corpus>/shards/train_manifest.parquet --root <corpus>/shards --out <labels>/train
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from model.qwen3_asr_target import (  # noqa: E402
    EOS_IDS, TOKENS_PER_WINDOW, WINDOW_FRAMES, Qwen3ASRTarget, audio_token_count, build_prompt_ids,
    load_hf_qwen3_asr, load_processor, window_layout,
)

PROMPT_BUCKETS = (64, 128, 192, 256, 320, 384, 448, 512)   # 30 s of audio = 390 audio tokens + 18


class Prep:
    def __init__(self):
        self.proc = load_processor()
        self.fe = self.proc.feature_extractor

    def __call__(self, pcm16: bytes):
        a = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        x = self.fe(a, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        nf = int(x["attention_mask"][0].sum())
        mw, nv = Qwen3ASRTarget.windows_from_mel(x["input_features"][0], nf)
        ids = build_prompt_ids(self.proc.tokenizer, audio_token_count(nf))
        return mw, nv, ids


def parse_asr(raw: str) -> tuple[str | None, str]:
    """``language X<asr_text>transcript`` -> (X, transcript); anything else -> (None, raw)."""
    if "<asr_text>" in raw:
        head, text = raw.split("<asr_text>", 1)
        lang = head.strip()
        lang = lang[len("language"):].strip() if lang.startswith("language") else lang or None
        return lang, text.strip()
    return None, raw.strip()


def bucket(n: int) -> int:
    for b in PROMPT_BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"prompt of {n} tokens exceeds the largest bucket (clip > 30 s?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=224)
    ap.add_argument("--check-every", type=int, default=16)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--no-compile", action="store_true")
    add_device_arg(ap)
    args = ap.parse_args()

    dev = get_device(args.device)
    dt = torch.bfloat16
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    man = pq.read_table(args.manifest, columns=["shard"])
    shards = sorted(set(man.column("shard").to_pylist()))[args.rank::args.world]
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    prep = Prep()
    tok_dec = prep.proc.tokenizer
    Pmax = PROMPT_BUCKETS[-1]
    S = Pmax + args.max_new + 1   # cache length (shared by every bucket)
    hf = load_hf_qwen3_asr(dt)
    tgt = Qwen3ASRTarget(hf, max_pos=S + 8).eval()
    del hf
    tgt = tgt.to(dev)
    B = args.batch
    Wmax = B * math.ceil(30 * 100 / WINDOW_FRAMES)   # windows per batch at most (4 per 30 s clip)
    tgt.alloc_cache(B, S, dt, dev)
    c = not args.no_compile
    encode = maybe_compile(tgt.encode_windows, dev, c)

    def _prefill(ids, aud_flat, aud_idx, pad):
        # ids [B,Pb] LEFT-padded; aud_flat [Wmax*104, H]; aud_idx [B,Pb] (index into aud_flat, -1 = text)
        emb = F.embedding(ids, tgt.embed)
        a = aud_flat.index_select(0, aud_idx.clamp(min=0).flatten()).view(*ids.shape, -1).to(emb.dtype)
        emb = torch.where((aud_idx >= 0)[..., None], a, emb)
        return tgt.prefill_left(emb, pad)
    prefill = maybe_compile(_prefill, dev, c)
    decode = maybe_compile(tgt.decode_left, dev, c)
    pool = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // max(1, args.world)))
    log = open(out_dir / f"rank{args.rank}.log", "a")
    t_start = time.time()
    n_done = 0
    for si, shard in enumerate(shards):
        dst = out_dir / (shard.replace("/", "__").replace(".parquet", "") + ".labels.parquet")
        if dst.exists():
            continue
        tbl = pq.read_table(Path(args.root) / shard)
        meta = tbl.drop(["audio_pcm16"]).to_pylist()
        audio = tbl.column("audio_pcm16").to_pylist()
        keep = sorted((i for i in range(len(meta)) if meta[i]["duration_s"] <= args.max_dur),
                      key=lambda i: meta[i]["duration_s"])
        rows_out = []
        for b0 in range(0, len(keep), B):
            idx = keep[b0:b0 + B]
            preps = list(pool.map(prep, [audio[i] for i in idx]))
            nreal = len(preps)
            while len(preps) < B:
                preps.append(preps[0])
            P = [len(p[2]) for p in preps]
            Pb = bucket(max(P))
            mel = torch.zeros(Wmax, 128, WINDOW_FRAMES)
            nv = torch.zeros(Wmax, dtype=torch.long)
            ids = torch.full((B, Pb), EOS_IDS[1], dtype=torch.long)
            aud_idx = torch.full((B, Pb), -1, dtype=torch.long)
            w = 0
            pad = torch.tensor([Pb - p for p in P])
            for j, (mw, nvj, pid) in enumerate(preps):
                W = mw.shape[0]
                mel[w:w + W] = mw
                nv[w:w + W] = nvj
                o = int(pad[j])
                ids[j, o:] = torch.tensor(pid)
                s, e = Qwen3ASRTarget.audio_span(ids[j, o:])
                flat = torch.cat([torch.arange(n) + (w + k) * TOKENS_PER_WINDOW for k, n in enumerate(nvj.tolist())])
                assert flat.numel() == e - s
                aud_idx[j, o + s:o + e] = flat
                w += W
            t0 = time.time()
            with torch.no_grad():
                enc = encode(mel.to(dt).to(dev), nv.to(dev))
                pad_d = pad.to(dev)
                tok = prefill(ids.to(dev), enc.reshape(-1, enc.shape[-1]), aud_idx.to(dev), pad_d)
                pos = torch.tensor([Pb], device=dev)
                toks = [tok]
                for t in range(args.max_new - 1):
                    tok = decode(tok, pos, pad_d)
                    pos = pos + 1
                    toks.append(tok)
                    if (t + 2) % args.check_every == 0:
                        seq = torch.stack(toks, 1)[:nreal].cpu()
                        done = (seq == EOS_IDS[0]) | (seq == EOS_IDS[1])
                        if bool(done.any(1).all()):
                            break
                seq = torch.stack(toks, 1)[:nreal].cpu()
            dt_b = time.time() - t0
            for j, i in enumerate(idx):
                s = seq[j].tolist()
                k = next((q for q, t in enumerate(s) if t in EOS_IDS), None)
                fin = k is not None
                s = s[: k + 1] if fin else s
                raw = tok_dec.decode([x for x in s if x not in EOS_IDS], skip_special_tokens=False)
                lang, hyp = parse_asr(raw)
                rows_out.append({**meta[i], "prompt_len": P[j], "prompt_ids": preps[j][2], "response_ids": s,
                                 "n_resp": len(s), "finished": fin, "hyp_text": hyp, "hyp_language": lang,
                                 "hyp_raw": raw})
            n_done += nreal
            el = time.time() - t_start
            print(json.dumps({"rank": args.rank, "shard": si, "of": len(shards), "bucket": Pb, "batch_s": round(dt_b, 2),
                              "steps": len(toks), "done": n_done, "clips_per_s": round(n_done / el, 2)}),
                  file=log, flush=True)
        pt = pa.Table.from_pylist(rows_out)
        pq.write_table(pt, str(dst) + ".tmp", compression="zstd")
        os.replace(str(dst) + ".tmp", dst)
    print(json.dumps({"rank": args.rank, "finished": True, "done": n_done, "sec": time.time() - t_start}),
          file=log, flush=True)


if __name__ == "__main__":
    main()
