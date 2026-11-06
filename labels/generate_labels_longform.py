# SPDX-License-Identifier: Apache-2.0
"""Greedy labels for long-form clips (31-300 s) with the static-shape Qwen3-ASR target (Trainium, CUDA or CPU).

Batch = 1, one process per core. The prompt is 18 + 13 tokens per second of audio (<= ~3.9k for 300 s).
Encoder: all 8-chunk windows of the clip in groups of ``ENC_GROUP`` (one graph). Prefill: chunked
(C = 512) against one static cache (one graph for every length). Decode: one token per step.
Output rows carry ``prompt_ids`` and ``response_ids`` like the short labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from model.qwen3_asr_target import (  # noqa: E402
    EOS_IDS, Qwen3ASRTarget, audio_token_count, build_prompt_ids, load_hf_qwen3_asr, load_processor,
)
from labels.generate_labels import parse_asr  # noqa: E402

ENC_GROUP = 16
MAX_PROMPT = 18 + 13 * 300 + 16


def long_embeds(tgt, enc, proc, pcm16: bytes, dev, dt, extra_ids=None):
    """-> (embeds [1, L, H] for prompt (+ extra_ids), prompt ids, n_windows). Audio spliced at the pad run."""
    a = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    x = proc.feature_extractor(a, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    nf = int(x["attention_mask"][0].sum())
    mw, nv = Qwen3ASRTarget.windows_from_mel(x["input_features"][0], nf)
    W = mw.shape[0]
    outs = []
    for g in range(0, W, ENC_GROUP):
        m = torch.zeros(ENC_GROUP, *mw.shape[1:])
        v = torch.zeros(ENC_GROUP, dtype=torch.long)
        n = min(ENC_GROUP, W - g)
        m[:n], v[:n] = mw[g:g + n], nv[g:g + n]
        outs.append(enc(m.to(dt).to(dev), v.to(dev))[:n])
    au = Qwen3ASRTarget.gather_audio(torch.cat(outs, 0), nv.tolist())
    pids = build_prompt_ids(proc.tokenizer, audio_token_count(nf))
    ids = torch.tensor(pids + (list(extra_ids) if extra_ids is not None else []))[None].to(dev)
    emb = tgt.build_embeds(ids, au[None], Qwen3ASRTarget.audio_span(ids[0, :len(pids)].cpu()))
    return emb, pids, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--check-every", type=int, default=32)
    ap.add_argument("--no-compile", action="store_true")
    add_device_arg(ap)
    args = ap.parse_args()
    proc = load_processor()
    dev = get_device(args.device)
    dt = torch.bfloat16
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(args.root).glob("*/shard-*.parquet"))[args.rank::args.world]
    C = 512
    S = MAX_PROMPT + C + args.max_new + 1            # one cache size for every clip length
    hf = load_hf_qwen3_asr(dt)
    tgt = Qwen3ASRTarget(hf, max_pos=S + 8).eval()
    del hf
    tgt = tgt.to(dev)
    c = not args.no_compile
    enc = maybe_compile(tgt.encode_windows, dev, c)
    dec = maybe_compile(tgt.decode, dev, c)
    pchunk = maybe_compile(tgt.prefill_chunk, dev, c)
    tgt.alloc_cache(1, S, dt, dev)
    log = open(out_dir / f"rank{args.rank}.log", "a")
    t0 = time.time()
    done = 0
    for sh in shards:
        dst = out_dir / (sh.parent.name + "__" + sh.stem + ".labels.parquet")
        if dst.exists():
            continue
        tbl = pq.read_table(sh)
        meta = tbl.drop(["audio_pcm16"]).to_pylist()
        audio = tbl.column("audio_pcm16").to_pylist()
        rows = []
        for i, m in enumerate(meta):
            with torch.no_grad():
                emb, pids, W = long_embeds(tgt, enc, proc, audio[i], dev, dt)
                P = len(pids)
                n_ch = (P + C - 1) // C
                emb = torch.cat([emb, emb.new_zeros(1, n_ch * C - P, tgt.H)], 1)
                for c in range(n_ch):
                    am = pchunk(emb[:, c * C:(c + 1) * C].contiguous(), torch.arange(c * C, (c + 1) * C, device=dev))
                tok = am[:, (P - 1) - (n_ch - 1) * C].clone()  # fresh stride-1 tensor (a 1-elem slice keeps stride 512)
                toks = [tok]
                for t in range(args.max_new - 1):
                    tok = dec(tok, torch.tensor([P + t], device=dev))
                    toks.append(tok)
                    if (t + 2) % args.check_every == 0:
                        tt = torch.stack(toks).cpu()
                        if bool(((tt == EOS_IDS[0]) | (tt == EOS_IDS[1])).any()):
                            break
                s = torch.stack(toks).flatten().cpu().tolist()
            k = next((q for q, v in enumerate(s) if v in EOS_IDS), None)
            fin = k is not None
            s = s[: k + 1] if fin else s
            raw = proc.tokenizer.decode([v for v in s if v not in EOS_IDS], skip_special_tokens=False)
            lang, hyp = parse_asr(raw)
            rows.append({**m, "prompt_len": P, "prompt_ids": pids, "response_ids": s, "n_resp": len(s),
                         "finished": fin, "hyp_text": hyp, "hyp_language": lang, "hyp_raw": raw})
            done += 1
            print(json.dumps({"rank": args.rank, "shard": sh.name, "i": i, "W": W, "n_resp": len(s), "fin": fin,
                              "done": done, "sec": round(time.time() - t0, 1)}), file=log, flush=True)
        pq.write_table(pa.Table.from_pylist(rows), str(dst) + ".tmp", compression="zstd")
        os.replace(str(dst) + ".tmp", dst)
    print(json.dumps({"rank": args.rank, "finished": True, "done": done}), file=log, flush=True)


if __name__ == "__main__":
    main()
