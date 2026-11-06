# SPDX-License-Identifier: Apache-2.0
"""Long-form feature producer (multi-window prompts) for DFlash fine-tuning.

One clip per item (B=1). Features are computed chunk by chunk (``Qwen3ASRTarget.feats_chunk``, C=512)
against a static cache, so one compiled graph serves every clip length (prompt <= ~3.9k tokens). Items are written to the same queue format as
``feature_producer.py`` (draft trainer unchanged), so short and long producers can feed one
trainer: the trainer's compiled graph is keyed on shapes, i.e. one graph per (Sc, B) bucket.

To bound the draft trainer's context (and the queue file size), the context fed to the draft is
cropped to the last ``--ctx-keep`` positions before the earliest anchor of the item. All anchors
are drawn from the response. Positions are absolute, so RoPE sees the true offsets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DraftConfig  # noqa: E402
from model.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from model.qwen3_asr_target import Qwen3ASRTarget, load_hf_qwen3_asr, load_processor  # noqa: E402
from labels.generate_labels_longform import MAX_PROMPT, long_embeds  # noqa: E402
from train.dflash_objective import build_mask, position_weights  # noqa: E402

C = 512


def _features(r, proc, tgt, enc, fch, dev, L_ids):
    resp = r["response_ids"]
    with torch.no_grad():
        emb, pids, W = long_embeds(tgt, enc, proc, r["audio_pcm16"], dev, torch.bfloat16, extra_ids=resp)
        assert pids == list(r["prompt_ids"]), "prompt ids differ from the label file"
        seq = torch.tensor(pids + list(resp))
        Lseq = seq.numel()
        n_ch = (Lseq + C - 1) // C
        emb = torch.cat([emb, emb.new_zeros(1, n_ch * C - Lseq, tgt.H)], 1)
        feats = torch.cat([fch(emb[:, c * C:(c + 1) * C].contiguous(),
                               torch.arange(c * C, (c + 1) * C, device=dev)) for c in range(n_ch)], 1)
    return seq, len(pids), W, feats[0, :Lseq].cpu()


def _item(seq, feats, anchors, K, n_blocks, cfg, w_slot):
    bs = cfg.block_size
    Lseq = seq.numel()
    a_max = anchors[-1]
    c0 = max(0, a_max - K)
    ctx = feats[c0:a_max]
    Sc = ctx.shape[0]
    ctx_full = torch.zeros(K, ctx.shape[1], dtype=ctx.dtype)
    ctx_full[:Sc] = ctx
    ctx_pos = torch.zeros(1, K, dtype=torch.long)
    ctx_pos[0, :Sc] = torch.arange(c0, a_max)
    Sq = n_blocks * bs
    block_ids = torch.full((1, Sq), cfg.mask_token_id, dtype=torch.long)
    q_pos = torch.zeros(1, Sq, dtype=torch.long)
    labels = torch.full((1, Sq), -100, dtype=torch.long)
    weights = torch.zeros(1, Sq)
    kk = torch.arange(bs)
    for i, an in enumerate(anchors):
        sl = slice(i * bs, (i + 1) * bs)
        block_ids[0, sl.start] = seq[an]
        pos = an + kk
        q_pos[0, sl] = pos
        inside = (pos < Lseq) & (kk >= 1)
        lab = torch.full((bs,), -100, dtype=torch.long)
        lab[inside] = seq[pos[inside]]
        labels[0, sl] = lab
        weights[0, sl] = torch.where(inside, w_slot, torch.zeros_like(w_slot))
    anc_local = torch.zeros(1, n_blocks, dtype=torch.long)
    anc_local[0, :len(anchors)] = torch.tensor(anchors) - c0
    mask = build_mask(anc_local, torch.tensor([len(anchors)]), torch.tensor([Sc]), K, n_blocks, bs)
    return {"feats": ctx_full[None], "block_ids": block_ids, "q_pos": q_pos, "ctx_pos": ctx_pos, "mask": mask,
            "labels": labels, "weights": weights, "n_blocks": n_blocks, "block_size": bs}


def write_eval(args, rows, proc, tgt, enc, fch, dev, cfg):
    """Exhaustive anchors over each clip's response, chunked into items of n_blocks consecutive anchors."""
    eo = Path(args.eval_out)
    eo.mkdir(parents=True, exist_ok=True)
    (eo / "names.json").write_text(json.dumps(["longform"]))
    w_slot = position_weights(cfg.block_size)
    k = 0
    for r in sorted(rows, key=lambda r: r["clip_id"])[: args.eval_n]:
        seq, P, W, feats = _features(r, proc, tgt, enc, fch, dev, cfg.target_layer_ids)
        anchors_all = list(range(P, seq.numel() - 1))
        for c in range(0, len(anchors_all), args.n_blocks):
            an = anchors_all[c:c + args.n_blocks]
            it = _item(seq, feats, an, args.ctx_keep, args.n_blocks, cfg, w_slot)
            it.update({"set_idx": torch.tensor([0]), "n_real": 1, "W": W})
            torch.save(it, eo / f"r{args.rank}_{k:06d}.pt")
            k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="dir with long-form *.labels.parquet")
    ap.add_argument("--audio-root", required=True, help="train_longform shard root")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--n-blocks", type=int, default=64)
    ap.add_argument("--ctx-keep", type=int, default=4352, help="draft context positions kept (bucket)")
    ap.add_argument("--max-queue", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-max", type=int, default=1536)
    ap.add_argument("--max-items", type=int, default=0, help="stop after N items (smoke tests)")
    ap.add_argument("--items-per-clip", type=int, default=4,
                    help="training items emitted per feature pass (disjoint anchor windows of one clip)")
    ap.add_argument("--eval-out", default="", help="write fixed eval items (exhaustive anchors) and exit")
    ap.add_argument("--eval-n", type=int, default=24)
    ap.add_argument("--no-compile", action="store_true")
    add_device_arg(ap)
    args = ap.parse_args()
    proc = load_processor()
    q = Path(args.queue)
    q.mkdir(parents=True, exist_ok=True)
    dev = get_device(args.device)
    cfg = DraftConfig()
    L_ids = cfg.target_layer_ids
    bs = cfg.block_size
    rows = []
    for f in sorted(Path(args.labels).glob("*.labels.parquet"))[args.rank::args.world]:
        t = pq.read_table(f).select(["clip_id", "response_ids", "n_resp", "finished", "prompt_ids"]).to_pylist()
        sh = Path(args.audio_root) / f.name.split("__")[0] / (f.name.split("__")[1].replace(".labels.parquet", ".parquet"))
        a = pq.read_table(sh, columns=["clip_id", "audio_pcm16"])
        aud = dict(zip(a.column("clip_id").to_pylist(), a.column("audio_pcm16").to_pylist()))
        for r in t:
            if r["finished"] and 16 <= r["n_resp"] <= args.r_max:
                r["audio_pcm16"] = aud[r["clip_id"]]
                rows.append(r)
    S_cache = ((MAX_PROMPT + args.r_max + C - 1) // C + 1) * C
    hf = load_hf_qwen3_asr(torch.bfloat16)
    tgt = Qwen3ASRTarget(hf, max_pos=S_cache + 8).eval()
    del hf
    del tgt.layers[max(L_ids) + 1:]
    tgt = tgt.to(dev)
    tgt.alloc_cache(1, S_cache, torch.bfloat16, dev)
    enc = maybe_compile(tgt.encode_windows, dev, not args.no_compile)
    fch = maybe_compile(lambda x, pos: tgt.feats_chunk(x, pos, L_ids), dev, not args.no_compile)
    log = open(q.parent / f"lf_producer{args.rank}.log", "a")
    print(json.dumps({"rank": args.rank, "rows": len(rows), "S_cache": S_cache}), file=log, flush=True)
    rng = random.Random(args.seed * 100 + args.rank)
    if args.eval_out:
        write_eval(args, rows, proc, tgt, enc, fch, dev, cfg)
        return
    w_slot = position_weights(bs)
    K = args.ctx_keep
    done = 0
    t0 = time.time()
    for ep in range(args.epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for k in order:
            r = rows[k]
            while len(list(q.glob(f"e*_L{args.rank}_*.pt"))) >= args.max_queue:
                time.sleep(0.05)
            seq, P, W, feats = _features(r, proc, tgt, enc, fch, dev, L_ids)
            cand = list(range(P, seq.numel() - 1))
            nb = min(args.n_blocks, len(cand))
            n_win = max(1, min(args.items_per_clip, len(cand) // max(1, nb * 2)))
            win = len(cand) // n_win
            starts = list(range(0, n_win * win, win))
            rng.shuffle(starts)
            for wi, st0 in enumerate(starts):
                span_c = cand[st0:st0 + win]
                k_ = min(nb, len(span_c))
                anchors = sorted(rng.sample(span_c, k_))
                item = _item(seq, feats, anchors, K, args.n_blocks, cfg, w_slot)
                item.update({"epoch": ep, "producer": f"L{args.rank}", "seq": done, "W": W})
                while len(list(q.glob(f"e*_L{args.rank}_*.pt"))) >= args.max_queue:
                    time.sleep(0.05)
                name = f"e{ep:03d}_L{args.rank}_{done:07d}_{wi}.pt"
                tmp = q / ("." + name + ".tmp")
                torch.save(item, tmp)
                os.replace(tmp, q / name)
            done += 1
            if args.max_items and done >= args.max_items:
                return
            if done % 20 == 0:
                print(json.dumps({"rank": args.rank, "done": done, "items_per_s": round(done / (time.time() - t0), 3)}),
                      file=log, flush=True)


if __name__ == "__main__":
    main()
