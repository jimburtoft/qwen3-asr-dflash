# SPDX-License-Identifier: Apache-2.0
"""Target-feature producer for DFlash training (one process per accelerator core / device).

Roles are split across processes (one 24 GB Trainium2 logical core each at LNC=2): the frozen bf16
target (~4.1 GB; encoder + decoder up to the last tapped layer) and the trained fp32 draft + grads +
AdamW state (~4.4 GB + activations) run on different cores:

    producers:  audio -> mel -> target forward -> features at the 5 target layers
    trainer:    draft fwd/bwd + optimizer (train/train_draft_queue.py --queue ...)

Producers write ready batches into a bounded file queue in shared memory (``/dev/shm``):
``<queue>/<epoch>_<rank>_<seq>.pt`` written atomically (tmp + rename). A batch contains the
features plus everything the trainer needs (packed blocks, labels, weights, mask).

Each producer owns label files ``i % world == rank`` and shuffles them per epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DraftConfig  # noqa: E402
from model.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from model.qwen3_asr_target import Qwen3ASRTarget, load_hf_qwen3_asr  # noqa: E402
from train.data import CTX_LEN, WIN_PER_CLIP, LabelData, MelPrep, build_cpu_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--queue", default="/dev/shm/q3df_q")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--r-max", type=int, default=224)
    ap.add_argument("--n-blocks", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-queue", type=int, default=24, help="max ready files from this producer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset-frac", type=float, default=1.0, help="deterministic clip_id-hash subset")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--layer-ids", default="", help="comma list overriding the target layer set")
    ap.add_argument("--eval-out", default="", help="eval mode: write fixed-anchor eval batches here and exit")
    ap.add_argument("--eval-per-set", type=int, default=32)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="target dtype (fp32: CPU reference)")
    add_device_arg(ap)
    args = ap.parse_args()

    q = Path(args.queue)
    q.mkdir(parents=True, exist_ok=True)
    dev = get_device(args.device)
    cfg = DraftConfig(block_size=args.block_size)
    if args.layer_ids:
        cfg.target_layer_ids = [int(x) for x in args.layer_ids.split(",")]
    prep = MelPrep()
    data = LabelData(args.labels, args.audio_root, args.r_max, args.rank, args.world, args.seed,
                     load_audio=not args.eval_out, subset_frac=args.subset_frac)
    tdt = getattr(torch, args.dtype)
    hf = load_hf_qwen3_asr(tdt)
    tgt = Qwen3ASRTarget(hf, max_pos=CTX_LEN + 64).eval()
    del hf
    layer_ids = cfg.target_layer_ids
    del tgt.layers[max(layer_ids) + 1:]       # layers past the last tap are never needed
    tgt = tgt.to(dev)

    @torch.no_grad()
    def _decoder_feats(enc, ids, aud_idx):
        # enc [n_win,104,H] encoder output; aud_idx [B,Sc] index into the flattened enc (-1 = text)
        emb = F.embedding(ids, tgt.embed)
        a = enc.reshape(-1, enc.shape[-1]).index_select(0, aud_idx.clamp(min=0).flatten())
        emb = torch.where((aud_idx >= 0)[..., None], a.view(*ids.shape, -1).to(emb.dtype), emb)
        # Right padding only: with causal attention a valid query never attends to a later pad, so the
        # features at valid positions are exact without a key-padding mask (one less mask input).
        feats, _ = tgt.forward_full(emb, layer_ids, key_valid=None)
        return feats

    # Encoder and decoder are compiled as two graphs (smaller graphs, and the encoder graph is
    # shared with label generation).
    enc = maybe_compile(tgt.encode_windows, dev, not args.no_compile)
    dec = maybe_compile(_decoder_feats, dev, not args.no_compile)
    pool = ThreadPoolExecutor(max_workers=3)
    gen = torch.Generator().manual_seed(args.seed * 1000 + args.rank)
    log = open(q.parent / f"producer{args.rank}.log", "a")
    n_batches = len(data) // args.batch
    print(json.dumps({"rank": args.rank, "rows": len(data), "excluded_eval_speakers": data.n_excluded,
                      "batches_per_epoch": n_batches}), file=log, flush=True)

    def prepare(rows):
        mels = list(pool.map(prep.mel, [r["audio_pcm16"] for r in rows]))
        return build_cpu_batch(rows, mels, args.r_max, args.n_blocks, cfg, gen)

    def feats_of(ids, aud_idx, mel, nv):
        e = enc(mel.to(tdt).to(dev), nv.to(dev))
        return dec(e, ids.to(dev), aud_idx.to(dev)).to(torch.bfloat16).cpu()

    if args.eval_out:
        from train.data import eval_items
        ed = LabelData(args.labels, args.audio_root, args.r_max, 0, 1, args.seed)
        names, items = eval_items(ed.rows, args.n_blocks, args.eval_per_set)
        items = items[args.rank::args.world]
        eo = Path(args.eval_out)
        eo.mkdir(parents=True, exist_ok=True)
        (eo / "names.json").write_text(json.dumps([f"{a.split('/')[-1]}:{c}:{b}" for a, c, b in names]))
        B = args.batch
        for s0 in range(0, len(items), B):
            chunk = items[s0:s0 + B]
            n_real = len(chunk)
            while len(chunk) < B:
                chunk = chunk + [chunk[0]]
            rows = [c[1] for c in chunk]
            mels = [prep.mel(r["audio_pcm16"]) for r in rows]
            ids, aud_idx, mel, nv, pb = build_cpu_batch(rows, mels, args.r_max, args.n_blocks, cfg, None,
                                                        anchors_override=[c[2] for c in chunk])
            feats = feats_of(ids, aud_idx, mel, nv)
            torch.save({"feats": feats, "block_ids": pb.block_ids, "q_pos": pb.q_pos, "ctx_pos": pb.ctx_pos,
                        "mask": pb.mask, "labels": pb.labels, "weights": pb.weights, "n_blocks": pb.n_blocks,
                        "block_size": pb.block_size, "set_idx": torch.tensor([c[0] for c in chunk]),
                        "n_real": n_real}, eo / f"r{args.rank}_{s0 // B:06d}.pt")
        print(json.dumps({"rank": args.rank, "eval_batches": (len(items) + B - 1) // B}), file=log, flush=True)
        return

    t0 = time.time()
    done = 0
    for ep in range(args.epochs):
        it = data.epoch_batches(ep, args.batch, n_batches)
        nxt = pool.submit(prepare, next(it))
        for k in range(n_batches):
            ids, aud_idx, mel, nv, pb = nxt.result()
            if k + 1 < n_batches:
                nxt = pool.submit(prepare, next(it))
            while len(list(q.glob(f"*_r{args.rank}_*.pt"))) >= args.max_queue:
                time.sleep(0.05)
            feats = feats_of(ids, aud_idx, mel, nv)
            item = {"feats": feats, "block_ids": pb.block_ids, "q_pos": pb.q_pos, "ctx_pos": pb.ctx_pos,
                    "mask": pb.mask, "labels": pb.labels, "weights": pb.weights, "n_blocks": pb.n_blocks,
                    "block_size": pb.block_size, "epoch": ep, "producer": args.rank, "seq": k}
            name = f"e{ep:03d}_r{args.rank}_{k:07d}.pt"
            tmp = q / ("." + name + ".tmp")
            torch.save(item, tmp)
            os.replace(tmp, q / name)
            done += 1
            if done % 50 == 0:
                print(json.dumps({"rank": args.rank, "epoch": ep, "k": k, "done": done,
                                  "batches_per_s": round(done / (time.time() - t0), 3)}), file=log, flush=True)


if __name__ == "__main__":
    main()
