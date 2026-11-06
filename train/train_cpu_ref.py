# SPDX-License-Identifier: Apache-2.0
"""CPU reference DFlash trainer (offline features).

Reads label shards from ``labels/generate_labels_hf.py`` and trains the draft with the
paper objective. Device-agnostic torch so the trn2 trainer can be checked
against it step by step.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import TARGET_ID, TARGET_REVISION, DraftConfig  # noqa: E402
from model.dflash import DFlashDraft, FrozenTargetHead  # noqa: E402
from model.release import save_release  # noqa: E402
from train.dflash_objective import acceptance_stats, dflash_loss, pack_batch  # noqa: E402


def load_target_head(dtype=torch.float32) -> FrozenTargetHead:
    """Target token embedding + LM head (Qwen3-ASR ties them: one tensor serves both)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    f = hf_hub_download(TARGET_ID, "model.safetensors", revision=TARGET_REVISION)
    with safe_open(f, "pt") as fh:
        emb = fh.get_tensor("model.language_model.embed_tokens.weight").to(dtype)
        lm = fh.get_tensor("lm_head.weight").to(dtype) if "lm_head.weight" in fh.keys() else emb
    return FrozenTargetHead(emb, lm)


def load_records(paths):
    recs = []
    for p in paths:
        d = torch.load(p, weights_only=False)
        recs.extend(d["records"])
    return recs


def collate(recs, n_feat_dim, dtype):
    Sc = max(int(r["ids"].numel()) for r in recs)
    feats = torch.zeros(len(recs), Sc, n_feat_dim, dtype=dtype)
    for i, r in enumerate(recs):
        L = int(r["ids"].numel())
        feats[i, :L] = r["feats"].to(dtype)
    return feats


def cosine_lr(step, total, base, warmup_frac=0.04):
    w = max(1, int(round(total * warmup_frac)))
    if step < w:
        return base * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def run_step(model, head, recs, cfg, n_blocks, gen, dtype, anchors_override=None):
    seqs = [r["ids"].long() for r in recs]
    P = [int(r["prompt_len"]) for r in recs]
    pb = pack_batch(seqs, P, n_blocks, cfg.block_size, cfg.mask_token_id, generator=gen,
                    anchors_override=anchors_override)
    feats = collate(recs, cfg.num_target_features * cfg.target_hidden_size, dtype)
    noise = head.embed_tokens(pb.block_ids).to(dtype)
    h = model(noise, feats, pb.q_pos, pb.ctx_pos, pb.mask)
    logits = head.logits(h)
    loss = dflash_loss(logits, pb)
    return loss, logits, pb


@torch.no_grad()
def evaluate(model, head, recs, cfg, dtype, bs=8):
    """Acceptance with an anchor at EVERY response position (exhaustive, deterministic)."""
    model.eval()
    agg = {"acc_sum": 0.0, "n": 0, "acc_full": 0.0, "nf": 0, "hits": 0, "valid": 0}
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        anchors = [torch.arange(int(r["prompt_len"]), int(r["ids"].numel()) - 1) for r in chunk]
        nb = max(int(a.numel()) for a in anchors)
        loss, logits, pb = run_step(model, head, chunk, cfg, nb, None, dtype, anchors_override=anchors)
        st = acceptance_stats(logits.argmax(-1), pb)
        agg["acc_sum"] += st["tau"] * st["n_blocks"]
        agg["n"] += st["n_blocks"]
        agg["acc_full"] += st["tau_full_blocks"] * st["n_full_blocks"]
        agg["nf"] += st["n_full_blocks"]
        valid = (pb.labels != -100)
        agg["hits"] += int(((logits.argmax(-1) == pb.labels) & valid).sum())
        agg["valid"] += int(valid.sum())
    model.train()
    return {"tau": agg["acc_sum"] / max(1, agg["n"]), "tau_full_blocks": agg["acc_full"] / max(1, agg["nf"]),
            "top1": agg["hits"] / max(1, agg["valid"]), "n_anchor_blocks": agg["n"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--intermediate", type=int, default=6144)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dtype = torch.float32

    cfg = DraftConfig(num_hidden_layers=args.layers, intermediate_size=args.intermediate, block_size=args.block_size)
    train = load_records(args.train)
    evalr = load_records(args.eval) if args.eval else []
    for r in train + evalr:
        assert r["feats"].shape[-1] == cfg.num_target_features * cfg.target_hidden_size
    model = DFlashDraft(cfg)
    head = load_target_head(dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"draft params {n_params/1e6:.1f}M, train clips {len(train)}, eval clips {len(evalr)}", flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.0}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)
    log = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def do_eval(step):
        rec = {"step": step, "train": evaluate(model, head, train, cfg, dtype)}
        if evalr:
            rec["heldout"] = evaluate(model, head, evalr, cfg, dtype)
        print("EVAL", json.dumps(rec), flush=True)
        log.append(rec)

    do_eval(0)
    t0 = time.time()
    order = torch.randperm(len(train), generator=gen).tolist()
    pos = 0
    for step in range(args.steps):
        if pos + args.batch > len(order):
            order = torch.randperm(len(train), generator=gen).tolist()
            pos = 0
        recs = [train[i] for i in order[pos:pos + args.batch]]
        pos += args.batch
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr)
        loss, logits, pb = run_step(model, head, recs, cfg, args.n_blocks, gen, dtype)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        st = acceptance_stats(logits.argmax(-1), pb)
        rec = {"step": step + 1, "loss": loss.item(), "grad_norm": gn.item(), "lr": opt.param_groups[0]["lr"],
               "batch_tau": st["tau"], "batch_top1": st["top1"], "sec": time.time() - t0}
        log.append(rec)
        if (step + 1) % 5 == 0 or step == 0:
            print(json.dumps(rec), flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            do_eval(step + 1)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    save_release(model, out / "release", dtype=torch.bfloat16)
    torch.save(model.state_dict(), out / "draft_fp32.pt")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
