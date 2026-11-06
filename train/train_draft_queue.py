# SPDX-License-Identifier: Apache-2.0
"""DFlash draft trainer, consuming target features from feature_producer.py.

Runs on Trainium (``neuron``), CUDA or CPU (``--device``). Default layout on one trn2.3xlarge
(LNC=2, 4 x 24 GB logical cores):
    cores 0-2: feature producers (frozen bf16 target up to the last tapped layer, ~3.6 GB each)
    core  3  : this trainer (fp32 master draft 1.1 GB + grads 1.1 GB + AdamW 2.2 GB + activations)
On a larger accelerator the producer and trainer can share a device.

Each step: one ready batch from the queue -> draft fwd/bwd (compiled) -> clip 1.0 -> AdamW ->
cosine LR with warmup. Held-out acceptance on precomputed fixed-anchor eval batches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import TARGET_ID, TARGET_REVISION, DraftConfig  # noqa: E402
from model.dflash import DFlashDraft  # noqa: E402
from model.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from model.release import save_release  # noqa: E402
from train.dflash_objective import PackedBatch, acceptance_stats  # noqa: E402
from train.train_cpu_ref import cosine_lr  # noqa: E402


def load_head(dtype=torch.bfloat16):
    """Target token embedding and LM head (tied in Qwen3-ASR: the same tensor twice)."""
    from train.train_cpu_ref import load_target_head
    h = load_target_head(dtype)
    return h.embed, h.lm_head


class DraftStep(torch.nn.Module):
    def __init__(self, draft: DFlashDraft, embed: torch.Tensor, lm_head: torch.Tensor, compute_dtype,
                 learn_mask_emb: bool = False):
        super().__init__()
        self.draft = draft
        self.register_buffer("embed", embed, persistent=False)
        self.register_buffer("lm_head", lm_head, persistent=False)
        self.cdt = compute_dtype
        self.mask_id = draft.cfg.mask_token_id
        # Optional separately trained mask embedding (vLLM `mask_embedding.pt` format), fp32 master,
        # initialised from the target's (unused) embedding row of mask_token_id.
        self.mask_emb = (torch.nn.Parameter(embed[self.mask_id].detach().float().clone())
                         if learn_mask_emb else None)

    def forward(self, feats, block_ids, q_pos, ctx_pos, mask, labels, weights):
        noise = F.embedding(block_ids, self.embed)
        if self.mask_emb is not None:
            is_m = (block_ids == self.mask_id).unsqueeze(-1)
            noise = torch.where(is_m, self.mask_emb.to(noise.dtype), noise)
        if self.cdt == torch.bfloat16:
            with torch.autocast(device_type=feats.device.type, dtype=torch.bfloat16):
                h = self.draft(noise, feats, q_pos, ctx_pos, mask)
            logits = F.linear(h.to(self.lm_head.dtype), self.lm_head)
        else:
            h = self.draft(noise.float(), feats.float(), q_pos, ctx_pos, mask)
            logits = F.linear(h, self.lm_head.float())
        V = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, V).float(), labels.reshape(-1).clamp(min=0), reduction="none")
        w = weights.reshape(-1)
        loss = (ce * w).sum() / w.sum().clamp(min=1e-6)
        return loss, logits.argmax(-1)


def init_from_target(draft: DFlashDraft, layer_ids: list[int]) -> None:
    """Copy target decoder layers ``layer_ids`` into the draft layers (same Qwen3 dims; ablation).
    ``fc`` / ``hidden_norm`` stay random; ``norm`` copies the target's final norm."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from model.config import TARGET_ID, TARGET_REVISION
    f = hf_hub_download(TARGET_ID, "model.safetensors", revision=TARGET_REVISION)
    sd = draft.state_dict()
    with safe_open(f, "pt") as fh, torch.no_grad():
        for i, l in enumerate(layer_ids):
            pre = f"model.language_model.layers.{l}."
            for k in list(sd):
                if k.startswith(f"layers.{i}."):
                    src = pre + k[len(f"layers.{i}."):]
                    sd[k].copy_(fh.get_tensor(src).to(sd[k].dtype))
        sd["norm.weight"].copy_(fh.get_tensor("model.language_model.norm.weight").to(sd["norm.weight"].dtype))
    draft.load_state_dict(sd)


def to_dev(item, dev):
    keys = ("feats", "block_ids", "q_pos", "ctx_pos", "mask", "labels", "weights")
    return [item[k].to(dev) for k in keys]


def pb_of(item):
    return PackedBatch(item["block_ids"], item["q_pos"], item["labels"], item["weights"], item["ctx_pos"],
                       item["mask"], item["n_blocks"], item["block_size"])


@torch.no_grad()
def evaluate(step_fn, eval_dir: Path, dev, block_size: int):
    names = json.loads((eval_dir / "names.json").read_text())
    acc = torch.zeros(len(names), 4, dtype=torch.float64)
    for f in sorted(eval_dir.glob("r*_*.pt")):
        it = torch.load(f, weights_only=False)
        _, pred = step_fn(*to_dev(it, dev))
        pred = pred.cpu()
        n, nb, bs = it["n_real"], it["n_blocks"], it["block_size"]
        B = pred.shape[0]
        p = pred.view(B, nb, bs)[:n, :, 1:]
        lab = it["labels"].view(B, nb, bs)[:n, :, 1:]
        v = lab != -100
        hit = (p == lab) & v
        a = hit.long().cumprod(-1).sum(-1)
        blk = v.any(-1)
        for j in range(n):
            si = int(it["set_idx"][j])
            acc[si, 0] += float(((a[j] + 1) * blk[j]).sum())
            acc[si, 1] += float(blk[j].sum())
            acc[si, 2] += float(hit[j].sum())
            acc[si, 3] += float(v[j].sum())
    res = {k: {"tau": round(float(acc[i, 0] / max(acc[i, 1], 1)), 4), "top1": round(float(acc[i, 2] / max(acc[i, 3], 1)), 4),
               "blocks": int(acc[i, 1])} for i, k in enumerate(names)}
    tot = acc.sum(0)
    en = [i for i, k in enumerate(names) if (k.split(":") + ["", ""])[1] in ("en", "all", "ihm", "en_us")
          or k.startswith(("librispeech", "ami"))]
    if en:
        te = acc[en].sum(0)
        res["english"] = {"tau": round(float(te[0] / max(te[1], 1)), 4), "top1": round(float(te[2] / max(te[3], 1)), 4),
                          "blocks": int(te[1])}
    res["all"] = {"tau": round(float(tot[0] / max(tot[1], 1)), 4), "top1": round(float(tot[2] / max(tot[3], 1)), 4),
                  "blocks": int(tot[1])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="/dev/shm/q3df_q")
    ap.add_argument("--eval-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--accum", type=int, default=1, help="queue batches per optimizer step")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.04)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--compute", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--s3", default="", help="optional s3:// prefix to mirror checkpoints to (aws cli)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="")
    ap.add_argument("--init-from-target", default="", help="comma list of target layers to copy into the draft layers")
    ap.add_argument("--keep-queue", action="store_true", help="do not delete consumed items (parity runs)")
    ap.add_argument("--long-queue", default="", help="second queue (long-form items, B=1 / Sc=4352)")
    ap.add_argument("--long-every", type=int, default=0, help="use one long item every N steps (0 = never)")
    ap.add_argument("--long-eval-dir", default="", help="precomputed long-form eval items")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--learn-mask-emb", action="store_true")
    ap.add_argument("--layer-ids", default="", help="must match the producers' --layer-ids")
    add_device_arg(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    cfg = DraftConfig(num_hidden_layers=args.layers)
    if args.layer_ids:
        cfg.target_layer_ids = [int(x) for x in args.layer_ids.split(",")]
    draft = DFlashDraft(cfg)
    if args.init:
        sd = torch.load(args.init, map_location="cpu", weights_only=False)
        draft.load_state_dict(sd.get("draft", sd))
    elif args.init_from_target:
        init_from_target(draft, [int(x) for x in args.init_from_target.split(",")])
    draft = draft.to(dev)
    embed, lm_head = load_head(torch.bfloat16)
    mod = DraftStep(draft, embed, lm_head, torch.bfloat16 if args.compute == "bf16" else torch.float32,
                    learn_mask_emb=args.learn_mask_emb).to(dev)
    step_fn = maybe_compile(mod, dev, enabled=not args.no_compile)
    params = list(draft.parameters()) + ([mod.mask_emb] if mod.mask_emb is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    q = Path(args.queue)
    eval_dir = Path(args.eval_dir) if args.eval_dir else None
    log = open(out / "train.jsonl", "a")

    def do_eval(s):
        if eval_dir is None:
            return
        draft.eval()
        t = time.time()
        res = evaluate(step_fn, eval_dir, dev, cfg.block_size)
        if args.long_eval_dir:
            res["longform"] = evaluate(step_fn, Path(args.long_eval_dir), dev, cfg.block_size)["all"]
        draft.train()
        rec = {"eval_step": s, "sec": round(time.time() - t, 1), **res}
        print("EVAL " + json.dumps(rec), flush=True)
        with open(out / "eval.jsonl", "a") as f:
            print(json.dumps(rec), file=f)

    lq = Path(args.long_queue) if args.long_queue else None

    def next_item(qq=None, pat="e*_r*_*.pt"):
        qq = qq or q
        while True:
            files = sorted(qq.glob(pat), key=lambda p: p.stat().st_mtime)
            if files:
                f = files[0]
                try:
                    it = torch.load(f, weights_only=False)
                except Exception:
                    time.sleep(0.05)
                    continue
                if not args.keep_queue:
                    f.unlink(missing_ok=True)
                return it
            time.sleep(0.02)

    do_eval(0)
    t0 = time.time()
    wait = 0.0
    n_lab = 0
    for step in range(args.total_steps):
        lr = cosine_lr(step, args.total_steps, args.lr, args.warmup_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tl = 0.0
        stats = None
        for a in range(args.accum):
            tw = time.time()
            use_long = lq is not None and args.long_every and (step % args.long_every == args.long_every - 1)
            it = next_item(lq, "e*_L*_*.pt") if use_long else next_item()
            wait += time.time() - tw
            loss, pred = step_fn(*to_dev(it, dev))
            (loss / args.accum).backward()
            tl += float(loss) / args.accum if (step + 1) % args.log_every == 0 or step == 0 else 0.0
            n_lab += int((it["weights"] > 0).sum())
            if a == 0 and ((step + 1) % args.log_every == 0 or step == 0):
                stats = acceptance_stats(pred.cpu(), pb_of(it))
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if stats is not None:
            el = time.time() - t0
            rec = {"step": step + 1, "loss": round(tl, 5), "grad_norm": round(float(gn), 4), "lr": lr,
                   "batch_tau": round(stats["tau"], 3), "batch_top1": round(stats["top1"], 4),
                   "sec": round(el, 1), "step_s": round(el / (step + 1), 3), "queue_wait_frac": round(wait / el, 3),
                   "label_tok_per_s": round(n_lab / el, 1)}
            print(json.dumps(rec), file=log, flush=True)
            print(json.dumps(rec), flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.total_steps:
            do_eval(step + 1)
        if (step + 1) % args.save_every == 0 or step + 1 == args.total_steps:
            ck = out / f"step{step + 1:07d}"
            sd = {k: v.detach().float().cpu() for k, v in draft.state_dict().items()}
            m = DFlashDraft(cfg)
            m.load_state_dict(sd)
            me = mod.mask_emb.detach().float().cpu() if mod.mask_emb is not None else None
            save_release(m, ck / "release", mask_embedding=me)
            torch.save({"draft": sd, "mask_emb": me, "step": step + 1}, ck / "train_state.pt")
            if args.s3:
                subprocess.Popen(["aws", "s3", "sync", str(ck), f"{args.s3.rstrip('/')}/{ck.name}", "--quiet"])


if __name__ == "__main__":
    main()
