# SPDX-License-Identifier: Apache-2.0
"""Parity: draft fwd/bwd + AdamW on an accelerator (Trainium or CUDA) vs the CPU fp32 reference.

Uses one batch written by feature_producer.py (target features from the accelerator target), so both
sides see identical inputs. Checks loss, per-tensor grad cosine / rel err, post-AdamW update
cosine, for fp32 compute and bf16-autocast compute.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DraftConfig  # noqa: E402
from model.dflash import DFlashDraft  # noqa: E402
from model.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from train.train_draft_queue import DraftStep, load_head, to_dev  # noqa: E402


def one(dev, compute, sd, cfg, embed, lm_head, item, compiled, steps=1):
    d = DFlashDraft(cfg)
    d.load_state_dict(sd)
    d = d.to(dev)
    m = DraftStep(d, embed, lm_head, compute).to(dev)
    f = maybe_compile(m, torch.device(dev), compiled)
    opt = torch.optim.AdamW(d.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.0)
    losses = []
    g0 = None
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = f(*to_dev(item, dev))
        loss.backward()
        if s == 0:
            g0 = {n: p.grad.detach().float().cpu().clone() for n, p in d.named_parameters()}
        torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    newp = {n: p.detach().float().cpu().clone() for n, p in d.named_parameters()}
    return losses, g0, newp


def cos64(a, b):
    a = a.double()
    b = b.double()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", required=True)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--steps", type=int, default=5)
    add_device_arg(ap)
    args = ap.parse_args()
    dev = get_device(args.device)
    torch.manual_seed(0)
    cfg = DraftConfig(num_hidden_layers=args.layers)
    item = torch.load(args.item, weights_only=False)
    embed, lm_head = load_head(torch.bfloat16)
    sd = {k: v.clone() for k, v in DFlashDraft(cfg).state_dict().items()}
    ref_l, ref_g, ref_p = one(torch.device("cpu"), torch.float32, sd, cfg, embed, lm_head, item, False, args.steps)
    print("cpu fp32 losses", [round(x, 5) for x in ref_l])
    for name, cdt in (("fp32", torch.float32), ("bf16-autocast", torch.bfloat16)):
        l, g, p = one(dev, cdt, sd, cfg, embed, lm_head, item, True, args.steps)
        allc = torch.cat([ref_g[k].flatten() for k in ref_g])
        alln = torch.cat([g[k].flatten() for k in ref_g])
        gcos = cos64(allc, alln)
        worst = min((cos64(ref_g[k].flatten(), g[k].flatten()), k) for k in ref_g)
        grel = float((alln.double() - allc.double()).norm() / allc.double().norm())
        du = torch.cat([(p[k] - sd[k]).flatten() for k in sd])
        dr = torch.cat([(ref_p[k] - sd[k]).flatten() for k in sd])
        ucos = cos64(du, dr)
        rel_l = max(abs(a - b) / abs(b) for a, b in zip(l, ref_l))
        ratios = sorted((float(g[k].double().norm() / ref_g[k].double().norm().clamp(min=1e-30)), k) for k in ref_g)
        print(f"  grad norm cpu {allc.double().norm():.4e} {dev.type} {alln.double().norm():.4e}; per-tensor ratio min {ratios[0]} max {ratios[-1]}")
        print(f"{dev.type} {name:14s} losses {[round(x, 5) for x in l]}  max loss rel {rel_l:.2e}  "
              f"grad cos {gcos:.6f} rel {grel:.2e} (worst {worst[0]:.5f} {worst[1]})  {args.steps}-step update cos {ucos:.6f}")
        print("  PARITY_OK" if rel_l < 1e-2 and gcos > 0.995 and worst[0] > 0.98 and ucos > 0.99 else "  PARITY_CHECK")


if __name__ == "__main__":
    main()
