# SPDX-License-Identifier: Apache-2.0
"""Offline DFlash acceptance evaluator on held-out real-audio clips.

Mirrors the vLLM / vllm-neuron DFlash greedy verify rule: the draft proposes ``k = S - 1`` tokens
after the bonus (anchor) token in one forward pass over a block of ``S`` positions
``[anchor, MASK x (S-1)]``; accepted = longest prefix equal to the target's greedy tokens;
tau = accepted + 1 (the target's bonus token). Anchors = every response position of every clip.

Serving block size ``S < trained block size`` is evaluated by running the draft on blocks of
exactly ``S`` positions (fewer mask tokens), as a server configured with
``num_speculative_tokens = S - 1`` does -- not by truncating a 16-wide block.

Input: eval batches from ``feature_producer.py --eval-out`` written at the serving block size.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DraftConfig  # noqa: E402
from model.dflash import DFlashDraft  # noqa: E402
from model.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from train.train_draft_queue import DraftStep, evaluate, load_head  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="train_state.pt")
    ap.add_argument("--eval-dirs", nargs="+", required=True, help="one dir per serving block size")
    ap.add_argument("--compute", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", default="")
    add_device_arg(ap)
    args = ap.parse_args()
    dev = get_device(args.device)
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    embed, lm_head = load_head(torch.bfloat16)
    results = {}
    for d in args.eval_dirs:
        d = Path(d)
        first = torch.load(next(iter(sorted(d.glob("r*_*.pt")))), weights_only=False)
        S = int(first["block_size"])
        cfg = DraftConfig(block_size=S)
        m = DFlashDraft(cfg)
        m.load_state_dict(st["draft"])
        m = m.to(dev).eval()
        step = DraftStep(m, embed, lm_head, torch.bfloat16 if args.compute == "bf16" else torch.float32).to(dev)
        if st.get("mask_emb") is not None:
            step.mask_emb = torch.nn.Parameter(st["mask_emb"].to(dev))
        fn = maybe_compile(step, dev)
        res = evaluate(fn, d, dev, S)
        results[f"serve_block_{S}"] = res
        print(json.dumps({"serve_block": S, **{k: v["tau"] for k, v in res.items()}}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"ckpt": args.ckpt, "step": st.get("step"), **results}, indent=1))


if __name__ == "__main__":
    main()
