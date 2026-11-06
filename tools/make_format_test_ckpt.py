# SPDX-License-Identifier: Apache-2.0
"""Write a random-init draft in the release format (format test only, NOT a trained model).

    python tools/make_format_test_ckpt.py --out <dir> [--seed 0]

Writes config.json + model.safetensors (+ a README marking it untrained) and reloads it with
model/release.py (strict) to check the round trip.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import TARGET_ID, DraftConfig  # noqa: E402
from model.dflash import DFlashDraft  # noqa: E402
from model.release import load_release, save_release  # noqa: E402

README = """---
license: apache-2.0
base_model: {target}
tags: [dflash, speculative-decoding, format-test, untrained]
---

# UNTRAINED -- format test only

Randomly initialised DFlash draft in the release format for `{target}`. It exists only so that
loaders / serving integrations can be tested before the trained draft is published. Its
proposals are random: acceptance is ~0 and it gives no speedup. Greedy speculative decoding
with it must still produce exactly the target's output.

config: hidden {H}, {L} layers, block_size {bs}, target_layer_ids {tl}, mask_token_id {mt},
rope_theta {rt}. No embed_tokens / lm_head (shared with the target).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    cfg = DraftConfig()
    m = DFlashDraft(cfg)
    out = save_release(m, a.out, dtype=torch.bfloat16)
    (out / "README.md").write_text(README.format(target=TARGET_ID, H=cfg.hidden_size, L=cfg.num_hidden_layers,
                                                 bs=cfg.block_size, tl=cfg.target_layer_ids,
                                                 mt=cfg.mask_token_id, rt=cfg.rope_theta))
    m2 = load_release(out)
    with safe_open(str(out / "model.safetensors"), "pt") as fh:
        names = sorted(fh.keys())
        shapes = {k: list(fh.get_slice(k).get_shape()) for k in names}
    n = sum(p.numel() for p in m2.parameters())
    print(json.dumps({"out": str(out), "n_tensors": len(names), "params_M": round(n / 1e6, 2),
                      "non_layer_tensors": {k: v for k, v in shapes.items() if not k.startswith("layers.")},
                      "layer0_tensors": {k: v for k, v in shapes.items() if k.startswith("layers.0.")}}, indent=1))


if __name__ == "__main__":
    main()
