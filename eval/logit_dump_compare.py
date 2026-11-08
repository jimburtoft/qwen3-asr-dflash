# SPDX-License-Identifier: Apache-2.0
"""Read vllm-neuron debug logits dumps (``neuron_config.debug_logits_dir``) and compare runs at the first
position where their greedy outputs differ, plus against HF fp32 logits.

Dump format (TP rank 0): ``step_{N}_dp0.bin``: int64 num_items, int64 vocab; per item int64 batch_idx,
int64 position_id, float32 logits[vocab]. One request is sent at a time, so all rows belong to the
current clip; a clip's dumps are moved into ``<dir>/<clip_index>/`` by the caller.

    python eval/logit_dump_compare.py --runs plain=<dir> spec=<dir> --hf <margins json> --out x.json
"""

import argparse
import glob
import json
import os

import numpy as np


def read_clip(d):
    """-> {position_id: logits(float32[V])} (last write wins)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "step_*_dp0.bin")), key=lambda p: int(p.split("step_")[1].split("_")[0])):
        with open(f, "rb") as fh:
            n, V = np.fromfile(fh, dtype=np.int64, count=2)
            for _ in range(int(n)):
                _b, pos = np.fromfile(fh, dtype=np.int64, count=2)
                out[int(pos)] = np.fromfile(fh, dtype=np.float32, count=int(V))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir (dir/<clip_index>/step_*.bin)")
    ap.add_argument("--clips", required=True, help="comma list of clip ids in request order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    clips = a.clips.split(",")
    res = []
    for ci, cid in enumerate(clips):
        L = {n: read_clip(os.path.join(d, str(ci))) for n, d in runs.items()}
        common = sorted(set.intersection(*[set(x) for x in L.values()])) if L else []
        names = list(L)
        row = {"clip_id": cid, "positions": len(common)}
        argm = {n: [int(L[n][p].argmax()) for p in common] for n in names}
        first = next((i for i in range(len(common)) if len({argm[n][i] for n in names}) > 1), None)
        row["first_diff_index"] = first
        if first is not None:
            p = common[first]
            row["position_id"] = p
            for n in names:
                lg = L[n][p]
                top = np.argsort(lg)[::-1][:2]
                row[n] = {"argmax": int(top[0]), "top2_margin": float(lg[top[0]] - lg[top[1]])}
            toks = sorted({row[n]["argmax"] for n in names})
            for n in names:
                row[n]["logits_of_candidates"] = {str(t): float(L[n][p][t]) for t in toks}
            ref = names[0]
            for n in names[1:]:
                row[f"maxabs_{ref}_vs_{n}"] = float(np.abs(L[ref][p] - L[n][p]).max())
        # mean max-abs diff over common positions before the divergence
        upto = common[:first] if first else common
        for n in names[1:]:
            if upto:
                row[f"mean_maxabs_before_{names[0]}_vs_{n}"] = float(np.mean([np.abs(L[names[0]][q] - L[n][q]).max() for q in upto]))
        print(json.dumps(row), flush=True)
        res.append(row)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
