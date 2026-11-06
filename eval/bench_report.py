# SPDX-License-Identifier: Apache-2.0
"""Summarise a pair of vllm_neuron_serve_bench.py BS=1 runs (plain vs DFlash) per held-out set.

    python eval/bench_report.py <plain.json> <spec.json> [--md out.md]

Per set: clips, identical transcripts, WER of each run vs the dataset reference (labels/label_wer.py
normaliser; CER for zh / ja / th / yue), mean / p50 / p90 latency, speedup, served tau (accepted + 1).
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labels.label_wer import edits, norm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plain")
    ap.add_argument("spec")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    P = json.load(open(a.plain))["clips"]
    S = {c["clip_id"]: c for c in json.load(open(a.spec))["clips"]}
    by = collections.defaultdict(list)
    for p in P:
        s = S.get(p["clip_id"])
        if s is None:
            continue
        by[f'{p["dataset"].split("/")[-1]}:{p.get("config", "")}:{p["split"]}'].append((p, s))
    rows = []
    allp = []
    for k in sorted(by):
        v = by[k]
        allp += v
        rows.append(stats(k, v))
    rows.append(stats("ALL", allp))
    hdr = ("| set | clips | identical | WER plain | WER DFlash | mean ms plain | mean ms DFlash | speedup | p90 ms DFlash | tau |\n"
           "|---|---|---|---|---|---|---|---|---|---|")
    md = hdr + "\n" + "\n".join(
        f"| {r['set']} | {r['n']} | {r['ident']} | {r['wer_p']:.2f} | {r['wer_s']:.2f} | {r['ms_p']:.1f} | {r['ms_s']:.1f} | "
        f"{r['speedup']:.2f}x | {r['p90_s']:.1f} | {r['tau']:.2f} |" for r in rows)
    print(md)
    if a.md:
        Path(a.md).write_text(md + "\n")


def stats(name, v):
    e_p = e_s = n = 0
    for p, s in v:
        r = norm(p["ref_text"])
        e_p += edits(r, norm(p["text"]))
        e_s += edits(r, norm(s["text"]))
        n += len(r)
    lp = np.array([p["latency_s"] for p, _ in v]) * 1000
    ls = np.array([s["latency_s"] for _, s in v]) * 1000
    d = sum(s["drafts"] for _, s in v)
    acc = sum(s["accepted"] for _, s in v)
    return {"set": name, "n": len(v), "ident": sum(p["text"] == s["text"] for p, s in v),
            "wer_p": 100 * e_p / max(n, 1), "wer_s": 100 * e_s / max(n, 1), "ms_p": lp.mean(), "ms_s": ls.mean(),
            "speedup": lp.mean() / ls.mean(), "p90_s": float(np.percentile(ls, 90)), "tau": 1 + acc / max(d, 1)}


if __name__ == "__main__":
    main()
