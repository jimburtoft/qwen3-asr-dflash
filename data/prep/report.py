# SPDX-License-Identifier: Apache-2.0
"""Render stats/*.json (from build_corpus.py / ami_longform.py / manifest_and_sanity.py) as markdown tables
for LICENSES.md and the task note, and a small per-set summary CSV.

Usage: python report.py > corpus_report.md   (also writes stats/summary.csv)
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STATS  # noqa: E402

ORDER = ["ls_train", "ltts_train", "mls_train", "cv_train", "vp_train", "ps_train", "ami_train",
         "ls_eval", "cv_eval", "vp_eval", "fleurs_eval", "ami_eval"]


def load(n):
    p = os.path.join(STATS, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    rows = []
    print("| Set | Role | Dataset / config | Split | Clips | Hours | Speakers | Pool rows | Dropped by filters (select stage) | Decode-stage rejects | Eligible after filters |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in ORDER:
        s, sel = load(n), load(f"{n}_select")
        if not s:
            continue
        drops = ", ".join(f"{k}={v}" for k, v in (sel or {}).get("dropped", {}).items() if v) or "none"
        dec = ", ".join(f"{k}={v}" for k, v in s.get("decode_stage_rejects_within_drawn_ranks", {}).items()) or "none"
        for split, v in s["splits"].items():
            print(f"| {n} | {s['role']} | `{s['dataset']}` / {s['config']} | {split} | {v['clips']:,} | {v['hours']:.2f} | "
                  f"{v['speakers'] or '-'} | {sel['pool_rows']:,} | {drops} | {dec} | {sel['eligible_after_filters']:,} |")
            rows.append([n, s["role"], s["dataset"], s["config"], split, v["clips"], v["hours"], v["speakers"],
                         sel["pool_rows"], json.dumps(sel.get("dropped", {})),
                         json.dumps(s.get("decode_stage_rejects_within_drawn_ranks", {}))])
    lf = load("ami_longform")
    if lf:
        print(f"| ami_longform | eval | `edinburghcstr/ami` ihm segments + official Mix-Headset audio | test | "
              f"{lf['windows']} | {lf['hours']:.2f} | 16 meetings | - | - | - | - |")
        rows.append(["ami_longform", "eval", "edinburghcstr/ami", "ihm+Mix-Headset", "test", lf["windows"], lf["hours"],
                     "", "", "", ""])
    for role in ("train", "eval"):
        c = sum(r[5] for r in rows if r[1] == role)
        h = sum(r[6] for r in rows if r[1] == role)
        print(f"| **total {role}** | | | | **{c:,}** | **{h:.1f}** | | | | | |")
    with open(os.path.join(STATS, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "role", "dataset", "config", "split", "clips", "hours", "speakers", "pool_rows",
                    "dropped_select", "dropped_decode"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
