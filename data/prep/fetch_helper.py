# SPDX-License-Identifier: Apache-2.0
"""Helper: run extra fetch workers for an in-progress round of build_corpus.py (same job list, reverse
order). Part files are written atomically and skipped if present, so the main process simply finds them
done. Use when per-stream CDN throughput is the bottleneck.

Usage: python fetch_helper.py <name> <round> <limit> [--workers 4]
"""
import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as bc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("round", type=int)
    ap.add_argument("limit", type=int)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    c = pq.read_table(os.path.join(bc.SEL, f"{a.name}.parquet")).to_pandas()
    sub = c[(c["rank"] >= a.lo) & (c["rank"] < a.limit)]
    byf = defaultdict(list)
    for r in sub.to_dict("records"):
        byf[r["file"]].append(r)
    jobs = [(a.name, a.round, f, rows) for f, rows in sorted(byf.items())][::-1]
    with Pool(a.workers) as p:
        for i, _ in enumerate(p.imap_unordered(bc._fetch_file, jobs)):
            if i % 25 == 0:
                bc.log(a.name, f"helper {i + 1}/{len(jobs)}")


if __name__ == "__main__":
    main()
