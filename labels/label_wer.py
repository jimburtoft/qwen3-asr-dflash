# SPDX-License-Identifier: Apache-2.0
"""Quick label sanity: WER of target greedy transcripts vs dataset references (simple normaliser)."""
import collections
import glob
import re
import sys

import pyarrow.parquet as pq


def norm(s):
    """Lowercase, strip punctuation (Unicode-aware), split. Languages written without spaces
    (zh, ja, th, yue) are scored per character (CER) instead."""
    s = re.sub(r"[^\w' ]", " ", s.lower().replace("-", " "))
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\u0e00-\u0e7f]", s):
        return [c for c in s if not c.isspace()]
    return s.split()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        nd = [i] + [0] * len(h)
        for j, b in enumerate(h, 1):
            nd[j] = min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (a != b))
        d = nd
    return d[-1]


def main(pattern):
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f)
        if t.num_rows == 0:
            continue
        cols = ["dataset", "config", "split", "ref_text", "hyp_text", "finished", "n_resp"]
        for r in t.select([c for c in cols if c in t.column_names]).to_pylist():
            k = (f'{r["dataset"]}:{r.get("config", "")}', r["split"])
            a, b = norm(r["ref_text"]), norm(r["hyp_text"])
            g = agg[k]
            g[0] += edits(a, b); g[1] += len(a); g[2] += 1; g[3] += int(r["finished"]); g[4] = max(g[4], r["n_resp"])
    for k, (e, n, c, fin, mx) in sorted(agg.items()):
        print(f"{k[0]:32s} {k[1]:24s} clips {c:6d} finished {fin/c:.3%}  max_resp {mx:4d}  WER {100*e/max(n,1):.2f}%")


if __name__ == "__main__":
    main(sys.argv[1])
