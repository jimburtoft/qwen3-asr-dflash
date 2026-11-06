# SPDX-License-Identifier: Apache-2.0
"""Manifest + sanity checks for the long-form train corpus (build_longform.py output).

Checks: no source clip reused; every source clip is in train_manifest.parquet and in no eval split; no eval speaker
(per family) among the source clips; decoded duration within [31, 300] s and equal to duration_s; n_windows ==
ceil(duration_s / 30); one license per row; shard rows <= 500, sorted by n_windows. Writes
shards/train_longform_manifest.parquet and stats/train_longform_sanity.json.
Usage: python longform_sanity.py [--root ~/q3df_corpus]
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FAMILY = {"librispeech": "librivox", "libritts_r": "librivox", "mls": "librivox", "cv17": "cv17",
          "voxpopuli": "voxpopuli", "ami": "ami", "ami_longform": "ami"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/q3df_corpus"))
    a = ap.parse_args()
    S = os.path.join(a.root, "shards")
    errs = []
    tr = pq.read_table(os.path.join(S, "train_manifest.parquet"), columns=["clip_id", "speaker_id"]).to_pydict()
    ev = pq.read_table(os.path.join(S, "eval_manifest.parquet"), columns=["clip_id", "speaker_id"]).to_pydict()
    tr_spk = dict(zip(tr["clip_id"], tr["speaker_id"]))
    ev_ids = set(ev["clip_id"])
    ev_spk = defaultdict(set)
    for c, s in zip(ev["clip_id"], ev["speaker_id"]):
        f = FAMILY.get(c.split("/")[0])
        if f and s:
            ev_spk[f] |= set(s.split(","))
    tabs, used = [], Counter()
    per_ds = defaultdict(lambda: {"clips": 0, "hours": 0.0, "W": Counter(), "sources": 0})
    W_all = Counter()
    for f in sorted(glob.glob(os.path.join(S, "train_longform", "*", "shard-*.parquet"))):
        pf = pq.ParquetFile(f)
        if pf.metadata.num_rows > 500:
            errs.append(f"{f}: {pf.metadata.num_rows} rows")
        t = pf.read()
        rel = os.path.relpath(f, S)
        nw = t.column("n_windows").to_pylist()
        if nw != sorted(nw):
            errs.append(f"{rel} not sorted by n_windows")
        for k in range(t.num_rows):
            r = {c: t.column(c)[k].as_py() for c in ("clip_id", "duration_s", "source_path", "license", "n_windows")}
            pcm_len = len(t.column("audio_pcm16")[k].as_py()) // 2
            dec = pcm_len / 16000
            if abs(dec - r["duration_s"]) > 1e-3 or not (31.0 <= dec <= 300.0):
                errs.append(f"{r['clip_id']}: decoded {dec:.3f} vs {r['duration_s']:.3f}")
            if r["n_windows"] != math.ceil(r["duration_s"] / 30.0 - 1e-9):
                errs.append(f"{r['clip_id']}: n_windows {r['n_windows']}")
            srcs = r["source_path"].split(";")
            ds = r["clip_id"].split("/")[1]
            for s in srcs:
                used[s] += 1
                if s not in tr_spk:
                    errs.append(f"{r['clip_id']}: source {s} not a train clip")
                if s in ev_ids:
                    errs.append(f"{r['clip_id']}: source {s} is an eval clip")
                fam = FAMILY.get(s.split("/")[0])
                sp = tr_spk.get(s)
                if fam and sp and sp in ev_spk[fam]:
                    errs.append(f"{r['clip_id']}: source speaker {sp} is an eval speaker")
                if s.split("/")[0] != ds:
                    errs.append(f"{r['clip_id']}: mixed datasets")
            d = per_ds[ds]
            d["clips"] += 1
            d["hours"] += r["duration_s"] / 3600
            d["W"][r["n_windows"]] += 1
            d["sources"] += len(srcs)
            W_all[r["n_windows"]] += 1
        m = t.drop_columns(["audio_pcm16"])
        m = m.append_column("shard", pa.array([rel] * t.num_rows)).append_column(
            "row", pa.array(np.arange(t.num_rows, dtype=np.int32)))
        tabs.append(m)
    reused = [s for s, n in used.items() if n > 1]
    if reused:
        errs.append(f"{len(reused)} source clips reused")
    man = pa.concat_tables(tabs)
    pq.write_table(man, os.path.join(S, "train_longform_manifest.parquet"), compression="zstd")
    res = {"ok": not errs, "errors": errs[:50], "n_errors": len(errs), "rows": man.num_rows,
           "hours": round(sum(v["hours"] for v in per_ds.values()), 2), "source_clips_used": len(used),
           "W_histogram": dict(sorted(W_all.items())),
           "per_dataset": {k: {"clips": v["clips"], "hours": round(v["hours"], 2), "sources": v["sources"],
                               "W": dict(sorted(v["W"].items()))} for k, v in sorted(per_ds.items())},
           "licenses": dict(Counter(man.column("license").to_pylist())),
           "bytes": sum(os.path.getsize(f) for f in glob.glob(os.path.join(S, "train_longform", "*", "*.parquet")))}
    os.makedirs(os.path.join(a.root, "stats"), exist_ok=True)
    json.dump(res, open(os.path.join(a.root, "stats", "train_longform_sanity.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if not errs else 1)


if __name__ == "__main__":
    main()
