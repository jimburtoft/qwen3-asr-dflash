"""Scan MLCommons/peoples_speech `clean` (or another config) for per-source license.

Reads only the `id` and `duration_ms` columns of each parquet shard (no audio), derives the
archive.org identifier from each clip id, and joins against the MLCommons credits file
(https://storage.googleapis.com/example_download_data/credits.csv, linked from
https://mlcommons.org/datasets/peoples-speech/).

Output: <out>/ps_<config>_<split>_ids.parquet (id, identifier, duration_ms) and a printed
license summary (hours by license URL, plus identifiers not found in credits.csv).

Usage: python ps_clean_license_scan.py --config clean --split train --out ../../tmp/ps
"""
import argparse
import collections
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "MLCommons/peoples_speech"
REV = "f10597c5d3d3a63f8b6827701297c3afdf178272"


def clip_to_identifier(clip_id: str) -> str:
    # clip ids look like "<identifier>/<file>_<NNNNN>.flac" with "/" -> "_SLASH_" and "." -> "_DOT_"
    head = clip_id.split("/")[0] if "/" in clip_id else clip_id.split("_SLASH_")[0]
    return head.replace("_DOT_", ".")


def norm_license(url: str) -> str:
    u = url.strip().lower().replace("http://", "https://").rstrip("/")
    if "creativecommons.org/licenses/" in u:
        parts = u.split("creativecommons.org/licenses/")[1].split("/")
        return "cc-" + parts[0] + (("-" + parts[1]) if len(parts) > 1 else "")
    if "publicdomain" in u:
        return "public-domain"
    return "other:" + u[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="clean")
    ap.add_argument("--split", default="train")
    ap.add_argument("--credits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(
        f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
        if f.startswith(f"{a.config}/{a.split}-") and f.endswith(".parquet")
    )
    print(f"{len(files)} shards")
    fs = HfFileSystem()

    def read(f):
        with fs.open(f"datasets/{REPO}@{REV}/{f}", block_size=1 << 20) as fh:
            t = pq.ParquetFile(fh).read(columns=["id", "duration_ms"])
        return t

    tables = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, t in enumerate(ex.map(read, files)):
            tables.append(t)
            if i % 50 == 0:
                print("read", i, flush=True)
    tb = pa.concat_tables(tables)
    ids = tb.column("id").to_pylist()
    dur = tb.column("duration_ms").to_pylist()
    idents = [clip_to_identifier(x) for x in ids]
    out = pa.table({"id": ids, "identifier": idents, "duration_ms": dur})
    pq.write_table(out, os.path.join(a.out, f"ps_{a.config}_{a.split}_ids.parquet"))

    lic = {}
    with open(a.credits, newline="") as fh:
        for r in csv.DictReader(fh):
            lic[r["identifier"]] = norm_license(r["licenseurl"])

    hours = collections.Counter()
    clips = collections.Counter()
    missing = collections.Counter()
    for ident, d in zip(idents, dur):
        L = lic.get(ident)
        if L is None:
            L = "NOT_IN_CREDITS"
            missing[ident] += 1
        hours[L] += d / 3.6e6
        clips[L] += 1
    print(f"rows={len(ids)} hours={sum(dur)/3.6e6:.1f} identifiers={len(set(idents))}")
    for k, v in hours.most_common():
        print(f"  {k:30s} clips={clips[k]:9d} hours={v:9.1f}")
    print(f"identifiers not in credits: {len(missing)}; sample: {list(missing)[:10]}")
    with open(os.path.join(a.out, f"ps_{a.config}_{a.split}_license_summary.csv"), "w") as fh:
        w = csv.writer(fh)
        w.writerow(["license", "clips", "hours"])
        for k, v in hours.most_common():
            w.writerow([k, clips[k], round(v, 2)])


if __name__ == "__main__":
    main()
