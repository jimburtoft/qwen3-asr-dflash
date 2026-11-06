"""Verify People's Speech per-source licenses against archive.org item metadata (public API).

1. Random-samples identifiers from the `clean` train id table that are NOT in credits.csv and
   fetches https://archive.org/metadata/<identifier>/metadata/licenseurl.
2. Control: reads a few `clean_sa` train shards (id column only) and joins them to credits.csv;
   they should map to CC BY-SA.

Usage: python ps_archive_license_check.py --ids ../../tmp/ps/ps_clean_train_ids.parquet \
           --credits ../../tmp/ps/ps_credits.csv --n 300 --out ../../tmp/ps
"""
import argparse
import collections
import csv
import json
import os
import random
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from ps_clean_license_scan import REPO, REV, clip_to_identifier, norm_license


def archive_license(ident):
    url = f"https://archive.org/metadata/{urllib.parse.quote(ident)}/metadata"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            md = json.load(r).get("result", {})
    except Exception as e:  # noqa: BLE001
        return ident, f"ERROR:{type(e).__name__}", None
    lic = md.get("licenseurl") or md.get("license") or ""
    if isinstance(lic, list):
        lic = lic[0]
    return ident, (norm_license(lic) if lic else "NO_LICENSEURL"), md.get("creator") if isinstance(md.get("creator"), str) else None


def main():
    import urllib.parse  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--credits", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sa-shards", type=int, default=8)
    a = ap.parse_args()

    cred = {}
    with open(a.credits, newline="") as fh:
        for r in csv.DictReader(fh):
            cred[r["identifier"]] = norm_license(r["licenseurl"])

    idents = sorted(set(pq.read_table(a.ids, columns=["identifier"]).column("identifier").to_pylist()))
    missing = [i for i in idents if i not in cred]
    random.seed(0)
    sample = random.sample(missing, min(a.n, len(missing)))
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(archive_license, sample))
    c = collections.Counter(r[1] for r in res)
    print(f"archive.org licenseurl for {len(sample)} random clean identifiers not in credits.csv:")
    for k, v in c.most_common():
        print(f"  {k:40s} {v}")
    # also check a sample of identifiers that ARE in credits, to validate the join
    present = random.sample([i for i in idents if i in cred], 50)
    with ThreadPoolExecutor(16) as ex:
        res2 = list(ex.map(archive_license, present))
    agree = sum(1 for i, L, _ in res2 if L == cred[i])
    print(f"credits.csv vs archive.org agreement on 50 matched identifiers: {agree}/50")
    print("  mismatches:", [(i, cred[i], L) for i, L, _ in res2 if L != cred[i]][:10])

    # control: clean_sa
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
                   if f.startswith("clean_sa/train-"))
    fs = HfFileSystem()
    sa_ids = set()
    for f in random.sample(files, a.sa_shards):
        with fs.open(f"datasets/{REPO}@{REV}/{f}", block_size=1 << 20) as fh:
            sa_ids |= {clip_to_identifier(x) for x in pq.ParquetFile(fh).read(columns=["id"]).column("id").to_pylist()}
    c2 = collections.Counter(cred.get(i, "NOT_IN_CREDITS") for i in sa_ids)
    print(f"control clean_sa ({a.sa_shards} shards, {len(sa_ids)} identifiers) joined to credits.csv: {dict(c2)}")
    print(f"clean/clean_sa identifier intersection: {len(sa_ids & set(idents))}")

    with open(os.path.join(a.out, "ps_archive_check.json"), "w") as fh:
        json.dump({"missing_sample": res, "matched_check": res2,
                   "clean_sa_join": dict(c2)}, fh, indent=1)


if __name__ == "__main__":
    main()
