"""Full per-source license table for People's Speech `clean` train.

For every archive.org identifier in the clean train id table: license from credits.csv if present,
else from the public archive.org metadata API (licenseurl field). Writes
<out>/ps_clean_source_licenses.csv (identifier, source, license, clips, hours) and prints totals.
Resumable: identifiers already in the output cache are skipped.

Usage: python ps_full_license_table.py --ids ../../tmp/ps/ps_clean_train_ids.parquet \
           --credits ../../tmp/ps/ps_credits.csv --out ../../tmp/ps
"""
import argparse
import collections
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

from ps_archive_license_check import archive_license
from ps_clean_license_scan import norm_license

ALLOWED = {"cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0", "cc-by-3.0-us", "cc-by-3.0-es",
           "cc-by-3.0-de", "public-domain", "cc-publicdomain", "cc-cc0-3.0", "cc-cc0-1.0",
           "other:https://www.usa.gov/government-works"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--credits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cred = {}
    with open(a.credits, newline="") as fh:
        for r in csv.DictReader(fh):
            cred[r["identifier"]] = norm_license(r["licenseurl"])
    t = pq.read_table(a.ids, columns=["identifier", "duration_ms"]).to_pydict()
    clips, ms = collections.Counter(), collections.Counter()
    for i, d in zip(t["identifier"], t["duration_ms"]):
        clips[i] += 1
        ms[i] += d
    cache_p = os.path.join(a.out, "ps_archive_cache.jsonl")
    cache = {}
    if os.path.exists(cache_p):
        for line in open(cache_p):
            r = json.loads(line)
            cache[r[0]] = r[1]
    todo = [i for i in clips if i not in cred and (i not in cache or cache[i].startswith("ERROR"))]
    print(f"identifiers={len(clips)} in_credits={sum(1 for i in clips if i in cred)} to_query={len(todo)}", flush=True)
    with open(cache_p, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        for n, (i, L, _) in enumerate(ex.map(archive_license, todo)):
            cache[i] = L
            fh.write(json.dumps([i, L]) + "\n")
            if n % 2000 == 0:
                fh.flush()
                print("queried", n, flush=True)
    tot_c, tot_h = collections.Counter(), collections.Counter()
    with open(os.path.join(a.out, "ps_clean_source_licenses.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["identifier", "source", "license", "allowed", "clips", "hours"])
        for i in sorted(clips):
            src, L = ("credits.csv", cred[i]) if i in cred else ("archive.org", cache.get(i, "UNKNOWN"))
            ok = L in ALLOWED
            w.writerow([i, src, L, int(ok), clips[i], round(ms[i] / 3.6e6, 4)])
            key = L if ok else f"NOT_ALLOWED:{L}"
            tot_c[key] += clips[i]
            tot_h[key] += ms[i] / 3.6e6
    for k, v in tot_h.most_common():
        print(f"  {k:50s} clips={tot_c[k]:9d} hours={v:8.1f}")


if __name__ == "__main__":
    main()
