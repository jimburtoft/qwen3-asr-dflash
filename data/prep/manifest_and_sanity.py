# SPDX-License-Identifier: Apache-2.0
"""Build {train,eval}_manifest.parquet from the shards and run the corpus sanity checks.

Checks:
  1. manifest row counts == shard row counts (per shard and total); every shard has <= 2000 rows, zstd
  2. no clip_id duplicated anywhere (within or across roles)
  3. no eval speaker in train, per speaker-id family:
       librispeech + libritts_r (same LibriVox reader ids), mls (LibriVox reader ids, cross-checked with the
       LibriSpeech family), cv17 (salted client-id hash), voxpopuli, ami (participant ids, incl. long-form)
  4. 5 random clips per dataset/split decode (PCM16 length / 16 kHz) to duration_s (+- 1 ms); train
     durations within [1, 30] s
  5. every row has a license in the approved set

Usage: python manifest_and_sanity.py [--seed 0]
Writes shards/{train,eval}_manifest.parquet and stats/sanity.json; exits non-zero on failure.
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SHARDS, SR, save_stats  # noqa: E402

APPROVED = {"CC-BY-4.0", "CC0-1.0", "CC-BY-2.0", "CC-BY-2.5", "CC-BY-3.0", "CC-BY-3.0-US", "CC-BY-3.0-ES",
            "CC-BY-3.0-DE", "PDM-1.0", "LicenseRef-Public-Domain", "LicenseRef-US-Government-Work"}
SPK_FAMILY = {"librispeech": "librivox", "libritts_r": "librivox", "mls": "librivox", "cv17": "cv17",
              "voxpopuli": "voxpopuli", "ami": "ami", "ami_longform": "ami"}


def family(short):
    """Speaker-id family of a corpus short name (multilingual names map onto their corpus family)."""
    if short in SPK_FAMILY:
        return SPK_FAMILY[short]
    for pre, fam in (("mls_", "librivox"), ("cv17_", "cv17"), ("voxpopuli_", "voxpopuli")):
        if short.startswith(pre):
            return fam
    return None


def build_manifest(role):
    files = sorted(glob.glob(os.path.join(SHARDS, role, "*", "*", "shard-*.parquet")))
    tabs = []
    for f in files:
        pf = pq.ParquetFile(f)
        t = pf.read(columns=[c for c in pf.schema_arrow.names if c != "audio_pcm16"])
        rel = os.path.relpath(f, SHARDS)
        t = t.append_column("shard", pa.array([rel] * t.num_rows)).append_column(
            "row", pa.array(np.arange(t.num_rows, dtype=np.int32)))
        tabs.append(t)
    m = pa.concat_tables(tabs)
    out = os.path.join(SHARDS, f"{role}_manifest.parquet")
    pq.write_table(m, out, compression="zstd")
    return m, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    res, ok = {"errors": []}, True

    def fail(msg):
        nonlocal ok
        ok = False
        res["errors"].append(msg)
        print("FAIL", msg, flush=True)

    man, files = {}, {}
    for role in ("train", "eval"):
        man[role], files[role] = build_manifest(role)
        # 1. row counts and shard properties
        n_shard_rows = 0
        for f in files[role]:
            md = pq.ParquetFile(f).metadata
            n_shard_rows += md.num_rows
            if md.num_rows > 2000:
                fail(f"{f} has {md.num_rows} rows")
            if md.row_group(0).column(0).compression != "ZSTD":
                fail(f"{f} not zstd")
        mm = pq.read_table(os.path.join(SHARDS, f"{role}_manifest.parquet"))
        if mm.num_rows != n_shard_rows:
            fail(f"{role}: manifest {mm.num_rows} != shards {n_shard_rows}")
        res[f"{role}_rows"] = n_shard_rows
        res[f"{role}_shards"] = len(files[role])
        res[f"{role}_hours"] = round(float(pc.sum(mm["duration_s"]).as_py()) / 3600, 2)
        lic = set(mm["license"].to_pylist())
        if not lic <= APPROVED:
            fail(f"{role}: unapproved licenses {lic - APPROVED}")
        res[f"{role}_licenses"] = sorted(lic)
        if role == "train":
            d = mm["duration_s"].to_numpy()
            if d.min() < 1.0 - 1e-6 or d.max() > 30.0 + 1e-6:
                fail(f"train durations out of [1,30]: {d.min()}..{d.max()}")
            res["train_duration_range"] = [float(d.min()), float(d.max())]

    # 2. clip_id uniqueness
    ids = man["train"]["clip_id"].to_pylist() + man["eval"]["clip_id"].to_pylist()
    dup = len(ids) - len(set(ids))
    res["duplicate_clip_ids"] = dup
    if dup:
        fail(f"{dup} duplicated clip_ids")

    # 3. eval speakers not in train (per family)
    spk = {r: defaultdict(set) for r in ("train", "eval")}
    for role in ("train", "eval"):
        for cid, s in zip(man[role]["clip_id"].to_pylist(), man[role]["speaker_id"].to_pylist()):
            fam = family(cid.split("/")[0])
            if fam and s:
                for x in s.split(","):
                    spk[role][fam].add(x)
    res["speaker_overlap"] = {}
    for fam in sorted(set(SPK_FAMILY.values())):
        ov = spk["train"][fam] & spk["eval"][fam]
        res["speaker_overlap"][fam] = {"train_speakers": len(spk["train"][fam]), "eval_speakers": len(spk["eval"][fam]),
                                       "overlap": sorted(ov)[:20], "n_overlap": len(ov)}
        if ov:
            fail(f"speaker overlap in {fam}: {len(ov)}")

    # 3b. LibriSpeech vs LibriTTS-R train draws are chapter-disjoint (same LibriVox chapters otherwise)
    tr_ids = man["train"]["clip_id"].to_pylist()
    ls_ch = {c.split("/")[2].split("-")[1] for c in tr_ids if c.startswith("librispeech/")}
    lt_ch = {c.split("/")[2].split("_")[1] for c in tr_ids if c.startswith("libritts_r/")}
    res["librispeech_vs_libritts_r_train_chapter_overlap"] = len(ls_ch & lt_ch)
    if ls_ch & lt_ch:
        fail(f"LibriSpeech/LibriTTS-R train chapter overlap: {len(ls_ch & lt_ch)}")

    # 4. decode check (5 random clips per dataset/split)
    res["decode_check"] = {}
    for role in ("train", "eval"):
        m = man[role]
        groups = defaultdict(list)
        for i, cid in enumerate(m["clip_id"].to_pylist()):
            groups["/".join(cid.split("/")[:2])].append(i)
        sh, rw, du = m["shard"].to_pylist(), m["row"].to_pylist(), m["duration_s"].to_pylist()
        for g, idx in sorted(groups.items()):
            pick = rng.sample(idx, min(5, len(idx)))
            out = []
            for i in pick:
                t = pq.ParquetFile(os.path.join(SHARDS, sh[i])).read(columns=["clip_id", "audio_pcm16"])
                b = t["audio_pcm16"][rw[i]].as_py()
                assert t["clip_id"][rw[i]].as_py() == m["clip_id"][i].as_py()
                x = np.frombuffer(b, dtype="<i2")
                dec = len(x) / SR
                good = abs(dec - du[i]) <= 1e-3 and len(b) % 2 == 0 and np.abs(x).max() > 0
                out.append({"clip_id": m["clip_id"][i].as_py(), "duration_s": round(du[i], 3),
                            "decoded_s": round(dec, 3), "peak": int(np.abs(x).max()), "ok": bool(good)})
                if not good:
                    fail(f"decode mismatch {m['clip_id'][i].as_py()}")
            res["decode_check"][f"{role}:{g}"] = out

    # sizes
    res["disk_bytes"] = {r: sum(os.path.getsize(f) for f in files[r]) for r in ("train", "eval")}
    res["ok"] = ok
    save_stats("sanity", res)
    print(json.dumps({k: v for k, v in res.items() if k != "decode_check"}, indent=1))
    print("decode groups checked:", len(res["decode_check"]), "all ok:",
          all(x["ok"] for v in res["decode_check"].values() for x in v))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
