# SPDX-License-Identifier: Apache-2.0
"""Long-form TRAIN corpus (31-300 s) built only from clips already in the train shards.

Every source clip is a train clip that passed all the data/filters + build_corpus.py filters (VoxPopuli speaker drops,
AMI IB4005, People's Speech license allow-list, transcript-overlap drops). Long clips are concatenations of
consecutive / related source clips with 0.2-0.5 s of silence between them (seed 0):
  ami             same train meeting, segments in begin-time order (hard break at meeting boundary)
  librispeech     same reader, chapter, utterance order
  libritts_r      same reader, chapter, paragraph/sentence order
  mls             same reader, book, segment order
  voxpopuli       same speaker, audio_id order (chronological sessions)
  peoples_speech  same source item, segment order (hard break at license change)
  cv17            hashed speakers have ~1 clip each, so effectively random clips
A sample prefers to stay inside one group (speaker / meeting / source). It crosses into the next group only if
the target window count has not been reached yet. It never crosses a hard break.

Targets: window count W = ceil(dur / 30) in {2,3,4,5,6,8,10}, roughly uniform per dataset. The target duration
for a sample is uniform in ((W-1)*30, W*30], capped at 300 s, and clips are added until the next one would
exceed it. Accepted samples have a length in [31, 300] s (W 7 / 9 can occur and are kept).

Output: shards/train_longform/<short>/shard-NNNNN.parquet (<= 500 rows, zstd, sorted by n_windows) and
shards/train_longform_manifest.parquet. Usage: python build_longform.py [--root ~/q3df_corpus] [--workers 4]
"""
import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SR = 16000
SEED = 0
BINS = [2, 3, 4, 5, 6, 8, 10]
MAX_S, MIN_S = 300.0, 31.0
ROWS_PER_SHARD = 500
QUOTA = {"ami": 1200, "peoples_speech": 600, "voxpopuli": 600, "librispeech": 520, "mls": 440,
         "libritts_r": 240, "cv17": 400}
SCHEMA = pa.schema([
    ("clip_id", pa.string()), ("dataset", pa.string()), ("config", pa.string()), ("split", pa.string()),
    ("revision", pa.string()), ("license", pa.string()), ("speaker_id", pa.string()),
    ("duration_s", pa.float32()), ("ref_text", pa.string()), ("audio_pcm16", pa.large_binary()),
    ("source_path", pa.string()), ("n_windows", pa.int32()),
])


def tail_int(s):
    m = re.search(r"(\d+)(?:\.\w+)?$", s)
    return int(m.group(1)) if m else 0


def order_keys(ds, df):
    """Return (sort_key, group_key, hard_key) lists for each row of one dataset."""
    nid = df["clip_id"].str.split("/", n=2).str[2]
    sk, gk, hk = [], [], []
    for cid, n, spk, lic, src in zip(df["clip_id"], nid, df["speaker_id"], df["license"], df["source_path"]):
        if ds == "librispeech":
            s, c, u = n.split("-")
            sk.append((s, int(c), int(u))); gk.append(s); hk.append("")
        elif ds == "libritts_r":
            s, c, p, q = n.split("_")
            sk.append((s, int(c), int(p), int(q))); gk.append(s); hk.append("")
        elif ds == "mls":
            s, b, q = n.split("_")
            sk.append((s, int(b), int(q))); gk.append(s); hk.append("")
        elif ds == "voxpopuli":
            sk.append((spk, n)); gk.append(spk); hk.append("")
        elif ds == "peoples_speech":
            src_id = n.split("/")[0].split("_SLASH_")[0]
            sk.append((lic, src_id, n.rsplit("_", 1)[0], tail_int(n))); gk.append(src_id); hk.append(lic)
        elif ds == "ami":
            p = n.split("_")  # AMI_<meeting>_<mic>_<spk>_<begin>_<end>
            sk.append((p[1], int(p[4]), int(p[5]), p[3])); gk.append(p[1]); hk.append(p[1])
        elif ds == "cv17":
            sk.append((0,)); gk.append(spk); hk.append("")
    return sk, gk, hk


def plan_dataset(ds, df, quota, rng):
    """Greedy chunking of the ordered clip sequence. Returns list of samples (list of row indices + gaps)."""
    sk, gk, hk = order_keys(ds, df)
    idx = list(range(len(df)))
    if ds == "cv17":
        rng.shuffle(idx)
    else:
        idx.sort(key=lambda i: sk[i])
    dur = df["duration_s"].to_numpy(dtype=np.float64)
    need = {w: quota // len(BINS) + (1 if k < quota % len(BINS) else 0) for k, w in enumerate(BINS)}
    samples = []

    def new_target():
        live = [w for w in BINS if need[w] > 0] or BINS
        wts = [max(need[w], 0) + 1e-3 for w in live]
        w = rng.choices(live, weights=wts)[0]
        lo = (w - 1) * 30.0 if w > 2 else MIN_S
        return w, rng.uniform(lo, min(w * 30.0, MAX_S))

    def close(cur, total):
        if total >= MIN_S:
            w = math.ceil(total / 30.0 - 1e-9)
            samples.append(cur)
            if w in need:
                need[w] -= 1
            return True
        return False

    w_t, T = new_target()
    cur, total = [], 0.0
    for pos, i in enumerate(idx):
        if sum(max(v, 0) for v in need.values()) <= 0:
            break
        if cur:
            prev = cur[-1][0]
            if hk[i] != hk[prev]:  # hard break: never cross
                close(cur, total)
                cur, total = [], 0.0
                w_t, T = new_target()
            elif gk[i] != gk[prev] and total > (w_t - 1) * 30.0 and total >= MIN_S:  # target reached: stay in group
                close(cur, total)
                cur, total = [], 0.0
                w_t, T = new_target()
        gap = rng.uniform(0.2, 0.5) if cur else 0.0
        if dur[i] > T:  # a single clip longer than the target: start fresh with this clip
            if cur:
                close(cur, total)
            cur, total = [], 0.0
            w_t, T = new_target()
            gap = 0.0
            if dur[i] > T:
                continue
        if total + gap + dur[i] > T:
            close(cur, total)
            cur, total = [], 0.0
            w_t, T = new_target()
            gap = 0.0
        g = int(round(gap * SR))
        cur.append((i, g))
        total += g / SR + dur[i]
    else:
        close(cur, total)
    return samples, need


def assemble(args):
    ds, root, rows, plan, out_dir, meta = args
    # rows: dataframe-like dict of source rows (clip_id, shard, row, ...), plan: list of [(local_idx, gap_samples)]
    need = defaultdict(list)
    for s in plan:
        for i, _ in s:
            need[rows["shard"][i]].append(i)
    audio = {}
    for shard, ii in need.items():
        t = pq.read_table(os.path.join(root, "shards", shard), columns=["clip_id", "audio_pcm16"])
        cid = t.column("clip_id").to_pylist()
        col = t.column("audio_pcm16")
        for i in ii:
            k = rows["row"][i]
            assert cid[k] == rows["clip_id"][i], (shard, k)
            audio[i] = col[k].as_py()
        del t, col
    out = []
    for n, s in enumerate(plan):
        parts, texts, srcs, spks = [], [], [], set()
        for i, g in s:
            if g:
                parts.append(b"\x00\x00" * g)
            parts.append(audio[i])
            texts.append(rows["ref_text"][i].strip())
            srcs.append(rows["clip_id"][i])
            spks.add(rows["speaker_id"][i])
        pcm = b"".join(parts)
        dur = len(pcm) / 2 / SR
        lic = {rows["license"][i] for i, _ in s}
        assert len(lic) == 1, lic
        spk = next(iter(spks)) if len(spks) == 1 else None
        out.append({"clip_id": None, "dataset": meta["dataset"], "config": meta["config"], "split": "train_longform",
                    "revision": meta["revision"], "license": lic.pop(), "speaker_id": spk,
                    "duration_s": np.float32(dur), "ref_text": " ".join(texts), "audio_pcm16": pcm,
                    "source_path": ";".join(srcs), "n_windows": int(math.ceil(dur / 30.0 - 1e-9))})
    out.sort(key=lambda r: (r["n_windows"], r["duration_s"]))
    for n, r in enumerate(out):
        r["clip_id"] = f"longform/{ds}/{n:05d}"
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.startswith("shard-"):
            os.remove(os.path.join(out_dir, f))
    for k in range(0, len(out), ROWS_PER_SHARD):
        chunk = out[k:k + ROWS_PER_SHARD]
        t = pa.table({f.name: [r[f.name] for r in chunk] for f in SCHEMA}, schema=SCHEMA)
        p = os.path.join(out_dir, f"shard-{k // ROWS_PER_SHARD:05d}.parquet")
        pq.write_table(t, p + ".tmp", compression="zstd", row_group_size=50)
        os.replace(p + ".tmp", p)
    return ds, len(out), sum(float(r["duration_s"]) for r in out) / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/q3df_corpus"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    man = pq.read_table(os.path.join(a.root, "shards", "train_manifest.parquet")).to_pandas()
    man["ds"] = man["clip_id"].str.split("/").str[0]
    rng = random.Random(SEED)
    plans, stats = {}, {}
    quota = dict(QUOTA)
    order = ["ami", "peoples_speech", "voxpopuli", "librispeech", "mls", "libritts_r", "cv17"]
    for ds in order:
        df = man[man["ds"] == ds].reset_index(drop=True)
        samples, need = plan_dataset(ds, df, quota[ds], rng)
        short = len(samples) < quota[ds]
        if ds == "ami" and short:  # AMI train holds only ~24 h: move the shortfall to the other conversational sets
            d = quota[ds] - len(samples)
            quota["peoples_speech"] += d // 2
            quota["voxpopuli"] += d - d // 2
        plans[ds] = (df, samples)
        stats[ds] = {"quota": quota[ds], "planned": len(samples), "unfilled_bins": {w: v for w, v in need.items() if v > 0}}
        print(ds, stats[ds], flush=True)
    used = [df["clip_id"][i] for ds, (df, ss) in plans.items() for s in ss for i, _ in s]
    assert len(used) == len(set(used)), "source clip reused"
    jobs = []
    for ds, (df, ss) in plans.items():
        rows = {c: df[c].tolist() for c in ("clip_id", "shard", "row", "ref_text", "speaker_id", "license")}
        meta = {"dataset": df["dataset"].iloc[0], "config": df["config"].iloc[0], "revision": df["revision"].iloc[0]}
        jobs.append((ds, a.root, rows, ss, os.path.join(a.root, "shards", "train_longform", ds), meta))
    with Pool(a.workers) as p:
        for ds, n, h in p.imap_unordered(assemble, jobs):
            stats[ds].update({"clips": n, "hours": round(h, 2)})
            print("assembled", ds, n, round(h, 2), flush=True)
    os.makedirs(os.path.join(a.root, "stats"), exist_ok=True)
    json.dump(stats, open(os.path.join(a.root, "stats", "train_longform_build.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
