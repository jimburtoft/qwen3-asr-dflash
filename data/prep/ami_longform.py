# SPDX-License-Identifier: Apache-2.0
"""Eval set `ami_longform`: 30 s - 5 min windows of the AMI test meetings (full-corpus-ASR SC set:
EN2002, ES2004, IS1009, TS3003), cut from the official headset-mix recordings at segment boundaries.

Audio: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<meeting>/audio/<meeting>.Mix-Headset.wav
(AMI Meeting Corpus, CC BY 4.0). Segment timing + transcripts: edinburghcstr/ami `ihm` test (pinned revision;
the HF meta table built by build_corpus.py `meta ami_eval`).

Windowing (deterministic, seed 0): segments of each meeting are sorted by begin time and merged into
"utterance blocks" (overlapping / touching segments go into the same block, so a cut never splits speech).
Windows are formed greedily from consecutive blocks, with a target length drawn uniformly from
[30 s, 300 s] per window; a window closes at the first block boundary at or after its target, as long as
it stays <= 300 s. Windows are non-overlapping and every one starts/ends at a block boundary with
0.25 s padding (clipped to neighbouring speech). ref_text = segment transcripts concatenated in begin-time
order. We keep up to ~60 windows spread evenly over the 16 meetings.

Usage: python ami_longform.py [--n 60]
"""
import argparse
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REV, ROOT, SHARDS, ShardWriter, array_to_pcm16, save_stats  # noqa: E402

AMI = "edinburghcstr/ami"
URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
PAD = 0.25
MIN_W, MAX_W = 30.0, 300.0


def fetch(m):
    d = os.path.join(CACHE, "ami_mix")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{m}.Mix-Headset.wav")
    if os.path.exists(p):
        return p
    import time
    for attempt in range(8):       # the AMI mirror drops long transfers now and then; resume with Range
        have = os.path.getsize(p + ".part") if os.path.exists(p + ".part") else 0
        hdr = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(URL.format(m=m), stream=True, timeout=(20, 120), headers=hdr) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(p + ".part", mode) as fh:
                    for c in r.iter_content(8 << 20):
                        fh.write(c)
            break
        except requests.exceptions.RequestException:
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"download failed: {m}")
    os.replace(p + ".part", p)
    return p


def blocks_of(seg):
    out = []
    for b, e, t, s in zip(seg["begin_time"], seg["end_time"], seg["text"], seg["speaker_id"]):
        if out and b <= out[-1]["end"] + 1e-3:
            out[-1]["end"] = max(out[-1]["end"], e)
            out[-1]["segs"].append((b, e, t, s))
        else:
            out.append({"begin": b, "end": e, "segs": [(b, e, t, s)]})
    return out


def windows_of(blocks, rng):
    wins, i = [], 0
    while i < len(blocks):
        target = rng.uniform(MIN_W, MAX_W)
        j = i
        while j + 1 < len(blocks) and blocks[j + 1]["end"] - blocks[i]["begin"] <= MAX_W and \
                blocks[j]["end"] - blocks[i]["begin"] < target:
            j += 1
        span = blocks[j]["end"] - blocks[i]["begin"]
        if MIN_W <= span <= MAX_W:
            wins.append((i, j))
        i = j + 1
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    meta = pq.read_table(os.path.join(ROOT, "meta", "ami_eval.parquet")).to_pandas()
    meetings = sorted(meta["meeting_id"].unique())
    rng = random.Random(0)
    cands = {}
    for m in meetings:
        seg = meta[meta["meeting_id"] == m].sort_values(["begin_time", "end_time"]).reset_index(drop=True)
        seg = seg[seg["text"].str.strip() != ""]
        bl = blocks_of(seg)
        cands[m] = [(bl, i, j) for i, j in windows_of(bl, rng)]
    # spread evenly: round-robin over meetings, windows in random order within a meeting
    for m in meetings:
        rng.shuffle(cands[m])
    pick, r = [], 0
    while len(pick) < a.n and any(r < len(cands[m]) for m in meetings):
        for m in meetings:
            if r < len(cands[m]) and len(pick) < a.n:
                pick.append((m,) + cands[m][r])
        r += 1
    pick.sort(key=lambda x: (x[0], x[1][x[2]]["begin"]))
    w = ShardWriter(os.path.join(SHARDS, "eval", "ami_longform", "test"))
    stats = {"meetings": meetings, "windows": 0, "hours": 0.0, "per_meeting": {}}
    cur, audio, sr = None, None, None
    for m, bl, i, j in pick:
        if m != cur:
            audio, sr = sf.read(fetch(m), dtype="float32", always_2d=True)
            cur = m
        prev_end = bl[i - 1]["end"] if i > 0 else 0.0
        next_begin = bl[j + 1]["begin"] if j + 1 < len(bl) else len(audio) / sr
        t0 = max(prev_end, bl[i]["begin"] - PAD, 0.0)
        t1 = min(next_begin, bl[j]["end"] + PAD, len(audio) / sr)
        x = audio[int(round(t0 * sr)):int(round(t1 * sr))]
        pcm, dur = array_to_pcm16(x, sr)
        segs = sorted(s for k in range(i, j + 1) for s in bl[k]["segs"])
        text = " ".join(t for _, _, t, _ in segs)
        spk = sorted({s for *_, s in segs})
        w.add({"clip_id": f"ami_longform/test/{m}_{t0:09.2f}_{t1:09.2f}", "dataset": AMI, "config": "ihm+Mix-Headset",
               "split": "test", "revision": REV[AMI], "license": "CC-BY-4.0",
               "speaker_id": ",".join(spk), "duration_s": np.float32(dur), "ref_text": text, "audio_pcm16": pcm,
               "source_path": f"{URL.format(m=m)}#t={t0:.2f},{t1:.2f} (segments: edinburghcstr/ami ihm test)"})
        stats["windows"] += 1
        stats["hours"] += dur / 3600
        pm = stats["per_meeting"].setdefault(m, {"windows": 0, "seconds": 0.0})
        pm["windows"] += 1
        pm["seconds"] += dur
    w.close()
    stats["hours"] = round(stats["hours"], 3)
    durs = [v["seconds"] for v in stats["per_meeting"].values()]
    stats["note"] = f"{sum(durs):.0f} s over {len(durs)} meetings"
    save_stats("ami_longform", stats)
    print(stats)


if __name__ == "__main__":
    main()
