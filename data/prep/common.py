# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the corpus build: schema, audio decoding, shard writing, sampling, filters."""
import io
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)  # .../data
sys.path.insert(0, os.path.join(DATA, "scripts"))

ROOT = os.environ.get("Q3DF_CORPUS", os.environ.get("VXDF_CORPUS", os.path.expanduser("~/q3df_corpus")))
SHARDS = os.path.join(ROOT, "shards")
CACHE = os.path.join(ROOT, "cache")
PARTS = os.path.join(ROOT, "parts")
STATS = os.path.join(ROOT, "stats")
SR = 16000
ROWS_PER_SHARD = 2000
SEED = 0

PLAN = json.load(open(os.path.join(DATA, "manifest_plan.json")))
REV = PLAN["revisions"]

SCHEMA = pa.schema([
    ("clip_id", pa.string()),
    ("dataset", pa.string()),
    ("config", pa.string()),
    ("split", pa.string()),
    ("revision", pa.string()),
    ("license", pa.string()),
    ("speaker_id", pa.string()),
    ("duration_s", pa.float32()),
    ("ref_text", pa.string()),
    ("audio_pcm16", pa.binary()),
    ("source_path", pa.string()),
])
PART_SCHEMA = SCHEMA.append(pa.field("rank", pa.int64()))

LICENSE = {
    "openslr/librispeech_asr": "CC-BY-4.0",
    "mythicinfinity/libritts_r": "CC-BY-4.0",
    "parler-tts/mls_eng": "CC-BY-4.0",
    "facebook/multilingual_librispeech": "CC-BY-4.0",
    "fsicoli/common_voice_17_0": "CC0-1.0",
    "facebook/voxpopuli": "CC0-1.0",
    "edinburghcstr/ami": "CC-BY-4.0",
    "google/fleurs": "CC-BY-4.0",
}


def ps_license_spdx(lic):
    """Map the per-source license label from data/ps_clean_source_licenses.csv to an SPDX-like id."""
    if lic.startswith("cc-by-"):
        parts = lic[len("cc-by-"):].split("-")
        return "CC-BY-" + "-".join(p.upper() for p in parts)
    return {
        "cc-cc0-3.0": "CC0-1.0",  # archive.org licenseurl .../cc0/3.0 (CC0 has only one version: 1.0)
        "cc-cc0-1.0": "CC0-1.0",
        "cc-publicdomain": "PDM-1.0",  # Creative Commons Public Domain Mark
        "public-domain": "LicenseRef-Public-Domain",
        "other:https://www.usa.gov/government-works": "LicenseRef-US-Government-Work",
    }[lic]


# ---------------------------------------------------------------- audio
def decode_to_pcm16(data: bytes):
    """Decode audio bytes -> (pcm16 little-endian bytes at 16 kHz mono, duration_s)."""
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return array_to_pcm16(x, sr)


def array_to_pcm16(x, sr):
    if x.ndim == 2:
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="HQ")
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(x * 32767.0).astype("<i2")
    return pcm.tobytes(), float(len(pcm)) / SR


# ---------------------------------------------------------------- sampling
def balanced_order(items, key, seed=SEED):
    """Round-robin over groups (shuffled), items shuffled within each group.

    Returns the items in priority order: every group contributes one item per round, so taking a
    prefix spreads the draw as evenly as possible across groups (speakers / books / sources).
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    out, r = [], 0
    while True:
        added = False
        for k in keys:
            g = groups[k]
            if r < len(g):
                out.append(g[r])
                added = True
        if not added:
            return out
        r += 1


# ---------------------------------------------------------------- text-overlap filter
from text_overlap import ngrams, norm  # noqa: E402  (same normalisation as data/scripts/text_overlap.py)


class EvalPool:
    """Exact (>= 8 words) and 12-gram index of every eval-set transcript (data/scripts/text_overlap.py rule)."""

    def __init__(self, path):
        d = json.load(open(path))
        self.exact, self.grams = set(), set()
        for texts in d.values():
            for t in texts:
                w = norm(t).split()
                if len(w) >= 8:
                    self.exact.add(" ".join(w))
                self.grams |= ngrams(w)

    def hit(self, text):
        w = norm(text).split()
        if len(w) >= 8 and " ".join(w) in self.exact:
            return True
        return any(g in self.grams for g in ngrams(w))


def load_flagged(name):
    p = os.path.join(DATA, "filters", f"text_overlap_flagged_{name}.txt")
    return set(open(p).read().split()) if os.path.exists(p) else set()


# ---------------------------------------------------------------- writing
class ShardWriter:
    """Writes rows into <outdir>/shard-%05d.parquet with <= ROWS_PER_SHARD rows each (zstd)."""

    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.buf = {f.name: [] for f in SCHEMA}
        self.n_shards = 0
        self.rows = 0

    def add(self, row):
        for k in self.buf:
            self.buf[k].append(row[k])
        if len(self.buf["clip_id"]) >= ROWS_PER_SHARD:
            self.flush()

    def flush(self):
        if not self.buf["clip_id"]:
            return
        t = pa.table(self.buf, schema=SCHEMA)
        p = os.path.join(self.outdir, f"shard-{self.n_shards:05d}.parquet")
        pq.write_table(t, p + ".tmp", compression="zstd", row_group_size=250)
        os.replace(p + ".tmp", p)
        self.n_shards += 1
        self.rows += t.num_rows
        self.buf = {f.name: [] for f in SCHEMA}

    def close(self):
        self.flush()
        return self.rows


def write_part(path, rows):
    cols = {f.name: [r[f.name] for r in rows] for f in PART_SCHEMA}
    tmp = f"{path}.{os.getpid()}.tmp"
    pq.write_table(pa.table(cols, schema=PART_SCHEMA), tmp, compression="zstd", row_group_size=200)
    os.replace(tmp, path)


def save_stats(name, d):
    os.makedirs(STATS, exist_ok=True)
    json.dump(d, open(os.path.join(STATS, f"{name}.json"), "w"), indent=1, sort_keys=True)


def nonempty(t):
    return t is not None and bool(re.sub(r"\s+", "", t))
