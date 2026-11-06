"""VoxPopuli en: count train clips/hours removed by dropping speakers that appear in any eval split
(en validation, en test, en_accented test). Metadata columns only.

Writes <out>/voxpopuli_drop_speakers.json (sorted speaker ids to drop from train).
Usage: python voxpopuli_filter.py --spk ../../tmp/spk/speakers_voxpopuli.json --out ../../tmp/spk
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO, REV = "facebook/voxpopuli", "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"

ap = argparse.ArgumentParser()
ap.add_argument("--spk", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
spk = json.load(open(a.spk))
PLACEHOLDER = {"None", ""}  # VoxPopuli uses "None" when the speaker is unknown; not a real speaker id
drop = (set(spk["validation"]["speakers"]) | set(spk["test"]["speakers"]) | set(spk["en_accented.test"]["speakers"])) - PLACEHOLDER
fs = HfFileSystem()
files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV) if f.startswith("en/train-"))


def rd(f):
    with fs.open(f"datasets/{REPO}@{REV}/{f}", block_size=1 << 20) as fh:
        p = pq.ParquetFile(fh)
        # audio length is not a column; use the parquet "audio" bytes size as a proxy is unreliable,
        # so only count clips here.
        return p.read(columns=["speaker_id"]).column("speaker_id").to_pylist()


ids = []
with ThreadPoolExecutor(16) as ex:
    for x in ex.map(rd, files):
        ids += x
tot = len(ids)
rm = sum(1 for s in ids if s in drop)
none = sum(1 for s in ids if s in PLACEHOLDER or s is None)
print(f"train clips={tot} removed={rm} ({100*rm/tot:.1f}%) kept={tot-rm}; drop-list speakers={len(drop)}; "
      f"train clips with empty/None speaker={none}")
print("'None' in drop list:", "None" in drop)
json.dump(sorted(drop), open(os.path.join(a.out, "voxpopuli_drop_speakers.json"), "w"))
