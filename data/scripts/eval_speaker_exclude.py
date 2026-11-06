# SPDX-License-Identifier: Apache-2.0
"""Train clips whose speaker (per speaker-id family) appears in ANY eval set -> exclusion list.

Multilingual sources share speakers across languages (Common Voice contributors record several
languages; MEPs speak several languages in VoxPopuli; LibriVox readers read several languages), so the
per-language eval-speaker filters of the corpus build are not enough. Writes
data/filters/train_exclude_eval_speakers.txt (one clip_id per line); the trainer drops these clips.

Usage: python eval_speaker_exclude.py [--shards ~/q3df_corpus/shards]
"""
import argparse
import collections
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prep"))
from manifest_and_sanity import family  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default=os.path.expanduser("~/q3df_corpus/shards"))
a = ap.parse_args()
tr = pq.read_table(os.path.join(a.shards, "train_manifest.parquet"), columns=["clip_id", "speaker_id"]).to_pylist()
ev = pq.read_table(os.path.join(a.shards, "eval_manifest.parquet"), columns=["clip_id", "speaker_id"]).to_pylist()


def keys(r):
    fam = family(r["clip_id"].split("/")[0])
    return {(fam, s) for s in (r["speaker_id"] or "").split(",") if fam and s}


evk = set().union(*(keys(r) for r in ev))
drop = [r["clip_id"] for r in tr if keys(r) & evk]
by = collections.Counter(c.split("/")[0] for c in drop)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "filters", "train_exclude_eval_speakers.txt")
open(out, "w").write("\n".join(sorted(drop)) + "\n")
print(json.dumps({"train_clips": len(tr), "excluded": len(drop), "by_corpus": dict(by.most_common())}))
