"""Transcript-level leakage check: flag train clips whose normalized transcript matches an eval
transcript (exact, >= 8 words) or shares any 12-word n-gram with an eval transcript.

Text columns only (no audio). Eval: LibriSpeech test.clean/test.other, Common Voice 17 en test,
VoxPopuli en test (+ en_accented test), FLEURS en_us test, AMI ihm test.
Train: LibriSpeech train, LibriTTS-R train, MLS en train (sampled shards), Common Voice 17 en
train, VoxPopuli en train, People's Speech clean train (sampled shards), AMI ihm train.

Writes <out>/text_overlap_summary.json and <out>/text_overlap_flagged_<tag>.txt (flagged clip ids).
Usage: python text_overlap.py --cv ../../tmp/cv --out ../../tmp/text [--mls-shards 60 --ps-shards 60]
"""
import argparse
import csv
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REVS = {
    "openslr/librispeech_asr": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "mythicinfinity/libritts_r": "0d1718db351207d5a493135a62d5ed8cc4733830",
    "parler-tts/mls_eng": "faf6604dcf0bb9ec0d9c280a140ad74da9c931b0",
    "facebook/voxpopuli": "42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
    "MLCommons/peoples_speech": "f10597c5d3d3a63f8b6827701297c3afdf178272",
    "edinburghcstr/ami": "46f28f2503e2ec48f8867a84eef356c70476beab",
    "google/fleurs": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
}
N = 12
fs = HfFileSystem()
_files = {}


def files(repo, prefix):
    if repo not in _files:
        _files[repo] = HfApi().list_repo_files(repo, repo_type="dataset", revision=REVS[repo])
    return sorted(f for f in _files[repo] if f.startswith(prefix) and f.endswith(".parquet"))


def read(repo, fl, cols, workers=16):
    def rd(f):
        with fs.open(f"datasets/{repo}@{REVS[repo]}/{f}", block_size=1 << 20) as fh:
            return pq.ParquetFile(fh).read(columns=cols).to_pydict()
    out = {c: [] for c in cols}
    with ThreadPoolExecutor(workers) as ex:
        for d in ex.map(rd, fl):
            for c in cols:
                out[c].extend(d[c])
    return out


def norm(t):
    t = (t or "").lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return " ".join(t.split())


def ngrams(words):
    return {" ".join(words[i:i + N]) for i in range(len(words) - N + 1)}


def cv_tsv(d, split):
    csv.field_size_limit(1 << 30)
    rows = csv.DictReader(open(f"{d}/{split}.tsv", newline="", encoding="utf-8"), delimiter="\t",
                          quoting=csv.QUOTE_NONE)
    ids, txt = [], []
    for r in rows:
        ids.append(r["path"])
        txt.append(r["sentence"])
    return ids, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mls-shards", type=int, default=60)
    ap.add_argument("--ps-shards", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(0)

    ev = {}
    d = read("openslr/librispeech_asr", files("openslr/librispeech_asr", "all/test."), ["text"])
    ev["librispeech_test"] = d["text"]
    ev["cv17_en_test"] = cv_tsv(a.cv, "test")[1]
    d = read("facebook/voxpopuli", files("facebook/voxpopuli", "en/test-") + files("facebook/voxpopuli", "en_accented/test-"),
             ["normalized_text"])
    ev["voxpopuli_en_test"] = d["normalized_text"]
    d = read("google/fleurs", files("google/fleurs", "parquet-data/en_us/test-"), ["transcription"])
    ev["fleurs_en_us_test"] = d["transcription"]
    d = read("edinburghcstr/ami", files("edinburghcstr/ami", "ihm/test-"), ["text"])
    ev["ami_ihm_test"] = d["text"]

    exact, grams = {}, {}
    for k, v in ev.items():
        for t in v:
            w = norm(t).split()
            if len(w) >= 8:
                exact.setdefault(" ".join(w), set()).add(k)
            for g in ngrams(w):
                grams.setdefault(g, set()).add(k)
    print("eval texts:", {k: len(v) for k, v in ev.items()}, "exact keys", len(exact), "12-grams", len(grams), flush=True)

    tr = {}
    d = read("openslr/librispeech_asr", files("openslr/librispeech_asr", "all/train."), ["id", "text"])
    tr["librispeech_train"] = (d["id"], d["text"])
    d = read("mythicinfinity/libritts_r", files("mythicinfinity/libritts_r", "data/train."), ["id", "text_normalized"])
    tr["libritts_r_train"] = (d["id"], d["text_normalized"])
    fl = random.sample(files("parler-tts/mls_eng", "data/train-"), a.mls_shards)
    d = read("parler-tts/mls_eng", fl, ["original_path", "transcript"])
    tr[f"mls_train_sample{a.mls_shards}sh"] = (d["original_path"], d["transcript"])
    tr["cv17_en_train"] = cv_tsv(a.cv, "train")
    d = read("facebook/voxpopuli", files("facebook/voxpopuli", "en/train-"), ["audio_id", "normalized_text"])
    tr["voxpopuli_en_train"] = (d["audio_id"], d["normalized_text"])
    fl = random.sample(files("MLCommons/peoples_speech", "clean/train-"), a.ps_shards)
    d = read("MLCommons/peoples_speech", fl, ["id", "text"])
    tr[f"ps_clean_train_sample{a.ps_shards}sh"] = (d["id"], d["text"])
    d = read("edinburghcstr/ami", files("edinburghcstr/ami", "ihm/train-"), ["audio_id", "text"])
    tr["ami_ihm_train"] = (d["audio_id"], d["text"])

    summary = {}
    for k, (ids, txt) in tr.items():
        flagged = []
        by_eval = {}
        for i, t in zip(ids, txt):
            w = norm(t).split()
            hit = set()
            if len(w) >= 8:
                hit |= exact.get(" ".join(w), set())
            for g in ngrams(w):
                if g in grams:
                    hit |= grams[g]
            if hit:
                flagged.append(i)
                for e in hit:
                    by_eval[e] = by_eval.get(e, 0) + 1
        summary[k] = {"rows": len(ids), "flagged": len(flagged),
                      "flagged_pct": round(100 * len(flagged) / max(1, len(ids)), 3), "by_eval_set": by_eval}
        with open(os.path.join(a.out, f"text_overlap_flagged_{k}.txt"), "w") as fh:
            fh.write("\n".join(map(str, flagged)))
        print(k, summary[k], flush=True)
    json.dump(summary, open(os.path.join(a.out, "text_overlap_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
