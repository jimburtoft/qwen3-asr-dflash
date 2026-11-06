"""Collect speaker / meeting / book ids per split for the approved corpora (metadata columns only,
no audio) and compute train-vs-eval overlaps.

Writes <out>/speakers_<tag>.json (split -> sorted list of ids) and <out>/overlap_report.json.

Usage: python speaker_overlap.py --out ../../tmp/spk [--only librispeech,libritts_r,mls,voxpopuli,ami]
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REVS = {
    "openslr/librispeech_asr": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "mythicinfinity/libritts_r": "0d1718db351207d5a493135a62d5ed8cc4733830",
    "parler-tts/mls_eng": "faf6604dcf0bb9ec0d9c280a140ad74da9c931b0",
    "facebook/voxpopuli": "42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
    "edinburghcstr/ami": "46f28f2503e2ec48f8867a84eef356c70476beab",
}

# tag -> (repo, {split_name: path_prefix}, columns)
SPECS = {
    "librispeech": ("openslr/librispeech_asr", {
        "train.clean.100": "all/train.clean.100/", "train.clean.360": "all/train.clean.360/",
        "train.other.500": "all/train.other.500/", "validation.clean": "all/validation.clean/",
        "validation.other": "all/validation.other/", "test.clean": "all/test.clean/",
        "test.other": "all/test.other/"}, ["speaker_id", "chapter_id"]),
    "libritts_r": ("mythicinfinity/libritts_r", {
        "train.clean.100": "data/train.clean.100/", "train.clean.360": "data/train.clean.360/",
        "train.other.500": "data/train.other.500/", "dev.clean": "data/dev.clean/",
        "dev.other": "data/dev.other/", "test.clean": "data/test.clean/",
        "test.other": "data/test.other/"}, ["speaker_id", "chapter_id"]),
    "mls": ("parler-tts/mls_eng", {
        "train": "data/train-", "dev": "data/dev-", "test": "data/test-"},
        ["speaker_id", "book_id", "audio_duration"]),
    "voxpopuli": ("facebook/voxpopuli", {
        "train": "en/train-", "validation": "en/validation-", "test": "en/test-",
        "en_accented.test": "en_accented/test-"}, ["speaker_id", "audio_id"]),
    "ami": ("edinburghcstr/ami", {
        "train": "ihm/train-", "validation": "ihm/validation-", "test": "ihm/test-"},
        ["speaker_id", "meeting_id", "begin_time", "end_time"]),
}


def read_cols(fs, repo, rev, path, cols):
    with fs.open(f"datasets/{repo}@{rev}/{path}", block_size=1 << 20) as fh:
        return pq.ParquetFile(fh).read(columns=cols).to_pydict()


def collect(tag, out, workers):
    repo, splits, cols = SPECS[tag]
    rev = REVS[repo]
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
    fs = HfFileSystem()
    res = {}
    for split, pref in splits.items():
        fl = sorted(f for f in files if f.startswith(pref) and f.endswith(".parquet"))
        acc = {c: [] for c in cols}
        with ThreadPoolExecutor(workers) as ex:
            for d in ex.map(lambda p: read_cols(fs, repo, rev, p, cols), fl):
                for c in cols:
                    acc[c].extend(d[c])
        entry = {"files": len(fl), "rows": len(acc[cols[0]]),
                 "speakers": sorted({str(x) for x in acc["speaker_id"]})}
        if "chapter_id" in acc:
            entry["chapters"] = sorted({str(x) for x in acc["chapter_id"]})
        if "book_id" in acc:
            entry["books"] = sorted({str(x) for x in acc["book_id"]})
        if "meeting_id" in acc:
            entry["meetings"] = sorted({str(x) for x in acc["meeting_id"]})
        if "audio_duration" in acc:
            entry["hours"] = round(sum(acc["audio_duration"]) / 3600, 2)
            entry["clips_le_30s"] = sum(1 for x in acc["audio_duration"] if x <= 30)
        if "begin_time" in acc:
            d = [e - b for b, e in zip(acc["begin_time"], acc["end_time"])]
            entry["hours"] = round(sum(d) / 3600, 2)
        res[split] = entry
        print(tag, split, entry["files"], entry["rows"], len(entry["speakers"]), flush=True)
    with open(os.path.join(out, f"speakers_{tag}.json"), "w") as fh:
        json.dump(res, fh)
    return res


def inter(a, b):
    return sorted(set(a) & set(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=",".join(SPECS))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {}
    for tag in a.only.split(","):
        p = os.path.join(a.out, f"speakers_{tag}.json")
        data[tag] = json.load(open(p)) if os.path.exists(p) else collect(tag, a.out, a.workers)

    rep = {}
    if "librispeech" in data:
        ls = data["librispeech"]
        ls_train = set().union(*(ls[s]["speakers"] for s in ls if s.startswith("train")))
        ls_eval = {s: ls[s]["speakers"] for s in ("test.clean", "test.other", "validation.clean", "validation.other")}
        rep["librispeech_train_vs_eval"] = {s: inter(ls_train, v) for s, v in ls_eval.items()}
        if "libritts_r" in data:
            lt = data["libritts_r"]
            lt_train = set().union(*(lt[s]["speakers"] for s in lt if s.startswith("train")))
            lt_train_ch = set().union(*(lt[s]["chapters"] for s in lt if s.startswith("train")))
            rep["libritts_r_train_vs_librispeech_eval_speakers"] = {s: inter(lt_train, v) for s, v in ls_eval.items()}
            rep["libritts_r_train_vs_librispeech_eval_chapters"] = {
                s: inter(lt_train_ch, ls[s]["chapters"]) for s in ls_eval}
            rep["libritts_r_train_vs_librispeech_train_speakers_count"] = len(lt_train & ls_train)
            rep["libritts_r_train_speakers_count"] = len(lt_train)
        if "mls" in data:
            m = data["mls"]
            m_train = set(m["train"]["speakers"])
            rep["mls_train_vs_librispeech_eval_speakers"] = {s: inter(m_train, v) for s, v in ls_eval.items()}
            rep["mls_train_vs_librispeech_train_speakers_count"] = len(m_train & ls_train)
            rep["mls_train_speakers_count"] = len(m_train)
            rep["mls_train_vs_mls_eval"] = {s: inter(m_train, m[s]["speakers"]) for s in ("dev", "test")}
    if "voxpopuli" in data:
        v = data["voxpopuli"]
        rep["voxpopuli_train_vs_eval"] = {s: inter(v["train"]["speakers"], v[s]["speakers"])
                                          for s in ("validation", "test", "en_accented.test")}
    if "ami" in data:
        am = data["ami"]
        rep["ami_train_vs_eval_speakers"] = {s: inter(am["train"]["speakers"], am[s]["speakers"])
                                             for s in ("validation", "test")}
        rep["ami_train_vs_eval_meetings"] = {s: inter(am["train"]["meetings"], am[s]["meetings"])
                                             for s in ("validation", "test")}
        rep["ami_meetings"] = {s: am[s]["meetings"] for s in am}
    summ = {k: ({kk: len(vv) for kk, vv in v.items()} if isinstance(v, dict) and k != "ami_meetings" else v)
            for k, v in rep.items()}
    print(json.dumps(summ, indent=1))
    with open(os.path.join(a.out, "overlap_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
