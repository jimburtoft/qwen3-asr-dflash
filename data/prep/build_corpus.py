# SPDX-License-Identifier: Apache-2.0
"""Corpus build (audio + reference text + metadata; no target labels).

Stages per dataset (all resumable, outputs under $Q3DF_CORPUS, default ~/q3df_corpus):
  meta      read metadata columns only (HTTP range reads)         -> meta/<name>.parquet
  select    apply filters, order candidates reproducibly (seed 0)  -> sel/<name>.parquet, stats/<name>_select.json
  fetch     download only the row groups / tars holding candidates, decode -> parts/<name>/*.parquet
  assemble  take the first N accepted candidates in rank order, write <= 2000-row zstd shards
            -> shards/<role>/<short>/<split>/shard-NNNNN.parquet, stats/<name>.json

Usage: python build_corpus.py <stage|all> <name> [--workers 8]
       python build_corpus.py evalpool            (needs the eval meta tables)
Names: see DATASETS below.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hfio  # noqa: E402
from common import (CACHE, DATA, LICENSE, PARTS, REV, ROOT, SEED, SHARDS,  # noqa: E402
                    EvalPool, ShardWriter, balanced_order, decode_to_pcm16,
                    load_flagged, nonempty, ps_license_spdx, save_stats, write_part)
from ps_clean_license_scan import clip_to_identifier  # noqa: E402

META = os.path.join(ROOT, "meta")
SEL = os.path.join(ROOT, "sel")
EVALPOOL = os.path.join(ROOT, "meta", "eval_pool_texts.json")

LS, LTTS, MLS, CV, VP, PS, AMI, FL = ("openslr/librispeech_asr", "mythicinfinity/libritts_r", "parler-tts/mls_eng",
                                      "fsicoli/common_voice_17_0", "facebook/voxpopuli",
                                      "MLCommons/peoples_speech", "edinburghcstr/ami", "google/fleurs")
MLSML = "facebook/multilingual_librispeech"

# name: (repo, short, config, role, target clips (None = all), file prefixes -> split, meta columns)
DATASETS = {
    "ls_eval": (LS, "librispeech", "all", "eval", None,
                {"all/test.clean/": "test.clean", "all/test.other/": "test.other"},
                ["id", "speaker_id", "chapter_id", "text"]),
    "ls_train": (LS, "librispeech", "all", "train", 25000,
                 {"all/train.clean.100/": "train.clean.100", "all/train.clean.360/": "train.clean.360",
                  "all/train.other.500/": "train.other.500"},
                 ["id", "speaker_id", "chapter_id", "text"]),
    "ltts_train": (LTTS, "libritts_r", "all", "train", 15000,
                   {"data/train.clean.100/": "train.clean.100", "data/train.clean.360/": "train.clean.360",
                    "data/train.other.500/": "train.other.500"},
                   ["id", "speaker_id", "chapter_id", "text_original", "text_normalized"]),
    "mls_train": (MLS, "mls", "default", "train", 20000, {"data/train-": "train"},
                  ["audio.path", "original_path", "begin_time", "end_time", "transcript", "audio_duration",
                   "speaker_id", "book_id"]),
    "cv_train": (CV, "cv17", "en", "train", 40000, {}, []),
    "cv_eval": (CV, "cv17", "en", "eval", 2000, {}, []),
    "vp_train": (VP, "voxpopuli", "en", "train", 30000, {"en/train-": "train"},
                 ["audio_id", "speaker_id", "raw_text", "normalized_text"]),
    "vp_eval": (VP, "voxpopuli", "en", "eval", None, {"en/test-": "test"},
                ["audio_id", "speaker_id", "raw_text", "normalized_text"]),
    "vp_accented_pool": (VP, "voxpopuli", "en_accented", "pool", None, {"en_accented/test-": "test"},
                         ["audio_id", "speaker_id", "raw_text", "normalized_text"]),
    "ps_train": (PS, "peoples_speech", "clean", "train", 25000, {"clean/train-": "train"},
                 ["id", "duration_ms", "text"]),
    "ami_train": (AMI, "ami", "ihm", "train", 20000, {"ihm/train-": "train"},
                  ["meeting_id", "audio_id", "text", "begin_time", "end_time", "speaker_id"]),
    "ami_eval": (AMI, "ami", "ihm", "eval", 2000, {"ihm/test-": "test"},
                 ["meeting_id", "audio_id", "text", "begin_time", "end_time", "speaker_id"]),
    "fleurs_eval": (FL, "fleurs", "en_us", "eval", None, {"parquet-data/en_us/test-": "test"},
                    ["id", "audio.path", "transcription", "raw_transcription"]),
}
# ---- multilingual additions (Qwen3-ASR); per-language clip targets ~ data/LICENSES.md section 0.2
# (clips ~ hours * 3600 / mean clip length: MLS ~15 s, VoxPopuli ~10 s, CV ~5 s)
MLS_LANGS = {"de": ("german", 4000), "fr": ("french", 4000), "es": ("spanish", 4000), "it": ("italian", 2400),
             "pt": ("portuguese", 2400), "nl": ("dutch", 2000), "pl": ("polish", 2000)}
VP_LANGS = {"de": 5000, "fr": 5000, "es": 5000, "it": 2500, "nl": 2000, "pl": 2000, "ro": 1000, "hu": 1000,
            "cs": 1000, "fi": 800}
CV_TRAIN_LANGS = {"de": 6000, "fr": 6000, "es": 6000, "it": 3000, "pt": 2000, "nl": 2000, "pl": 2000,
                  "ru": 3600, "ja": 3600, "zh-CN": 3600, "ar": 3600, "tr": 3600, "fa": 3600, "th": 3600,
                  "ro": 1500, "hu": 1500, "cs": 1500, "fi": 600, "sv-SE": 3600}
CV_EVAL_LANGS = ["de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja", "zh-CN", "ar", "tr", "fa", "th", "ro", "hu",
                 "cs", "fi", "sv-SE", "yue", "zh-HK", "ko", "vi", "hi", "id", "el", "da", "mk"]
FLEURS_LANGS = ["cmn_hans_cn", "yue_hant_hk", "de_de", "fr_fr", "es_419", "it_it", "pt_br", "nl_nl", "pl_pl",
                "ru_ru", "ja_jp", "ko_kr", "ar_eg", "tr_tr", "fa_ir", "th_th", "vi_vn", "hi_in", "id_id", "ms_my",
                "fil_ph", "sv_se", "da_dk", "fi_fi", "cs_cz", "el_gr", "ro_ro", "hu_hu", "mk_mk"]
for _l, (_d, _n) in MLS_LANGS.items():
    DATASETS[f"mlsml_{_l}_train"] = (MLSML, f"mls_{_l}", _d, "train", _n, {f"{_d}/train-": "train"},
                                     ["original_path", "begin_time", "end_time", "transcript", "audio_duration",
                                      "speaker_id", "chapter_id", "id"])
    DATASETS[f"mlsml_{_l}_eval"] = (MLSML, f"mls_{_l}", _d, "eval", 300, {f"{_d}/test-": "test"},
                                    ["original_path", "begin_time", "end_time", "transcript", "audio_duration",
                                     "speaker_id", "chapter_id", "id"])
for _l, _n in VP_LANGS.items():
    DATASETS[f"vp_{_l}_train"] = (VP, f"voxpopuli_{_l}", _l, "train", _n, {f"{_l}/train-": "train"},
                                  ["audio_id", "speaker_id", "raw_text", "normalized_text"])
    DATASETS[f"vp_{_l}_eval"] = (VP, f"voxpopuli_{_l}", _l, "eval", 300, {f"{_l}/test-": "test"},
                                 ["audio_id", "speaker_id", "raw_text", "normalized_text"])
    DATASETS[f"vp_{_l}_valpool"] = (VP, f"voxpopuli_{_l}", _l, "pool", None, {f"{_l}/validation-": "validation"},
                                    ["audio_id", "speaker_id", "raw_text", "normalized_text"])
for _l, _n in CV_TRAIN_LANGS.items():
    DATASETS[f"cv_{_l}_train"] = (CV, f"cv17_{_l}", _l, "train", _n, {}, [])
for _l in CV_EVAL_LANGS:
    DATASETS[f"cv_{_l}_eval"] = (CV, f"cv17_{_l}", _l, "eval", 300, {}, [])
for _l in FLEURS_LANGS:
    DATASETS[f"fleurs_{_l}_eval"] = (FL, f"fleurs_{_l}", _l, "eval", 300, {f"parquet-data/{_l}/test-": "test"},
                                     ["id", "audio.path", "transcription", "raw_transcription"])


def kind(name):
    """Selector family of a dataset name (multilingual names map onto the English selectors)."""
    if name.startswith("mlsml_"):
        return "mlsml_train" if name.endswith("_train") else "mlsml_eval"
    if name.startswith("vp_") and name not in ("vp_train", "vp_eval", "vp_accented_pool"):
        return {"train": "vp_train", "eval": "vp_eval", "valpool": "pool"}[name.rsplit("_", 1)[1]]
    if name.startswith("cv_") and name not in ("cv_train", "cv_eval"):
        return "cv_train" if name.endswith("_train") else "cv_eval"
    if name.startswith("fleurs_") and name != "fleurs_eval":
        return "fleurs_eval"
    return name


OVERSAMPLE = {"ls_train": 1.06, "ltts_train": 1.3, "mls_train": 1.04, "cv_train": 1.06, "vp_train": 1.06,
              "ps_train": 1.04, "ami_train": 1.04, "cv_eval": 1.05, "ami_eval": 1.03}
MLS_PREFIX = 200000
MIN_DUR = {"ami_train": 1.5, "ami_eval": 1.5}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================ meta
def _scan_one(args):
    repo, rev, path, split, cols = args
    f = hfio.CachedRangeFile(repo, rev, path)
    pf = pq.ParquetFile(f)
    md = pf.metadata
    f.prefetch(hfio.chunk_ranges(md, range(md.num_row_groups), set(cols)), max_gap=256 << 10)
    t = pf.read(columns=cols).flatten()
    t = t.rename_columns([c.replace(".", "_") for c in t.column_names])
    rg = np.concatenate([np.full(md.row_group(g).num_rows, g, dtype=np.int32) for g in range(md.num_row_groups)])
    rir = np.concatenate([np.arange(md.row_group(g).num_rows, dtype=np.int32) for g in range(md.num_row_groups)])
    t = t.append_column("file", pa.array([path] * t.num_rows)).append_column("split", pa.array([split] * t.num_rows))
    t = t.append_column("rg", pa.array(rg)).append_column("row_in_rg", pa.array(rir))
    t = t.append_column("row_in_file", pa.array(np.arange(t.num_rows, dtype=np.int64)))
    return t


def stage_meta(name, threads=16):
    os.makedirs(META, exist_ok=True)
    out = os.path.join(META, f"{name}.parquet")
    if os.path.exists(out):
        log("meta exists", out)
        return
    repo, short, config, role, n, prefixes, cols = DATASETS[name]
    rev = REV[repo]
    if repo == CV:
        return meta_cv(name)
    files = hfio.list_files(repo, rev)
    jobs = []
    for pref, split in prefixes.items():
        for f in sorted(x for x in files if x.startswith(pref) and x.endswith(".parquet")):
            jobs.append((repo, rev, f, split, cols))
    log(name, "meta scan", len(jobs), "files")
    tabs = []
    with ThreadPoolExecutor(threads) as ex:
        for i, t in enumerate(ex.map(_scan_one, jobs)):
            tabs.append(t)
            if i % 50 == 0:
                log(name, "meta", i, "/", len(jobs))
    t = pa.concat_tables(tabs, promote_options="default")
    pq.write_table(t, out + ".tmp", compression="zstd")
    os.replace(out + ".tmp", out)
    log(name, "meta rows", t.num_rows)


def cv_tsv(split, lang="en"):
    import csv
    csv.field_size_limit(1 << 30)
    d = os.path.join(CACHE, "cv17", lang)
    os.makedirs(d, exist_ok=True)
    p = hfio.download(CV, REV[CV], f"transcript/{lang}/{split}.tsv", os.path.join(d, f"{split}.tsv"))
    return pd.read_csv(p, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def meta_cv(name):
    lang = DATASETS[name][2]
    split = "train" if kind(name) == "cv_train" else "test"
    df = cv_tsv(split, lang)
    dur = cv_tsv("clip_durations", lang)
    dur = dict(zip(dur["clip"], dur["duration[ms]"].astype(float) / 1000.0))
    df["meta_duration"] = df["path"].map(dur)
    df["split"] = split
    df = df[["client_id", "path", "sentence", "sentence_id", "meta_duration", "split"]]
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), os.path.join(META, f"{name}.parquet"))
    log(name, "meta rows", len(df))


# ============================================================ eval pool (text-overlap filter)
def stage_evalpool():
    if os.path.exists(EVALPOOL):
        return
    pool = {}
    m = pq.read_table(os.path.join(META, "ls_eval.parquet")).to_pandas()
    pool["librispeech_test"] = list(m["text"])
    pool["cv17_en_test"] = list(pq.read_table(os.path.join(META, "cv_eval.parquet")).to_pandas()["sentence"])
    vp = [pq.read_table(os.path.join(META, f"{n}.parquet")).to_pandas() for n in ("vp_eval", "vp_accented_pool")]
    pool["voxpopuli_en_test"] = [t for d in vp for c in ("raw_text", "normalized_text") for t in d[c] if t]
    fl = pq.read_table(os.path.join(META, "fleurs_eval.parquet")).to_pandas()
    pool["fleurs_en_us_test"] = list(fl["transcription"]) + list(fl["raw_transcription"])
    pool["ami_ihm_test"] = list(pq.read_table(os.path.join(META, "ami_eval.parquet")).to_pandas()["text"])
    # multilingual eval / validation transcripts (every one whose meta table exists)
    cols = {"mlsml_eval": ["transcript"], "vp_eval": ["raw_text", "normalized_text"], "pool": ["raw_text", "normalized_text"],
            "cv_eval": ["sentence"], "fleurs_eval": ["transcription", "raw_transcription"]}
    for n in DATASETS:
        k = kind(n)
        if n == k or k not in cols:
            continue
        mp = os.path.join(META, f"{n}.parquet")
        if os.path.exists(mp):
            d = pq.read_table(mp).to_pandas()
            pool[n] = [t for c in cols[k] for t in d[c] if t]
    json.dump(pool, open(EVALPOOL, "w"))
    log("eval pool", {k: len(v) for k, v in pool.items()})


# ============================================================ select
def _cand(df, rank_order, name, clip_ids, texts, speakers, licenses, source_paths):
    """Build the candidate table from a filtered dataframe and an ordering (list of df positions)."""
    repo, short, config, role, n, _, _ = DATASETS[name]
    o = np.asarray(rank_order, dtype=np.int64)
    c = pd.DataFrame({
        "rank": np.arange(len(o)),
        "clip_id": np.asarray(clip_ids, dtype=object)[o],
        "split": df["split"].to_numpy()[o],
        "ref_text": np.asarray(texts, dtype=object)[o],
        "speaker_id": np.asarray(speakers, dtype=object)[o],
        "license": np.asarray(licenses, dtype=object)[o],
        "source_path": np.asarray(source_paths, dtype=object)[o],
    })
    for col in ("file", "rg", "row_in_rg", "path"):
        if col in df.columns:
            c[col] = df[col].to_numpy()[o]
    return c


def drop(df, mask, reason, stats):
    stats["dropped"][reason] = stats["dropped"].get(reason, 0) + int((~mask).sum())
    return df[mask].reset_index(drop=True)


def concentrated_order(df, key, k, seed=SEED):
    """Shuffle groups; per group take up to k rows from ONE random row group (cuts download volume),
    then append all remaining rows (random order) as reserve candidates."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, (g, f, r) in enumerate(zip(df[key], df["file"], df["rg"])):
        groups[g].append((i, (f, r)))
    keys = sorted(groups)
    rng.shuffle(keys)
    first, rest = [], []
    for g in keys:
        items = groups[g]
        byrg = defaultdict(list)
        for i, frg in items:
            byrg[frg].append(i)
        pick = rng.choice(sorted(byrg))
        rows = byrg[pick]
        rng.shuffle(rows)
        first += rows[:k]
        rest += rows[k:] + [i for frg, rr in byrg.items() if frg != pick for i in rr]
    rng.shuffle(rest)
    return first + rest


def nested_balanced(df, outer, inner, seed=SEED):
    """Round-robin over `outer` groups; inside each group the rows are pre-ordered round-robin over `inner`."""
    rng = random.Random(seed)
    inner_ordered = {}
    for g, sub in df.groupby(outer, sort=True):
        inner_ordered[g] = balanced_order(list(sub.index), key=lambda i: df.at[i, inner], seed=rng.randrange(1 << 30))
    keys = sorted(inner_ordered, key=str)
    rng.shuffle(keys)
    out, r = [], 0
    while True:
        added = False
        for g in keys:
            lst = inner_ordered[g]
            if r < len(lst):
                out.append(lst[r])
                added = True
        if not added:
            return out
        r += 1


def ls_ltts_chapter_split():
    """Partition LibriSpeech train chapters into an LS pool (~60 %) and a LibriTTS-R pool (~40 %),
    stratified per speaker, so the two draws are chapter-disjoint by construction."""
    p = os.path.join(SEL, "ls_ltts_chapter_split.json")
    if os.path.exists(p):
        return json.load(open(p))
    m = pq.read_table(os.path.join(META, "ls_train.parquet"), columns=["speaker_id", "chapter_id"]).to_pandas()
    m2 = pq.read_table(os.path.join(META, "ltts_train.parquet"), columns=["speaker_id", "chapter_id"]).to_pandas()
    spk_ch = defaultdict(set)
    for s, c in zip(m["speaker_id"].astype(str), m["chapter_id"].astype(str)):
        spk_ch[s].add(c)
    for s, c in zip(m2["speaker_id"].astype(str), m2["chapter_id"].astype(str)):
        spk_ch[s].add(c)
    rng = random.Random(SEED)
    assign = {}
    for s in sorted(spk_ch):
        chs = sorted(spk_ch[s])
        rng.shuffle(chs)
        start_ls = rng.random() < 0.6
        for j, c in enumerate(chs):
            assign[c] = "ls" if (j % 2 == 0) == start_ls else "ltts"
    os.makedirs(SEL, exist_ok=True)
    json.dump(assign, open(p, "w"))
    return assign


def stage_select(name):
    os.makedirs(SEL, exist_ok=True)
    out = os.path.join(SEL, f"{name}.parquet")
    repo, short, config, role, n, _, _ = DATASETS[name]
    rev = REV[repo]
    df = pq.read_table(os.path.join(META, f"{name}.parquet")).to_pandas()
    stats = {"name": name, "dataset": repo, "revision": rev, "config": config, "role": role,
             "pool_rows": int(len(df)), "dropped": {}}
    pool = EvalPool(EVALPOOL) if role == "train" else None
    lic = LICENSE.get(repo)

    def src(df):
        return [f"{f}#row={r}" for f, r in zip(df["file"], df["row_in_file"])]

    def overlap_filter(df, textcol, idcol, flagged_name):
        flagged = load_flagged(flagged_name) if flagged_name else set()
        m = ~df[idcol].astype(str).isin(flagged)
        df = drop(df, m, "task001_flagged_list", stats)
        m = ~df[textcol].map(pool.hit)
        return drop(df, m, "text_overlap_vs_eval", stats)

    k = kind(name)
    if name == "ls_eval":
        df["clip_id"] = [f"{short}/{s}/{i}" for s, i in zip(df["split"], df["id"])]
        order = list(range(len(df)))
        c = _cand(df, order, name, df["clip_id"], df["text"], df["speaker_id"].astype(str), [lic] * len(df), src(df))
    elif name in ("ls_train", "ltts_train"):
        assign = ls_ltts_chapter_split()
        want = "ls" if name == "ls_train" else "ltts"
        df["speaker_id"] = df["speaker_id"].astype(str)
        df["chapter_id"] = df["chapter_id"].astype(str)
        df = drop(df, df["chapter_id"].map(assign).eq(want), "chapter_reserved_for_other_corpus", stats)
        tcol = "text" if name == "ls_train" else "text_original"
        df = drop(df, df[tcol].map(nonempty), "empty_text", stats)
        df = overlap_filter(df, tcol, "id", "librispeech_train" if name == "ls_train" else "libritts_r_train")
        order = nested_balanced(df, "speaker_id", "chapter_id")
        df["clip_id"] = [f"{short}/{s}/{i}" for s, i in zip(df["split"], df["id"])]
        c = _cand(df, order, name, df["clip_id"], df[tcol], df["speaker_id"], [lic] * len(df), src(df))
        stats["chapters_in_pool"] = int(df["chapter_id"].nunique())
    elif name == "mls_train":
        df = drop(df, df["transcript"].map(nonempty), "empty_text", stats)
        df = drop(df, df["audio_duration"].between(1.0, 30.0), "meta_duration_outside_1_30s", stats)
        order = concentrated_order(df, "speaker_id", 4)
        # the text-overlap check runs on the drawn prefix only (10.8 M rows in the pool)
        df = df.iloc[order[:MLS_PREFIX]].reset_index(drop=True)
        stats["ordered_prefix_checked_for_overlap"] = int(len(df))
        df = overlap_filter(df, "transcript", "original_path", None)
        df["nid"] = df["audio_path"].str.replace(r"\.[a-z0-9]+$", "", regex=True)
        order = list(range(len(df)))
        sp = [f"{f}#row={r} ({o} {b:.2f}-{e:.2f})" for f, r, o, b, e in
              zip(df["file"], df["row_in_file"], df["original_path"], df["begin_time"], df["end_time"])]
        c = _cand(df, order, name, [f"{short}/train/{i}" for i in df["nid"]], df["transcript"], df["speaker_id"],
                  [lic] * len(df), sp)
    elif k in ("cv_train", "cv_eval"):
        split = "train" if k == "cv_train" else "test"
        lang = DATASETS[name][2]
        df = drop(df, df["sentence"].map(nonempty), "empty_text", stats)
        # project-salted hash: keeps speaker grouping without publishing Mozilla client_ids
        df["spk"] = [hashlib.sha256(("vxdf-cv17:" + x).encode()).hexdigest()[:16] for x in df["client_id"]]
        if k == "cv_train":
            df = drop(df, df["meta_duration"].between(1.0, 30.0), "meta_duration_outside_1_30s", stats)
            df = overlap_filter(df, "sentence", "path", "cv17_en_train" if lang == "en" else None)
            order = balanced_order(list(range(len(df))), key=lambda i: df.at[i, "spk"])
        else:
            order = list(range(len(df)))
            random.Random(SEED).shuffle(order)
        nid = df["path"].str.replace(".mp3", "", regex=False)
        c = _cand(df, order, name, [f"{short}/{split}/{i}" for i in nid], df["sentence"], df["spk"],
                  [lic] * len(df), [f"audio/{lang}/{split}/*.tar:{p}" for p in df["path"]])
    elif k in ("vp_train", "vp_eval"):
        txt = [r if nonempty(r) else nz for r, nz in zip(df["raw_text"], df["normalized_text"])]
        stats["ref_text_fallback_to_normalized"] = int(sum(1 for r in df["raw_text"] if not nonempty(r)))
        df["ref"] = txt
        df = drop(df, df["ref"].map(nonempty), "empty_text", stats)
        if k == "vp_train":
            if name == "vp_train":
                dl = set(json.load(open(os.path.join(DATA, "filters", "voxpopuli_drop_speakers.json")))["drop_speakers"])
            else:  # this language's test + validation speakers
                lang = DATASETS[name][2]
                dl = set()
                for nn in (f"vp_{lang}_eval", f"vp_{lang}_valpool"):
                    stage_meta(nn)
                    dl |= set(pq.read_table(os.path.join(META, f"{nn}.parquet"), columns=["speaker_id"])
                              .column("speaker_id").to_pylist())
                dl -= {"None", ""}
                stats["eval_speakers_dropped_from_train"] = len(dl)
            df = drop(df, ~df["speaker_id"].isin(["None", ""]), "speaker_unknown_None", stats)
            df = drop(df, ~df["speaker_id"].isin(dl), "speaker_in_eval_droplist", stats)
            df = overlap_filter(df, "ref", "audio_id", "voxpopuli_en_train" if name == "vp_train" else None)
            order = balanced_order(list(range(len(df))), key=lambda i: df.at[i, "speaker_id"])
        else:
            order = list(range(len(df)))
            if name != "vp_eval":
                random.Random(SEED).shuffle(order)
        spk = [None if s in ("None", "") else s for s in df["speaker_id"]]
        c = _cand(df, order, name, [f"{short}/{s}/{i}" for s, i in zip(df["split"], df["audio_id"])], df["ref"], spk,
                  [lic] * len(df), src(df))
    elif name == "ps_train":
        lt = pd.read_csv(os.path.join(DATA, "ps_clean_source_licenses.csv"), dtype=str)
        allowed = dict(zip(lt["identifier"], lt["allowed"]))
        lic_map = dict(zip(lt["identifier"], lt["license"]))
        df["identifier"] = df["id"].map(clip_to_identifier)
        df = drop(df, df["identifier"].map(allowed).eq("1"), "source_license_not_allowed", stats)
        df = drop(df, df["text"].map(nonempty), "empty_text", stats)
        df = drop(df, (df["duration_ms"] / 1000.0).between(1.0, 30.0), "meta_duration_outside_1_30s", stats)
        df = overlap_filter(df, "text", "id", None)
        order = concentrated_order(df, "identifier", 10)
        lics = [ps_license_spdx(lic_map[i]) for i in df["identifier"]]
        c = _cand(df, order, name, [f"{short}/train/{i}" for i in df["id"]], df["text"], [None] * len(df), lics, src(df))
        stats["sources_in_pool"] = int(df["identifier"].nunique())
    elif name in ("ami_train", "ami_eval"):
        df["dur"] = df["end_time"] - df["begin_time"]
        df = drop(df, df["text"].map(nonempty), "empty_text", stats)
        df = drop(df, df["dur"].between(1.5, 30.0), "meta_duration_outside_1.5_30s", stats)
        if name == "ami_train":
            dm = set(json.load(open(os.path.join(DATA, "filters", "ami_drop.json")))["drop_train_meetings"])
            df = drop(df, ~df["meeting_id"].isin(dm), "meeting_IB4005_dropped", stats)
            df = overlap_filter(df, "text", "audio_id", "ami_ihm_train")
            order = balanced_order(list(range(len(df))), key=lambda i: df.at[i, "speaker_id"])
        else:
            order = list(range(len(df)))
            random.Random(SEED).shuffle(order)
        sp = [f"{f}#row={r} ({m} {b:.2f}-{e:.2f})" for f, r, m, b, e in
              zip(df["file"], df["row_in_file"], df["meeting_id"], df["begin_time"], df["end_time"])]
        c = _cand(df, order, name, [f"{short}/{s}/{i}" for s, i in zip(df["split"], df["audio_id"])], df["text"],
                  df["speaker_id"], [lic] * len(df), sp)
    elif k in ("mlsml_train", "mlsml_eval"):
        df = drop(df, df["transcript"].map(nonempty), "empty_text", stats)
        if k == "mlsml_train":
            df = drop(df, df["audio_duration"].between(1.0, 30.0), "meta_duration_outside_1_30s", stats)
            # MLS test speakers are disjoint from train by construction (MLS paper); re-check anyway
            stage_meta(name.replace("_train", "_eval"))
            ev = set(pq.read_table(os.path.join(META, name.replace("_train", "_eval") + ".parquet"),
                                   columns=["speaker_id"]).column("speaker_id").to_pylist())
            df = drop(df, ~df["speaker_id"].isin(ev), "speaker_in_eval", stats)
            order = concentrated_order(df, "speaker_id", 4)
            df = df.iloc[order[:MLS_PREFIX]].reset_index(drop=True)
            df = overlap_filter(df, "transcript", "id", None)
            order = list(range(len(df)))
        else:
            order = list(range(len(df)))
            random.Random(SEED).shuffle(order)
        sp = [f"{f}#row={r} ({o} {b:.2f}-{e:.2f})" for f, r, o, b, e in
              zip(df["file"], df["row_in_file"], df["original_path"], df["begin_time"], df["end_time"])]
        c = _cand(df, order, name, [f"{short}/{s}/{i}" for s, i in zip(df["split"], df["id"])], df["transcript"],
                  df["speaker_id"], [lic] * len(df), sp)
    elif k == "fleurs_eval":
        nid = df["audio_path"].str.replace(".wav", "", regex=False)
        order = list(range(len(df)))
        if name != "fleurs_eval":
            random.Random(SEED).shuffle(order)
        c = _cand(df, order, name, [f"{short}/test/{i}" for i in nid], df["raw_transcription"], [None] * len(df),
                  [lic] * len(df), [f"{f}#row={r} (sentence id {i})" for f, r, i in zip(df["file"], df["row_in_file"], df["id"])])
    else:
        raise SystemExit(f"no selector for {name}")
    assert c["clip_id"].is_unique, name
    stats["eligible_after_filters"] = int(len(c))
    pq.write_table(pa.Table.from_pandas(c, preserve_index=False), out + ".tmp", compression="zstd")
    os.replace(out + ".tmp", out)
    save_stats(f"{name}_select", stats)
    log(name, "select", json.dumps(stats))


# ============================================================ fetch
def _decode_rows(name, rows, blobs):
    """rows: candidate dicts; blobs: audio bytes. Returns (accepted records, rejects)."""
    repo, short, config, role, n, _, _ = DATASETS[name]
    lo = MIN_DUR.get(name, 1.0) if role == "train" or name in MIN_DUR else 0.0
    hi = 30.0 if (role == "train" or kind(name) != name) else float("inf")
    acc, rej = [], []
    for r, b in zip(rows, blobs):
        try:
            pcm, dur = decode_to_pcm16(b)
        except Exception as e:  # noqa: BLE001
            rej.append({"clip_id": r["clip_id"], "rank": int(r["rank"]), "reason": f"decode_error:{type(e).__name__}"})
            continue
        if not (lo <= dur <= hi):
            rej.append({"clip_id": r["clip_id"], "rank": int(r["rank"]), "reason": "decoded_duration_out_of_range",
                        "duration_s": dur})
            continue
        spk = r["speaker_id"]
        spk = None if spk is None or (isinstance(spk, float) and math.isnan(spk)) else str(spk)
        acc.append({"clip_id": r["clip_id"], "dataset": repo, "config": config, "split": r["split"],
                    "revision": REV[repo], "license": r["license"], "speaker_id": spk,
                    "duration_s": np.float32(dur), "ref_text": r["ref_text"], "audio_pcm16": pcm,
                    "source_path": r["source_path"], "rank": int(r["rank"])})
    return acc, rej


def _fetch_file(args):
    name, rnd, path, rows = args
    outdir = os.path.join(PARTS, name)
    tag = hashlib.md5(path.encode()).hexdigest()[:10]
    part = os.path.join(outdir, f"r{rnd}_{tag}.parquet")
    rejp = part.replace(".parquet", ".rej.json")
    if os.path.exists(part) and os.path.exists(rejp):
        return 0, json.load(open(rejp))
    repo = DATASETS[name][0]
    f = hfio.CachedRangeFile(repo, REV[repo], path)
    pf = pq.ParquetFile(f)
    md = pf.metadata
    byrg = defaultdict(list)
    for r in rows:
        byrg[int(r["rg"])].append(r)
    acc_all, rej_all = [], []
    for g in sorted(byrg):
        f.drop_cache()
        f.prefetch(hfio.chunk_ranges(md, [g], {"audio"}), max_gap=0)
        t = pf.read_row_group(g, columns=["audio"])
        idx = [int(r["row_in_rg"]) for r in byrg[g]]
        blobs = [a["bytes"] for a in t.column("audio").take(pa.array(idx)).to_pylist()]
        a, rj = _decode_rows(name, byrg[g], blobs)
        acc_all += a
        rej_all += rj
    write_part(part, acc_all)
    json.dump(rej_all, open(rejp + f".{os.getpid()}", "w"))
    os.replace(rejp + f".{os.getpid()}", rejp)
    return len(acc_all), rej_all


def _fetch_cv_tar(args):
    name, rnd, tar_rel, rows = args
    outdir = os.path.join(PARTS, name)
    tag = hashlib.md5(tar_rel.encode()).hexdigest()[:10]
    part = os.path.join(outdir, f"r{rnd}_{tag}.parquet")
    rejp = part.replace(".parquet", ".rej.json")
    if os.path.exists(part) and os.path.exists(rejp):
        return 0, json.load(open(rejp))
    want = {r["path"]: r for r in rows}
    d = os.path.join(CACHE, "cv17", "tars", tar_rel.split("/")[1])
    os.makedirs(d, exist_ok=True)
    local = hfio.download(CV, REV[CV], tar_rel, os.path.join(d, os.path.basename(tar_rel)))
    sel_rows, blobs = [], []
    with tarfile.open(local, "r") as tf:
        for m in tf:
            b = os.path.basename(m.name)
            if m.isfile() and b in want:
                r = dict(want[b])
                r["source_path"] = f"{tar_rel}:{m.name}"
                sel_rows.append(r)
                blobs.append(tf.extractfile(m).read())
    acc, rej = _decode_rows(name, sel_rows, blobs)
    # (candidates are spread over all tars; a candidate found in no tar simply never gets accepted)
    write_part(part, acc)
    json.dump(rej, open(rejp, "w"))
    return len(acc), rej


def stage_fetch(name, limit, rnd, workers):
    os.makedirs(os.path.join(PARTS, name), exist_ok=True)
    c = pq.read_table(os.path.join(SEL, f"{name}.parquet")).to_pandas()
    lo = 0 if rnd == 0 else json.load(open(os.path.join(PARTS, name, f"round{rnd - 1}.json")))["limit"]
    sub = c[(c["rank"] >= lo) & (c["rank"] < limit)]
    log(name, f"fetch round {rnd}: ranks [{lo},{limit}) -> {len(sub)} candidates")
    recs = sub.to_dict("records")
    if DATASETS[name][0] == CV:
        lang = DATASETS[name][2]
        n_sh = json.loads(open(hfio.download(CV, REV[CV], "n_shards.json",
                                             os.path.join(CACHE, "cv17", "n_shards.json"))).read())[lang]
        split = "train" if kind(name) == "cv_train" else "test"
        jobs = [(name, rnd, f"audio/{lang}/{split}/{lang}_{split}_{i}.tar", recs) for i in range(n_sh[split])]
        fn = _fetch_cv_tar
    else:
        byf = defaultdict(list)
        for r in recs:
            byf[r["file"]].append(r)
        jobs = [(name, rnd, f, rows) for f, rows in sorted(byf.items())]
        fn = _fetch_file
    rej = []
    with Pool(workers) as p:
        for i, (_, rj) in enumerate(p.imap_unordered(fn, jobs)):
            rej += rj
            if i % 25 == 0:
                log(name, f"fetch {i + 1}/{len(jobs)}")
    json.dump({"limit": int(limit), "rejects": Counter(r["reason"] for r in rej)},
              open(os.path.join(PARTS, name, f"round{rnd}.json"), "w"))
    return rej


# ============================================================ assemble
def accepted_ranks(name):
    out = []
    for p in glob.glob(os.path.join(PARTS, name, "r*.parquet")):
        t = pq.read_table(p, columns=["clip_id", "rank"])
        out += t.column("rank").to_pylist()
    return sorted(set(out))


def stage_assemble(name):
    repo, short, config, role, n, _, _ = DATASETS[name]
    parts = sorted(glob.glob(os.path.join(PARTS, name, "r*.parquet")))
    idx = []
    for p in parts:
        t = pq.read_table(p, columns=["rank"])
        idx += [(rk, p, i) for i, rk in enumerate(t.column("rank").to_pylist())]
    idx.sort()
    seen, keep = set(), []
    for rk, p, i in idx:
        if rk in seen:
            continue
        seen.add(rk)
        keep.append((rk, p, i))
    if n is not None:
        keep = keep[:n]
    keep_by_part = defaultdict(list)
    for rk, p, i in keep:
        keep_by_part[p].append(i)
    rows = []
    for p, ii in keep_by_part.items():
        want = set(ii)
        off = 0
        for b in pq.ParquetFile(p).iter_batches(batch_size=200):
            loc = [i - off for i in range(off, off + b.num_rows) if i in want]
            if loc:
                rows += b.take(pa.array(loc)).to_pylist()
            off += b.num_rows
    rows.sort(key=lambda r: r["rank"])
    base = os.path.join(SHARDS, role, short)
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    stats = {"name": name, "dataset": repo, "config": config, "role": role, "target": n, "splits": {}}
    rejects = Counter()
    for p in glob.glob(os.path.join(PARTS, name, "r*.rej.json")):
        for r in json.load(open(p)):
            if r["rank"] < (keep[-1][0] if keep else 0):
                rejects[r["reason"]] += 1
    for split, rr in sorted(by_split.items()):
        d = os.path.join(base, split)
        if os.path.exists(d):
            shutil.rmtree(d)
        w = ShardWriter(d)
        for r in rr:
            r.pop("rank", None)
            w.add(r)
        w.close()
        stats["splits"][split] = {"clips": len(rr), "hours": round(sum(float(r["duration_s"]) for r in rr) / 3600, 3),
                                  "shards": w.n_shards,
                                  "speakers": len({r["speaker_id"] for r in rr if r["speaker_id"]}),
                                  "max_duration_s": round(max(float(r["duration_s"]) for r in rr), 2),
                                  "min_duration_s": round(min(float(r["duration_s"]) for r in rr), 2)}
    stats["decode_stage_rejects_within_drawn_ranks"] = dict(rejects)
    stats["total_clips"] = sum(s["clips"] for s in stats["splits"].values())
    stats["total_hours"] = round(sum(s["hours"] for s in stats["splits"].values()), 3)
    save_stats(name, stats)
    log(name, "assembled", json.dumps(stats))
    return stats


def stage_all(name, workers):
    stage_meta(name)
    if DATASETS[name][3] == "pool":
        return
    if DATASETS[name][3] == "train":
        stage_evalpool()
    stage_select(name)
    n = DATASETS[name][4]
    total = len(pq.read_table(os.path.join(SEL, f"{name}.parquet"), columns=["rank"]))
    limit = total if n is None else min(total, math.ceil(n * OVERSAMPLE.get(name, 1.05)))
    rnd = 0
    while True:
        rd = os.path.join(PARTS, name, f"round{rnd}.json")
        if os.path.exists(rd):
            limit = json.load(open(rd))["limit"]
        else:
            stage_fetch(name, limit, rnd, workers)
        got = len(accepted_ranks(name))
        log(name, f"after round {rnd}: accepted {got} (target {n}, candidates {total})")
        if n is None or got >= n or limit >= total:
            break
        rnd += 1
        limit = min(total, limit + math.ceil((n - got) * 1.3) + 50)
    stage_assemble(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "evalpool":
        return stage_evalpool()
    {"meta": lambda: stage_meta(a.name), "select": lambda: stage_select(a.name),
     "assemble": lambda: stage_assemble(a.name), "all": lambda: stage_all(a.name, a.workers)}[a.stage]()


if __name__ == "__main__":
    main()
