# Data licenses, revisions and plan (Qwen3-ASR-1.7B DFlash draft)

Sections 1-9 below are the English-data audit of the public `jimburtoft/voxtral-dflash` release
(audit date 2026-10-08), reused as is: **every English dataset revision was re-checked on
2026-11-04 and is unchanged** (same `sha`, same card license; `data/scripts/` re-run via the HF
API). The target model is Qwen3-ASR-1.7B (sections 0.3 and 4).
Section 0 adds the multilingual sources and the Qwen3-ASR language mix.

## 0. Qwen3-ASR additions (audit 2026-11-04)

### 0.1 Multilingual sources

Only datasets from a fixed allow-list of permissively licensed (CC0 / CC BY) corpora are used. Qwen3-ASR's 30 `support_languages`
were matched against what those datasets carry under a clean license.

| Dataset (HF id) | Revision (`sha`) | Gated | Card license | Languages used (Qwen3-ASR-supported) | Role |
|---|---|---|---|---|---|
| `facebook/multilingual_librispeech` | `2e83e61823b4c47dcbcb1980bb88601274127609` | no | cc-by-4.0 (openslr.org/94: CC BY 4.0) | de, fr, es, it, pt, nl, pl (train rows 469,942 / 258,213 / 220,701 / 59,623 / 37,533 / 374,287 / 25,043) | train |
| `facebook/voxpopuli` | `42f01879c780b4a2e90ec0b4f616c2ece526e4f1` (unchanged) | no | cc0-1.0 + European Parliament notice (note C) | de, fr, es, it, nl, pl, ro, hu, cs, fi (train rows 108,473 / 73,561 / 50,922 / 22,576 / 20,968 / 34,665 / 24,259 / 18,120 / 18,902 / 7,878); test = eval | train + eval |
| `fsicoli/common_voice_17_0` | `8262c16bf297c87a9cd88c51997c4758ed7a8ba2` (unchanged) | no | cc0-1.0 (note B) | train: de, fr, es, it, pt, nl, pl, ru, ja, zh-CN, ar, tr, fa, th, hu, cs, ro, sv-SE, ... (small sets as available); test = eval for every supported language present, incl. yue, zh-HK, ko, vi, hi, id, el, da, fi, mk | train + eval |
| `google/fleurs` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (unchanged) | no | cc-by-4.0 (note D: eval only) | test split of every supported language (cmn_hans_cn, yue_hant_hk, en_us, de_de, fr_fr, es_419, ja_jp, ko_kr, ...) | eval only |

The multilingual `facebook/voxpopuli` `multilang` config is not used (it mixes languages and
overlaps the per-language configs). VoxPopuli's speaker-overlap filter (section 6) is re-run per
language against that language's test speakers.

### 0.2 Language mix (decision)

Target: ~600 h of 1-30 s clips. **English ~60 %** (~360 h, the voxtral-dflash English recipe:
LibriSpeech, LibriTTS-R, MLS-en, CV-en, VoxPopuli-en, People's Speech clean, AMI ihm).
**Other languages ~40 %** (~240 h), weighted toward languages with large clean sets:

| Language | Sources | ~hours |
|---|---|---|
| German | MLS-de, VoxPopuli-de, CV-de | 40 |
| French | MLS-fr, VoxPopuli-fr, CV-fr | 40 |
| Spanish | MLS-es, VoxPopuli-es, CV-es | 40 |
| Italian | MLS-it, VoxPopuli-it, CV-it | 20 |
| Portuguese | MLS-pt, CV-pt | 15 |
| Dutch | MLS-nl, VoxPopuli-nl, CV-nl | 15 |
| Polish | MLS-pl, VoxPopuli-pl, CV-pl | 15 |
| Romanian, Hungarian, Czech, Finnish | VoxPopuli, CV | 5 each (20) |
| Russian, Japanese, Chinese (Mandarin), Arabic, Turkish, Persian, Thai, Swedish | CV only (small) | up to 5 each, all available clean clips up to that cap |

**Chinese / Cantonese:** no approved source has a large clean Mandarin or Cantonese **train**
set (WenetSpeech, AISHELL-family and similar corpora are non-commercial or unclear and are
excluded). Mandarin: Common Voice zh-CN train (CC0, small) only, capped as above. Cantonese:
**eval only** (Common Voice yue / zh-HK test, FLEURS yue_hant_hk). Korean, Vietnamese, Hindi,
Indonesian, Malay, Filipino, Greek, Danish, Macedonian: eval only (CV test / FLEURS), because their
clean train data in the approved sets is too small to matter.

The labels are the target's own transcripts, so mis-identified language in the source metadata
does not matter for training; the language each clip was drawn as is kept in the manifest.

### 0.3 Target model

| | |
|---|---|
| Model | `Qwen/Qwen3-ASR-1.7B` rev `7278e1e70fe206f11671096ffdd38061171dd6e5` (weights also as `Qwen/Qwen3-ASR-1.7B-hf` rev `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`, identical tensors) |
| License | Apache-2.0 (model card) |
| Use | teacher: greedy transcripts (labels) and hidden states (features) |
| Derivative | the draft is released under Apache-2.0 with attribution to the target and to the data sources |


### 0.4 Built corpus (2026-11-05)

Train: **281,801 clips, 700.9 h** (1-30 s): English 464.0 h (66 %), other languages
236.9 h (34 %). Eval: 34,248 clips, 81.2 h (80 sets; 34,220 clips are <= 30 s and are
labelled). Every row carries dataset, config, split, revision, license and source path
(`shards/{train,eval}_manifest.parquet`).

| name | dataset | config | clips | hours |
|---|---|---|---|---|
| ami_train | edinburghcstr/ami | ihm | 20000 | 24.202 |
| cv_ar_train | fsicoli/common_voice_17_0 | ar | 3600 | 4.197 |
| cv_cs_train | fsicoli/common_voice_17_0 | cs | 1500 | 1.944 |
| cv_de_train | fsicoli/common_voice_17_0 | de | 6000 | 9.366 |
| cv_es_train | fsicoli/common_voice_17_0 | es | 6000 | 9.517 |
| cv_fa_train | fsicoli/common_voice_17_0 | fa | 3600 | 3.974 |
| cv_fi_train | fsicoli/common_voice_17_0 | fi | 600 | 0.745 |
| cv_fr_train | fsicoli/common_voice_17_0 | fr | 6000 | 8.845 |
| cv_hu_train | fsicoli/common_voice_17_0 | hu | 1500 | 2.136 |
| cv_it_train | fsicoli/common_voice_17_0 | it | 3000 | 4.707 |
| cv_ja_train | fsicoli/common_voice_17_0 | ja | 3600 | 4.604 |
| cv_nl_train | fsicoli/common_voice_17_0 | nl | 2000 | 2.653 |
| cv_pl_train | fsicoli/common_voice_17_0 | pl | 2000 | 2.635 |
| cv_pt_train | fsicoli/common_voice_17_0 | pt | 2000 | 2.242 |
| cv_ro_train | fsicoli/common_voice_17_0 | ro | 1500 | 1.653 |
| cv_ru_train | fsicoli/common_voice_17_0 | ru | 3600 | 5.233 |
| cv_sv-SE_train | fsicoli/common_voice_17_0 | sv-SE | 3600 | 3.722 |
| cv_th_train | fsicoli/common_voice_17_0 | th | 3600 | 4.178 |
| cv_tr_train | fsicoli/common_voice_17_0 | tr | 3600 | 3.654 |
| cv_train | fsicoli/common_voice_17_0 | en | 40000 | 65.44 |
| cv_zh-CN_train | fsicoli/common_voice_17_0 | zh-CN | 3600 | 5.195 |
| ls_train | openslr/librispeech_asr | all | 25000 | 85.899 |
| ltts_train | mythicinfinity/libritts_r | all | 15000 | 27.167 |
| mls_train | parler-tts/mls_eng | default | 20000 | 82.691 |
| mlsml_de_train | facebook/multilingual_librispeech | german | 4000 | 16.694 |
| mlsml_es_train | facebook/multilingual_librispeech | spanish | 4000 | 16.582 |
| mlsml_fr_train | facebook/multilingual_librispeech | french | 4000 | 16.653 |
| mlsml_it_train | facebook/multilingual_librispeech | italian | 2400 | 9.928 |
| mlsml_nl_train | facebook/multilingual_librispeech | dutch | 2000 | 8.28 |
| mlsml_pl_train | facebook/multilingual_librispeech | polish | 2000 | 8.199 |
| mlsml_pt_train | facebook/multilingual_librispeech | portuguese | 2400 | 10.329 |
| ps_train | MLCommons/peoples_speech | clean | 25000 | 95.934 |
| vp_cs_train | facebook/voxpopuli | cs | 1000 | 2.843 |
| vp_de_train | facebook/voxpopuli | de | 5000 | 11.908 |
| vp_es_train | facebook/voxpopuli | es | 5000 | 14.583 |
| vp_fi_train | facebook/voxpopuli | fi | 601 | 1.595 |
| vp_fr_train | facebook/voxpopuli | fr | 5000 | 13.938 |
| vp_hu_train | facebook/voxpopuli | hu | 1000 | 2.704 |
| vp_it_train | facebook/voxpopuli | it | 2500 | 8.517 |
| vp_nl_train | facebook/voxpopuli | nl | 2000 | 4.547 |
| vp_pl_train | facebook/voxpopuli | pl | 2000 | 5.377 |
| vp_ro_train | facebook/voxpopuli | ro | 1000 | 3.041 |
| vp_train | facebook/voxpopuli | en | 30000 | 82.674 |

Sanity (`data/prep/manifest_and_sanity.py`): row counts, zstd, no duplicate clip ids, licenses all
in the approved set, decode check of 5 clips per dataset/split (127 groups, all ok), LibriSpeech /
LibriTTS-R train chapter-disjoint. **Cross-language speaker overlap:** the per-language filters left
train speakers who also appear in another language's eval set (Common Voice contributors and MEPs
speak several languages; some LibriVox readers read MLS German as well as English). Resolution:
`data/scripts/eval_speaker_exclude.py` lists every train clip whose speaker (per id family) is in
any eval set -- **8,676 clips (3.1 %)**, `data/filters/train_exclude_eval_speakers.txt` -- and the
trainer drops them (`train/data.py`). Common Voice speaker ids are `sha256("vxdf-cv17:" + client_id)[:16]`
(salt name inherited from the voxtral-dflash builder; consistent across this corpus; no client ids
are stored or published).

## 1. Summary

| # | Dataset (HF id) | Revision (`sha`) | Gated | Card license | Upstream license | Mismatch? | Config / splits used | Role | OK to republish derivatives |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `openslr/librispeech_asr` | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | no | cc-by-4.0 | CC BY 4.0 (openslr.org/12, `LICENSE.TXT`) | no | `all`: train.clean.100/360, train.other.500 (train); test.clean, test.other (eval) | train + eval | **yes** (attribution) |
| 2 | `mythicinfinity/libritts_r` | `0d1718db351207d5a493135a62d5ed8cc4733830` | no | cc-by-4.0 | CC BY 4.0 (openslr.org/141) | see note A | `all`: train.* | train | **yes** (attribution) |
| 3 | `parler-tts/mls_eng` | `faf6604dcf0bb9ec0d9c280a140ad74da9c931b0` | no | cc-by-4.0 | CC BY 4.0 (openslr.org/94) | no | `default`: train | train | **yes** (attribution) |
| 4 | `fsicoli/common_voice_17_0` (`en`) | `8262c16bf297c87a9cd88c51997c4758ed7a8ba2` | no | cc0-1.0 | CC0-1.0 (Mozilla Data Collective listing) | no; see note B | `en`: train (train); test (eval) | train + eval | **yes** (no-re-identification condition; do not re-host audio) |
| 5 | `facebook/voxpopuli` (`en`) | `42f01879c780b4a2e90ec0b4f616c2ece526e4f1` | no | cc0-1.0 + "other" | CC0 (github voxpopuli) + European Parliament legal notice for the raw audio | see note C | `en`: train (filtered), test (eval) | train + eval | **yes** (acknowledge the European Parliament) |
| 6 | `MLCommons/peoples_speech` (`clean`) | `f10597c5d3d3a63f8b6827701297c3afdf178272` | no | cc-by-2.0/2.5/3.0/4.0 + cc-by-sa-3.0/4.0 (whole repo) | CC BY 4.0 dataset; per-source CC BY / public domain in `clean`, CC BY-SA only in `*_sa` | see section 3 | `clean`: train only, per-source filtered | train | **yes**, `clean` only, per-source license filter applied |
| 7 | `edinburghcstr/ami` | `46f28f2503e2ec48f8867a84eef356c70476beab` | no | cc-by-4.0 | CC BY 4.0 (groups.inf.ed.ac.uk/ami/corpus/license.shtml) | no | `ihm`: train (filtered), test (eval) | train + eval | **yes** (attribution) |
| 8 | `google/fleurs` (`en_us`) | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | no | cc-by-4.0 | CC BY 4.0 (card; arXiv:2205.12446) | see note D | `en_us`: test only | eval only | **yes** for eval use (we do not redistribute it) |
| M | `Qwen/Qwen3-ASR-1.7B` (target model) | `7278e1e70fe206f11671096ffdd38061171dd6e5` | no | apache-2.0 | Apache-2.0 | no | target for labels and features | teacher | **yes**: draft released as Apache-2.0 with attribution (section 4) |

"OK to republish derivatives" covers what this project publishes: the trained draft weights, the
training code, and (optionally) a label/feature manifest of clip ids plus target-generated text.
**We do not plan to re-host any source audio.** Users fetch the audio from the original HF repos.
That keeps us clear of every redistribution condition below.

### Notes on mismatches / caveats

- **A. LibriTTS-R:** the card's BibTeX entry carries `copyright = "http://creativecommons.org/licenses/by-nc-nd/4.0/"`.
  That is the arXiv *paper* license (2305.18802). The dataset is CC BY 4.0 on both the card
  (`License: CC BY 4.0`) and openslr.org/141 (`License: CC BY 4.0`). This is not a data-license
  mismatch, but we will cite the paper rather than copy its text. The card's example code points to
  `blabble-io/libritts_r`; we use the `mythicinfinity/libritts_r` repo, which is the one on the approved list.
- **B. Common Voice:** the official HF repo `mozilla-foundation/common_voice_17_0` (sha
  `11dc88355e899d1bf2df74f01b904a8544a17b33`) is now an empty stub. Since October 2025 its card has
  read: "Mozilla Common Voice datasets are now exclusively available through Mozilla Data Collective."
  It is not gated and has no data. Mozilla Data Collective (MDC) lists every Common Voice Scripted
  Speech release as "Creative Commons Zero v1.0 Universal (CC0-1.0)" (MDC public API, queried
  2026-10-08). The MDC platform terms (datacollective.mozillafoundation.org/terms) prohibit
  (viii) "attempt to determine, trace, match the identity of, or re-identify the individual
  contributor" and (ix) mirroring or hosting a Dataset "on any platform ... other than the Platform".
  MDC's FAQ explains why: consent revocation. The fsicoli mirror predates that change (last
  modified 2024-08-08) and is CC0 on its card. CC0 itself has no conditions. The re-identification
  and no-mirroring terms bind people who download through MDC. **Our rules regardless:** (1) never
  attempt speaker identification, and never publish `client_id`s, demographic fields, or anything
  linking clips to people; (2) never re-host Common Voice audio or the TSVs; publish only clip
  filenames (`common_voice_en_<n>.mp3`) and our own model outputs. Old HF cards for
  Common Voice (e.g. the archived `mozilla-foundation/common_voice_11_0` card) carried the same
  condition: "You agree to not attempt to determine the identity of speakers in the Common Voice
  dataset." Both conditions are compatible with releasing a draft model trained on target outputs.
- **C. VoxPopuli:** the card lists `cc0-1.0` and `other`. "Other" refers to the European Parliament
  legal notice for the raw recordings. Under that notice, reuse "for personal use or for further
  non-commercial or commercial dissemination" is authorised "provided that the entire item is
  reproduced and the source is acknowledged". It also says partial reproduction must cite the
  source URL. We do not redistribute audio, so the reproduction clause does not apply to us. We
  include the acknowledgement anyway (section 5). The pre-trained *models* in the VoxPopuli repo
  are CC BY-NC 4.0. We do not use them.
- **D. FLEURS:** the card says "All datasets are licensed under the Creative Commons license
  (CC-BY)", and the card metadata says `cc-by-4.0`. The FLEURS transcripts are the English
  side of FLoRes, whose current HF repos (`openlanguagedata/flores_plus`, `facebook/flores`) are
  CC BY-SA 4.0. We use FLEURS **for evaluation only**. We neither train on nor redistribute its
  text, so the share-alike question does not arise. If anyone proposes training on FLEURS, raise
  this as an issue first.
- **E. LibriSpeech and MLS cards** both say: "You agree to not attempt to determine the identity
  of speakers in this dataset." We apply the same rule as for Common Voice. The LibriSpeech
  metadata (`SPEAKERS.TXT`) contains LibriVox reader names. We used only `CHAPTERS.TXT` (book and
  project ids), deleted the local copy of `SPEAKERS.TXT`, and will never publish reader names.

## 2. Per-dataset detail

Row counts come from the HF dataset-viewer `/size` API at the pinned revision (`partial: false`).
Hours come from our scans where marked (s), otherwise from the card or paper (c).

### 2.1 `openslr/librispeech_asr` (LibriSpeech)

| Field | Value |
|---|---|
| Revision | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` (lastModified 2025-07-25), parquet, not gated |
| License | card `cc-by-4.0`; openslr.org/12 "License: CC BY 4.0"; `LICENSE.TXT`: "LibriSpeech (c) 2014 by Vassil Panayotov. LibriSpeech ASR corpus is licensed under a Creative Commons Attribution 4.0 International License." Source audio: LibriVox (public domain); texts: Project Gutenberg |
| Splits used | `all`/train.clean.100 (28,539 rows, 100.6 h), train.clean.360 (104,014, 363.6 h), train.other.500 (148,688, 496.9 h); eval: test.clean (2,620, 5.4 h), test.other (2,939, 5.3 h). Hours (s) from `CHAPTERS.TXT` |
| Held out, never trained | validation.clean (2,703), validation.other (2,864) |
| Clip length | apply the <= 30 s cap (filter on decoded length at corpus build) |
| Attribution | "LibriSpeech ASR corpus (c) 2014 Vassil Panayotov, CC BY 4.0, derived from LibriVox public-domain audiobooks"; cite Panayotov et al., ICASSP 2015 |
| Verdict | OK to republish derivatives: **yes** |

### 2.2 `mythicinfinity/libritts_r` (LibriTTS-R)

| Field | Value |
|---|---|
| Revision | `0d1718db351207d5a493135a62d5ed8cc4733830` (2024-02-09), parquet (+ `libritts_asr_builder.py`, not needed), not gated |
| License | card `cc-by-4.0` / "License: CC BY 4.0"; openslr.org/141 "License: CC BY 4.0"; LibriTTS (openslr.org/60) CC BY 4.0 and, per its paper (arXiv:1904.02882), "distributed under the same non-restrictive license" as LibriSpeech |
| Splits | train.clean.100 (33,232), train.clean.360 (116,426), train.other.500 (205,035); 53.8 + 191.3 + 310.1 = 555.2 h (c, LibriTTS paper Table 2). Not used: dev.*, test.* |
| Attribution | "LibriTTS-R (Koizumi et al., 2023), CC BY 4.0, derived from LibriTTS (Zen et al., 2019) and LibriSpeech" |
| Verdict | **yes** |

### 2.3 `parler-tts/mls_eng` (Multilingual LibriSpeech, English)

| Field | Value |
|---|---|
| Revision | `faf6604dcf0bb9ec0d9c280a140ad74da9c931b0` (2024-04-09), parquet (1,416 train shards), not gated |
| License | card `cc-by-4.0` ("Public Domain, Creative Commons Attribution 4.0 International Public License"); openslr.org/94 "License: CC BY 4.0" |
| Splits | train 10,808,037 rows, 44,659.7 h (s, sum of `audio_duration`; matches the card), 5,490 speakers; dev 3,807 (15.75 h), test 3,769 (15.55 h). Only train is used. All clips are <= 30 s (s) |
| Attribution | "Multilingual LibriSpeech (Pratap et al., 2020), CC BY 4.0, derived from LibriVox" |
| Verdict | **yes** |

### 2.4 `fsicoli/common_voice_17_0` (Common Voice 17.0, English)

| Field | Value |
|---|---|
| Revision | `8262c16bf297c87a9cd88c51997c4758ed7a8ba2` (2024-08-08), **not gated** |
| Format | **script-based loader** (`common_voice_17_0.py`), audio as tar shards (`audio/en/<split>/en_<split>_<n>.tar`, 28 train shards totalling 45.1 GB, 1 test shard of 711 MB) plus TSVs (`transcript/en/*.tsv`). The HF dataset viewer refuses it ("runs arbitrary Python code"). `datasets` >= 4 no longer runs loading scripts, so the corpus build should **read the tars + TSVs directly** (the tar member names are the TSV `path` values). |
| Completeness | train.tsv 1,101,170 rows, dev 16,393, test 16,393. These match the official CV 17.0 release stats for `en` (common-voice/cv-dataset `cv-corpus-17.0-2024-03-15.json`: train 1,101,170, dev 16,393, test 16,393). We downloaded the test tar: 16,393 members, 0 missing versus test.tsv. Hours (s, `clip_durations.tsv`): train 1,738.5 h (mean 5.7 s), test 27.0 h |
| License | card `cc0-1.0` ("Public Domain, CC-0"); MDC lists Common Voice releases as CC0-1.0. Conditions: see note B |
| Official alternative | `mozilla-foundation/common_voice_17_0` is now an empty stub that points to Mozilla Data Collective, where CV is account-gated (MDC API key). Not needed: the fsicoli mirror is complete for `en` |
| Split hygiene | (s) train/dev/test share **0** `client_id`s, 0 `sentence_id`s and 0 paths (`data/filters/cv_split_check.json`). CorporaCreator splits are speaker- and sentence-disjoint |
| Attribution (courtesy; CC0 needs none) | "Common Voice Corpus 17.0, Mozilla Foundation, CC0-1.0"; cite Ardila et al., LREC 2020 |
| Verdict | **yes**, conditions: no re-identification, no re-hosting of audio/TSVs |

### 2.5 `facebook/voxpopuli` (`en`)

| Field | Value |
|---|---|
| Revision | `42f01879c780b4a2e90ec0b4f616c2ece526e4f1` (2026-01-30), parquet, not gated |
| License | card `cc0-1.0` + `other`; github facebookresearch/voxpopuli: "VoxPopuli Data: CC0 (see also European Parliament's legal notice for the raw data)". See note C |
| Splits | en/train 182,482 rows, ~522.6 h (s, estimated from parquet footer byte sizes: 16 kHz float32 WAV); validation 1,753 (~5.0 h); test 1,842 (~4.9 h). en_accented/test 8,387 (~26.5 h). Card: 543 transcribed hours, 1,313 speakers |
| Clip length | segmented to <= 20 s (card) |
| Speaker filter | see section 6: drop 47,276 train clips (25.9 %), leaving 135,206 (~387 h, proportional estimate) |
| Attribution | "VoxPopuli (Wang et al., ACL 2021), CC0. Raw data: European Parliament event recordings 2009-2020, (c) European Union -- Source: European Parliament. We acknowledge the European Parliament for creating and sharing these materials." |
| Verdict | **yes** |

### 2.6 `MLCommons/peoples_speech` (`clean`)

See section 3 for the license decision.

| Field | Value |
|---|---|
| Revision | `f10597c5d3d3a63f8b6827701297c3afdf178272` (2024-11-20), parquet, not gated |
| Configs | `clean` (train 1,501,271 / validation 18,622 / test 34,898), `clean_sa` (train 257,093), `dirty` (5,476,898), `dirty_sa` (548,014), `microset`, standalone `validation`, `test` |
| Used | `clean`/train only: 1,501,271 clips, 5,894.7 h (s), 54,190 archive.org source items (s), clips <= 16.6 s, mean 14.1 s. After the per-source filter: 1,500,679 clips, 5,893.1 h |
| Not used | `clean_sa`, `dirty_sa` (share-alike), `dirty` (not approved), validation/test (not needed) |
| Attribution | "The People's Speech (Galvez et al., NeurIPS 2021 Datasets and Benchmarks), MLCommons, CC BY 4.0 / CC BY-SA 4.0 subsets. We use the CC BY `clean` subset only." Per-source credits: MLCommons `credits.csv` (licenseurl, creator, identifier), plus our `data/ps_clean_source_licenses.csv` |
| Verdict | **yes**, `clean` only, after the per-source filter |

### 2.7 `edinburghcstr/ami` (AMI Meeting Corpus)

| Field | Value |
|---|---|
| Revision | `46f28f2503e2ec48f8867a84eef356c70476beab` (2026-01-13), parquet, not gated |
| License | card `cc-by-4.0`; groups.inf.ed.ac.uk/ami/corpus/license.shtml: "The AMI corpus and its annotations are released under the Creative Commons Attribution 4.0 license agreement (also called CC BY 4.0)". Some legacy annotations used an older NC-SA-like AMI license. We use only signals and orthographic transcripts, which are CC BY 4.0 |
| Data | Kaldi `s5` segmentation. `ihm`: train 108,502 clips (77.9 h, s), validation 13,098 (8.9 h), test 12,643 (8.7 h); mean 2.5 s per clip, max 33 s. `sdm` is the same split with a far-field mic (not planned) |
| Splits | the HF splits match the AMI **full-corpus-ASR partition** (groups.inf.ed.ac.uk/ami/corpus/datasets.shtml), except for IB4005 (see section 6). Test (SC) = EN2002, ES2004, IS1009, TS3003 (16 meetings); validation (SB) = ES2011, IS1008, TS3004, IB4001-4004, IB4010, IB4011 |
| Long-form eval | the HF data only has short segments. For 30 s - 5 min eval windows, use the official AMI download (`Mix-Headset.wav` per meeting, CC BY 4.0, e.g. `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/EN2002a/audio/EN2002a.Mix-Headset.wav`, 68.6 MB, HTTP 200 checked) for the 16 test meetings, with references concatenated from the HF `ihm` test segments by `begin_time` |
| Attribution | "AMI Meeting Corpus (Carletta, 2006; AMI Consortium), CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/" |
| Verdict | **yes** |

### 2.8 `google/fleurs` (`en_us`)

| Field | Value |
|---|---|
| Revision | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (2026-05-15), parquet (`parquet-data/en_us/`), not gated |
| License | card `cc-by-4.0`. See note D |
| Used | `en_us`/test: 647 clips, 1.77 h (s), 350 unique sentences. Eval only. train (2,602 clips, 7.5 h) and validation (394) are not used |
| Split hygiene | FLEURS paper: "train, development (dev) and test sets with disjoint speakers between train/dev and test"; segments are <= 30 s. Sentences are disjoint by construction (1509/150/350 split of FLoRes sentences) |
| Attribution | "FLEURS (Conneau et al., 2022), CC BY 4.0" |
| Verdict | **yes** (eval use; not redistributed) |

## 3. People's Speech: per-sample license decision

**Question:** is `clean` CC BY (not share-alike) per sample?

**Evidence**
1. The HF card says: "We provide the following configurations ... `cc-by-clean` (`"clean"`),
   `cc-by-dirty` (`"dirty"`), `cc-by-sa-clean` (`"clean_sa"`), `cc-by-sa-dirty` (`"dirty_sa"`)". So the
   license split is at the **config** level. The card's `license:` list (cc-by-2.0 ... cc-by-sa-4.0)
   covers the whole repo.
2. **Samples carry no license field.** The schema is `id, audio, duration_ms, text`. Each clip `id`
   encodes the archive.org item identifier (`<identifier>/<file>_<NNNNN>.flac`, with `.` written as
   `_DOT_` and `/` as `_SLASH_`).
3. The paper (arXiv:2111.09344, section 3.1) says the corpus "consists of public domain,
   CC-BY-licensed, and CC-BY-SA-licensed data". It says the authors "separate the CC-BY-SA data
   (3,100 hours) from the rest (27,700 hours)". It also says that "a commercial user ... could
   straightforwardly filter out the CC-BY-SA sources using this CSV file" (`credits.csv`, linked
   from mlcommons.org/datasets/peoples-speech: `licenseurl, creator, identifier, credits`).
   CC BY-NC and CC BY-ND sources were excluded at collection time.
4. **Our scan** (`data/scripts/ps_clean_license_scan.py`, id + duration columns of all 804
   `clean/train` shards, joined to `credits.csv`):
   - 1,501,271 clips, 5,894.7 h, 54,190 source identifiers.
   - In `credits.csv`: CC BY 3.0 163,179 clips (630.7 h), CC BY 2.0 6,168 (24.0 h), CC BY 4.0 3,351
     (13.3 h), CC BY 2.5 23 (0.1 h). **0 CC BY-SA.**
   - Not in `credits.csv`: 49,820 identifiers (1,328,550 clips, 5,226.7 h). `credits.csv` lists
     only CC-licensed works that need attribution. Public-domain and US-government items are not
     listed. We looked up every missing identifier through the public archive.org metadata API
     (`data/scripts/ps_full_license_table.py`). Result: of the 54,190 identifiers, 4,370 come from `credits.csv` and 49,820 from archive.org. By clips: US Government Works 885,348 (3,519.1 h), CC Public Domain Mark 232,769 (910.0 h), archive.org `publicdomain` 207,766 (790.4 h), CC BY 3.0 163,180 (630.7 h), CC BY 2.0 6,168 (24.0 h), CC BY 4.0 3,351 (13.3 h), CC0 2,074 (5.5 h), CC BY 2.5 23 (0.1 h). **0 CC BY-SA, 0 NC, 0 ND.** Not allowed (dropped): 578 clips (1.6 h) whose item has no `licenseurl`, and 14 clips (0.0 h) marked `publicdomain-prelinger`. **Kept: 1,500,679 clips, 5,893.1 h from 54,146 sources** (`data/ps_clean_source_licenses.csv`). The `clean` config is therefore effectively CC BY / public domain per source, not mixed with SA.
   - Control: identifiers from 8 random `clean_sa` shards map to cc-by-sa-4.0 (229) / cc-by-sa-3.0
     (105) / not-in-credits (1). `clean` and `clean_sa` share 0 identifiers. The join works, and the
     config split matches the per-source license.
   - Spot check: 50 random `clean` identifiers that are in `credits.csv` agree with archive.org in
     41/50 cases. The other 9 were 7 items that now have no `licenseurl` on archive.org and 2
     timeouts, with no license conflicts.

**Decision**
- Use `clean` only. Never use `clean_sa`, `dirty_sa` or `dirty`.
- **Apply a per-source allow-list filter** (cheap, and it protects against metadata drift). Keep a
  clip only if its source identifier has `allowed=1` in `data/ps_clean_source_licenses.csv`.
  Allowed licenses: CC BY 2.0 / 2.5 / 3.0 (incl. ported 3.0 us/es/de) / 4.0, CC0, Public Domain
  Mark, and US Government Works (usa.gov/government-works). Anything else is dropped: SA, NC, ND,
  no `licenseurl`, or lookup error.
- Attribution: CC BY 2.0-4.0 require attribution of the work's creator. We will ship
  `data/ps_clean_source_licenses.csv` (identifier, license) plus the matching `credits.csv` rows
  (creator) for every source used, as `ATTRIBUTION_peoples_speech.csv` in the release. A trained
  draft is not an adaptation that reproduces the audio, but listing sources keeps the release
  unambiguous.

## 4. Target model: `Qwen/Qwen3-ASR-1.7B`

| Field | Value |
|---|---|
| Revision | `7278e1e70fe206f11671096ffdd38061171dd6e5`; weights identical to `Qwen/Qwen3-ASR-1.7B-hf` `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` (checked tensor by tensor; the -hf repo is what the code loads) |
| License | model card `apache-2.0` |
| Gate | not gated |
| Training on outputs | Apache-2.0 places no restriction on the use of a model's outputs. Target-generated transcripts and hidden states are outputs; the draft trained on them is our own work. The release does **not** copy the target's embedding / LM-head tensors (the draft references the target's, as public DFlash checkpoints do), so only attribution applies |
| Release plan | draft under Apache-2.0; model card and NOTICE state that it was trained on outputs and hidden states of Qwen/Qwen3-ASR-1.7B (Apache-2.0, Qwen team) and list the data sources |
| Verdict | **yes** |

## 5. Attribution text

The release attribution is the `NOTICE` file of the code repository (copied into the model repo):
target model, method / format, every training dataset with license, citation and the required
acknowledgements (European Parliament for VoxPopuli; per-source People's Speech credits in
`ATTRIBUTION_peoples_speech.csv`; the no-speaker-identification condition), and FLEURS as
evaluation-only data. No source audio is redistributed.

## 6. Train / eval separation

Eval sets (never trained on): LibriSpeech test.clean + test.other, Common Voice 17 `en` test,
VoxPopuli `en` test, FLEURS `en_us` test, AMI `ihm` test (full-corpus-ASR SC meetings EN2002,
ES2004, IS1009, TS3003) as short segments **and** as 30 s - 5 min long-form windows from the
Mix-Headset recordings. Optional extras: VoxPopuli `en_accented` test, LibriSpeech validation
(pilot checks only).

Everything below was computed with `data/scripts/speaker_overlap.py` (speaker, chapter, book and
meeting id columns only). Results are in `data/filters/speaker_overlap_report.json`.

| Check | Result | Source / evidence | Rule |
|---|---|---|---|
| LibriSpeech train vs test/dev speakers | 0 / 0 (2,338 train speakers; test.clean 40, test.other 33) | scan; LibriSpeech README | none |
| **LibriTTS-R train vs LibriSpeech test/dev** | speakers 0, chapters 0. All 2,311 LibriTTS-R train speakers are LibriSpeech train speakers; LibriTTS-R dev/test speakers equal LibriSpeech dev/test speakers (40/40, 33/33, 39/39, 33/33) | scan; LibriTTS paper: "same speakers and subset split as the LibriSpeech corpus" | none (do not use LibriTTS-R dev/test) |
| **MLS-en train vs LibriSpeech test/dev** | speakers **0** for all four LibriSpeech eval splits. MLS train shares 1,677 of 2,338 speakers with LibriSpeech *train* (harmless) | scan; MLS paper section 4.1: "For English ... we additionally make it exclusive to the previous LibriSpeech dataset ... there is speaker and book overlap between the two train sets, but the development and test sets are completely independent from one another" | none |
| MLS train vs MLS dev/test | 0 / 0 | scan | none |
| Book (text) overlap, LibriVox family | MLS train `book_id` (LibriVox project id) matches 2 of 54 test-clean and 2 of 50 test-other LibriSpeech projects. Note that LibriSpeech's *own* train shares 42-43 projects with its test sets (different chapters/readers), which is the accepted protocol | scan with `CHAPTERS.TXT` | covered by the text n-gram filter below |
| Common Voice train vs test | 0 client_ids, 0 sentences, 0 clips | `cv_split_check.py` | none |
| **VoxPopuli en train vs test** | the paper says splits have "disjoint speakers". In the HF data, though, 108 of the 286 test speaker ids, 37 of 71 validation ids and 238 of 364 en_accented ids also appear in train (excluding the placeholder `"None"`, used for 64,260 train / 150 test clips). No shared speech paragraphs or clips (0 / 619) | scan; arXiv:2101.00390 section 3 | **drop train clips whose `speaker_id` is in `data/filters/voxpopuli_drop_speakers.json`** (555 ids = en validation + test + en_accented test real speakers): -47,276 clips, 135,206 kept. Use only the en test (and optionally en_accented) for eval |
| **AMI train vs test** | meetings 0, speakers **0** | scan | none |
| AMI train vs validation | meetings 0. Speakers FIE038 and MIO036 occur in train meeting **IB4005** (866 clips) and in validation. IB4005 is in none of the published partition lists | scan; AMI full-corpus-ASR partition | **drop IB4005 from train** (`data/filters/ami_drop.json`) |
| FLEURS vs anything | eval only. Train/dev speakers are disjoint from test (paper). Text is FLoRes Wikipedia sentences | FLEURS paper | never train on FLEURS. The text filter below checks the n-grams of the FLEURS test text |
| People's Speech vs eval | sources are archive.org items (council meetings, lectures, etc.). No speaker ids, and no source family shared with the eval sets. 0 clips containing LibriVox items | scan | text filter only |

**Transcript leakage filter** (`data/scripts/text_overlap.py`, text columns only). A train clip is
flagged if its normalized transcript (lowercase, alphanumerics + apostrophe) has >= 8 words and
exactly equals an eval transcript, or shares any 12-word n-gram with one. The eval pool is
LibriSpeech test, CV17 test, VoxPopuli test + en_accented, FLEURS test and AMI ihm test.

| Train set | Rows scanned | Flagged | Main eval-set hit |
|---|---|---|---|
| LibriSpeech train | 281,241 | 101 (0.036 %) | LibriSpeech test 97 |
| LibriTTS-R train | 354,693 | 70 (0.020 %) | LibriSpeech test 63 |
| MLS train (60 of 1,416 shards sampled) | 457,968 | 39 (0.009 %) | LibriSpeech test 39 |
| CV17 en train | 1,101,170 | 83 (0.008 %) | CV17 test 78 (near-duplicate sentence text under a different sentence_id) |
| VoxPopuli en train | 182,482 | 17 (0.009 %) | VoxPopuli test 16 |
| People's Speech clean (60 of 804 shards sampled) | 112,033 | 0 | -- |
| AMI ihm train | 108,502 | 7 (0.006 %) | AMI test 7 |

**Rule:** drop every flagged clip (`data/filters/text_overlap_flagged_<set>.txt`). Before the corpus build
samples from MLS and People's Speech, re-run `text_overlap.py` on the actually-sampled clips of
those two corpora (the full scans were sampled here). For FLEURS, the 12-gram check over all
train sets found 0 hits against FLEURS test.

## 7. Planned training mix

Goal: several hundred hours and >= 100k clips (corpus build). Headline set: clips <= 30 s. AMI
long-form is eval only. Proportions favour diversity over raw size, so no corpus exceeds about
25 % of clips. Hours for the planned clips are estimated from each corpus's mean clip length.

| Dataset | Config / split | Domain | Available after filters | Planned clips | Planned hours (approx) |
|---|---|---|---|---|---|
| LibriSpeech | all / train.* | read audiobook | 281,140 clips / ~961 h | 25,000 | 85 |
| LibriTTS-R | all / train.* | read audiobook (restored, 24 kHz; resample to 16 kHz) | 354,623 / ~555 h | 15,000 | 23 |
| MLS English | default / train | read audiobook, 5.5k readers | 10.8 M / 44,660 h | 20,000 | 83 |
| Common Voice 17 | en / train | crowd-sourced read, many accents | 1,101,087 / ~1,738 h | 40,000 | 63 |
| VoxPopuli | en / train | parliamentary, semi-spontaneous, L2 accents | ~135,189 / ~387 h | 30,000 | 86 |
| People's Speech | clean / train (per-source allow-list) | conversational, public meetings, talks | 1,500,679 / 5,893 h | 25,000 | 98 |
| AMI | ihm / train (no IB4005, >= 1.5 s) | meetings, spontaneous | ~56,426 / 67.7 h | 20,000 | 24 |
| **Total** | | | | **175,000** | **~462** |

Notes:
- LibriSpeech and LibriTTS-R come from the same LibriVox chapters. Sample LibriTTS-R only from
  chapters not sampled from LibriSpeech, so the two do not duplicate content.
- AMI clips average 2.5 s. The >= 1.5 s floor drops backchannels ("okay", "yeah") that carry
  almost no draft signal. For more AMI hours, merge adjacent same-speaker segments into windows of
  up to 30 s. That is still train-split audio.
- The mix can scale about 3x (to ~500k clips) without changing the filters, because every corpus
  except AMI has much more data available. The final size is set by the acceptance ablations.
- Machine-readable version: `data/manifest_plan.json` (generated by
  `data/scripts/make_manifest_plan.py`).

## 8. Reproducing the numbers

```
cd data/scripts
python ps_clean_license_scan.py --config clean --split train --credits <credits.csv> --out ../../tmp/ps
python ps_archive_license_check.py --ids ../../tmp/ps/ps_clean_train_ids.parquet --credits <credits.csv> --out ../../tmp/ps
python ps_full_license_table.py   --ids ../../tmp/ps/ps_clean_train_ids.parquet --credits <credits.csv> --out ../../tmp/ps
python speaker_overlap.py --out ../../tmp/spk          # LibriSpeech, LibriTTS-R, MLS, VoxPopuli, AMI
python voxpopuli_filter.py --spk ../../tmp/spk/speakers_voxpopuli.json --out ../../tmp/spk
python voxpopuli_hours.py
python cv_split_check.py ../../tmp/cv                  # needs transcript/en/{train,dev,test}.tsv
python text_overlap.py --cv ../../tmp/cv --out ../../tmp/text
python make_manifest_plan.py --ps-table ../ps_clean_source_licenses.csv > ../manifest_plan.json
```
`credits.csv`: https://storage.googleapis.com/example_download_data/credits.csv (linked from
mlcommons.org/datasets/peoples-speech). The LibriSpeech `CHAPTERS.TXT` comes from
`raw-metadata.tar.gz` on openslr.org/12.

## 9. Corpus build

Built on 2026-10-09 on the project trn2 instance (CPU only; no Neuron devices used) with
`data/prep/build_corpus.py` (+ `ami_longform.py`, `manifest_and_sanity.py`, `report.py`). Every dataset is
read at the revision pinned in section 1 / `manifest_plan.json`. Only the row groups (or CV tars) that hold
selected clips were downloaded. Sampling is reproducible (seed 0).

**Output format.** zstd parquet shards of at most 2,000 rows. Columns: `clip_id`
(`<short>/<split>/<native id>`), `dataset`, `config`, `split`, `revision`, `license`, `speaker_id`,
`duration_s` (float32, measured on the decoded audio), `ref_text`, `audio_pcm16` (16 kHz mono
little-endian int16), and `source_path`. Manifests have every column except the audio, plus `shard` and `row`.

| Where | Path |
|---|---|
| Shards (instance) | `~/vxdf_corpus/shards/{train,eval}/<short>/<split>/shard-NNNNN.parquet` |
| Shards (S3) | `s3://<bucket>/corpus/shards/{train,eval}/...` |
| Manifests | `.../corpus/shards/{train,eval}_manifest.parquet` (S3 + instance), local copy `data/corpus_manifest/` |
| Build stats / sanity report | `.../corpus/stats/` (S3), local copy `data/corpus_manifest/stats/` |
| Build code snapshot | `.../corpus/code/prep/` (S3), source in `data/prep/` |

### 9.1 Per set: clips, hours, filter drops

"Pool" is the number of rows in the pinned HF split(s). "Dropped (select)" counts rows removed by the
filters before sampling. "Decode rejects" counts sampled candidates whose decoded length fell outside
[1, 30] s (train) or [1.5, 30] s (AMI). Those candidates were replaced by the next ones in sampling order.

| Set | Role | Split(s) | Clips | Hours | Speakers | Pool | Dropped (select) | Decode rejects |
|---|---|---|---|---|---|---|---|---|
| LibriSpeech | train | train.clean.100 / .360 / other.500 | 2,720 / 9,843 / 12,437 = **25,000** | **85.90** | 2,154 | 281,241 | chapters reserved for LibriTTS-R 131,059; text-overlap list 36 | 0 |
| LibriTTS-R | train | train.clean.100 / .360 / other.500 | 1,614 / 5,710 / 7,676 = **15,000** | **27.17** | 1,981 | 354,693 | chapters reserved for LibriSpeech 186,144; text-overlap list 43 | 747 (< 1 s) |
| MLS English | train | train | **20,000** | **82.69** | 5,193 | 10,808,037 | text overlap vs eval 15 (re-run on the 200k-row drawn prefix) | 0 |
| Common Voice 17 | train | train | **40,000** | **65.44** | 37,475 | 1,101,170 | metadata duration outside 1–30 s: 4; text-overlap list 83 | 0 |
| VoxPopuli | train | train | **30,000** | **82.67** | 320 | 182,482 | speaker in eval drop list 47,276; unknown speaker (`None`) 64,260; text-overlap list 4 | 357 (> 30 s or < 1 s) |
| People's Speech | train | clean / train | **25,000** | **95.93** | 3,317 sources | 1,501,271 | source license not allowed 592; text overlap vs eval 1 (re-run on the full allowed pool) | 0 |
| AMI | train | ihm / train | **20,000** | **24.20** | 152 | 108,502 | < 1.5 s or > 30 s: 51,652; meeting IB4005: 421; text-overlap list 7 | 0 |
| **Train total** | | | **175,000** | **464.0** | | | | |
| LibriSpeech | eval | test.clean / test.other (all) | 2,620 / 2,939 | 5.40 / 5.34 | 40 / 33 | 5,559 | none | 0 |
| Common Voice 17 | eval | test (random 2,000) | 2,000 | 3.27 | 1,939 | 16,393 | none | 0 |
| VoxPopuli | eval | en test (all) | 1,842 | 4.93 | 285 (+ `None`) | 1,842 | none | 0 |
| FLEURS | eval | en_us test (all) | 647 | 1.77 | - | 647 | none | 0 |
| AMI | eval | ihm test (random 2,000 of the segments >= 1.5 s) | 2,000 | 2.44 | 16 | 12,643 | < 1.5 s or > 30 s: 6,304 | 0 |
| AMI long-form | eval | `ami_longform/test`: 60 windows (42–298 s, mean 177 s) from the 16 SC test meetings | 60 | 2.95 | 16 | - | - | - |
| **Eval total** | | | **12,108** | **26.1** | | | | |

Train licenses by clip: CC-BY-4.0 80,000; CC0-1.0 70,042; US Government Works 19,538; Public Domain 1,990;
CC-BY-3.0 1,816; Public Domain Mark 1,534; CC-BY-2.0 80. The six non-CC-BY-4.0 / non-CC0 labels all come
from People's Speech and are set per clip from `data/ps_clean_source_licenses.csv`.

### 9.2 Sampling and diversity

- **LibriSpeech / LibriTTS-R:** every LibriVox chapter in the two train sets is assigned to exactly one corpus
  (per reader, alternating, seed 0). LibriSpeech draws only from its chapters and LibriTTS-R only from the rest.
  The draw then round-robins over readers, and over chapters within each reader. The sanity check confirms
  0 shared chapters between the two drawn sets.
- **MLS:** readers are shuffled. Each contributes up to 4 clips from one random row group, which limits the
  download to about 30 GB. Result: 5,193 readers and 2,656 books.
- **CV / VoxPopuli / AMI:** round-robin over speakers. CV covers 37,475 speakers at about 1 clip each.
- **People's Speech:** sources are shuffled, up to 10 clips per source from one row group. Result: 3,317 sources.
- **Eval subsets:** CV test and AMI test are seeded random draws. The other eval sets are complete.

### 9.3 Deviations / notes

- **Common Voice `speaker_id`** is `sha256("vxdf-cv17:" + client_id)[:16]`, not the Mozilla `client_id`. This
  follows note B (never publish client ids). Speaker grouping and the train/eval speaker check still work.
- **VoxPopuli `ref_text`** is `raw_text` (orthographic). It falls back to `normalized_text` when `raw_text` is
  empty: 5,463 pool rows in train and 52 in eval. Train excludes clips whose speaker is `None`, per the section 6
  rule. Eval keeps them, with `speaker_id` set to null.
- **Eval clip lengths are not capped:** LibriSpeech test goes up to 35.0 s and VoxPopuli test up to 69.2 s,
  because those sets are kept whole. AMI eval segments are 1.5–30 s.
- **People's Speech** has no speaker ids, so `speaker_id` is null.
- **AMI long-form:** the audio is the official `Mix-Headset.wav` from `groups.inf.ed.ac.uk/ami/AMICorpusMirror`
  (CC BY 4.0, the source named in section 2.7). Segment times and text come from the pinned HF `ihm` test
  split. Overlapping or touching segments are merged into blocks. Windows cut only at block boundaries, with
  0.25 s padding, and never overlap. `speaker_id` is the comma-joined list of participants. `ref_text` joins
  the segment transcripts in begin-time order, in the Kaldi-normalized upper-case form the HF data uses.
- The text-overlap filter was re-run with the same 12-gram / exact-match rule as section 6. The eval pool for
  this run also includes VoxPopuli `en_accented` test and the FLEURS raw transcripts. The full MLS and People's
  Speech candidate pools were checked: 15 + 1 hits, all dropped. The other corpora found no new hits beyond
  the original lists.

### 9.4 Sanity (instance, `manifest_and_sanity.py`, all passed)

- Manifest rows equal shard rows: train 175,000 (90 shards), eval 12,108 (9 shards). Every shard has at most
  2,000 rows and is zstd.
- 0 duplicate `clip_id`s across train and eval.
- Eval speakers present in train, per speaker-id family: LibriVox readers (LibriSpeech + LibriTTS-R + MLS)
  0 of 73; CV 0 of 1,939; VoxPopuli 0 of 285; AMI 0 of 16 (long-form participants included).
- LibriSpeech vs LibriTTS-R train chapter overlap: 0.
- 5 random clips per dataset/split, 18 groups: the PCM length / 16 kHz matches `duration_s` within 1 ms,
  and no clip is silent. Train durations lie in [1.0, 30.0] s.
- Every row's license is in the approved set.
- Size: train 43.62 GB, eval 2.32 GB (45.95 GB of shards). S3 `corpus/` holds 138 objects, 45.95 GB.

## 10. Long-form train corpus (long-form adaptation)

Built 2026-10-09 with `data/prep/build_longform.py` and checked with `data/prep/longform_sanity.py`. The build
ran CPU-only on the project trn2 instance.

**Provenance:** every long clip is a concatenation of clips that are **already in the train shards** (section 9).
No new data was added, and nothing comes from an eval split. That means every earlier filter still holds: the
VoxPopuli eval-speaker drop list, the AMI IB4005 drop, the People's Speech per-source license allow-list,
the transcript-overlap drops, and the 1–30 s per-clip bounds. Each long clip draws from one dataset and one
license. Its `source_path` lists the source `clip_id`s (separated by `;`), so attribution and license trace
back per source clip. Each source clip is used at most once.

**Construction:**
- Source clips are joined with 0.2–0.5 s of digital silence between them (seed 0).
- Ordering rules:
  - AMI: same train meeting, in begin-time order, never crossing meetings.
  - LibriSpeech, LibriTTS-R, MLS: same reader, in chapter/book then utterance order.
  - VoxPopuli: same speaker, in session/time order.
  - People's Speech: same archive.org source, in segment order, never crossing a license change.
  - CV17: random clips (the hashed speakers have about one clip each).
- A sample stays within its group (speaker, meeting or source) until it reaches its target window count.
- `ref_text` joins the source references with a single space, keeping native casing.
- `split` is `train_longform`, and `n_windows` = ceil(duration_s / 30).
- Shards hold at most 500 rows, use zstd, and are sorted by `n_windows`.

| Dataset | Long clips | Hours | Source clips used | Single-speaker share |
|---|---|---|---|---|
| AMI (ihm segments, same meeting) | 678 | 25.86 | 19,829 | 2 % |
| People's Speech | 940 | 39.88 | 10,200 | - (no speaker ids) |
| VoxPopuli | 947 | 40.60 | 14,240 | 85 % |
| LibriSpeech | 564 | 23.77 | 6,801 | 27 % |
| MLS | 499 | 21.63 | 5,123 | 8 % |
| LibriTTS-R | 259 | 10.90 | 5,810 | 8 % |
| CV17 | 414 | 16.15 | 9,340 | 0 % |
| **Total** | **4,301** | **178.79** | **71,343** | |

Window histogram: W=2 584, W=3 588, W=4 583, W=5 578, W=6 563, W=7 169, W=8 562, W=9 147, W=10 527.
W=7 and W=9 were not targeted; they come from samples whose length landed in those ranges. Durations run
from 31.0 to 299.3 s on the decoded audio. Licenses by long clip: CC-BY-4.0 2,000; CC0-1.0 1,365; US Government
Works 604; CC-BY-3.0 165; Public Domain 158; CC-BY-2.0 9.

**AMI share:** the AMI train shards hold 20,000 segments (24.2 h), enough for 678 long clips (15.8 %) instead
of the 30 % target. The shortfall went to the other conversational sets: People's Speech 21.9 % and VoxPopuli
22.0 %. The read-speech family (LibriSpeech + LibriTTS-R + MLS) is 30.7 % and CV17 is 9.6 %. Many AMI
meetings have gaps, because the base draw sampled about 37 % of the ≥1.5 s segments. Inside an AMI
long clip, consecutive segments are therefore in time order but not always adjacent in the meeting.

**Sanity (all passed, 0 errors):**
- No source clip is reused.
- Every source clip is in `train_manifest.parquet` and none is in `eval_manifest.parquet`.
- No source speaker is an eval speaker in any family (LibriVox, CV, VoxPopuli, AMI).
- Decoded length equals `duration_s` and lies within [31, 300] s.
- `n_windows` is correct for every row.
- Every shard has at most 500 rows and is sorted by `n_windows`.

**Locations:**
- Shards: `s3://<bucket>/corpus/shards/train_longform/<short>/shard-NNNNN.parquet` (16.1 GB).
- Manifest: `s3://<bucket>/corpus/shards/train_longform_manifest.parquet`, local copy
  `data/corpus_manifest/train_longform_manifest.parquet`.
- Stats: `corpus/stats/train_longform_{build,sanity}.json`.
