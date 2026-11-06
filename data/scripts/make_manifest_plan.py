"""Emit data/manifest_plan.json: the machine-readable data plan (train mix + eval sets).

Numbers come from our own metadata scans (see LICENSES.md "Provenance of numbers"):
row counts from the HF dataset-viewer /size API at the pinned revision, hours from the
scripts in data/scripts/ or the dataset cards/papers where noted.

Usage: python make_manifest_plan.py [--ps-table ../../tmp/ps/ps_clean_source_licenses.csv] > ../manifest_plan.json
"""
import argparse
import csv
import json

REV = {
    "openslr/librispeech_asr": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "mythicinfinity/libritts_r": "0d1718db351207d5a493135a62d5ed8cc4733830",
    "parler-tts/mls_eng": "faf6604dcf0bb9ec0d9c280a140ad74da9c931b0",
    "fsicoli/common_voice_17_0": "8262c16bf297c87a9cd88c51997c4758ed7a8ba2",
    "facebook/voxpopuli": "42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
    "MLCommons/peoples_speech": "f10597c5d3d3a63f8b6827701297c3afdf178272",
    "edinburghcstr/ami": "46f28f2503e2ec48f8867a84eef356c70476beab",
    "google/fleurs": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
    "facebook/multilingual_librispeech": "2e83e61823b4c47dcbcb1980bb88601274127609",
    "Qwen/Qwen3-ASR-1.7B": "7278e1e70fe206f11671096ffdd38061171dd6e5",
}
LIC = {
    "openslr/librispeech_asr": "CC-BY-4.0",
    "mythicinfinity/libritts_r": "CC-BY-4.0",
    "parler-tts/mls_eng": "CC-BY-4.0",
    "fsicoli/common_voice_17_0": "CC0-1.0",
    "facebook/voxpopuli": "CC0-1.0",
    "MLCommons/peoples_speech": "CC-BY-4.0 (dataset); per-source CC-BY-2.0/2.5/3.0/4.0, public domain, US-government works",
    "edinburghcstr/ami": "CC-BY-4.0",
    "google/fleurs": "CC-BY-4.0",
}

FILTERS = {
    "common": [
        "duration_s <= 30 (headline set)",
        "drop clips listed in data/filters/text_overlap_flagged_<dataset>.txt (transcript shares a 12-word n-gram or an exact >=8-word transcript with any eval set); re-run text_overlap.py over the full MLS / People's Speech train before corpus sampling",
    ],
    "facebook/voxpopuli": ["drop train clips whose speaker_id is in data/filters/voxpopuli_drop_speakers.json (real speakers of en validation/test and en_accented test; 'None' is not a speaker id)"],
    "edinburghcstr/ami": ["drop train meeting IB4005 (data/filters/ami_drop.json)", "duration_s >= 1.5"],
    "MLCommons/peoples_speech": ["keep only clips whose source identifier has allowed=1 in data/ps_clean_source_licenses.csv"],
    "parler-tts/mls_eng": ["no speaker filter needed (MLS en dev/test are disjoint from LibriSpeech per the MLS paper and our id check)"],
}


def ps_allowed(path):
    if not path:
        return None
    c = h = 0
    for r in csv.DictReader(open(path)):
        if r["allowed"] == "1":
            c += int(r["clips"])
            h += float(r["hours"])
    return c, round(h, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps-table", default=None)
    a = ap.parse_args()
    ps = ps_allowed(a.ps_table)

    train = [
        # dataset, config, split, available_clips, available_hours, planned_clips, planned_hours, domain
        ("openslr/librispeech_asr", "all", "train.clean.100+train.clean.360+train.other.500", 281140, 961.1, 25000, 85, "read (audiobook)"),
        ("mythicinfinity/libritts_r", "all", "train.clean.100+train.clean.360+train.other.500", 354623, 555.2, 15000, 23, "read (audiobook, restored 24 kHz); sample only chapters not sampled from LibriSpeech"),
        ("parler-tts/mls_eng", "default", "train", 10808037, 44659.7, 20000, 83, "read (audiobook, many readers)"),
        ("fsicoli/common_voice_17_0", "en", "train", 1101087, 1738.5, 40000, 63, "crowd-sourced read, many accents"),
        ("facebook/voxpopuli", "en", "train", 135189, 387.0, 30000, 86, "parliamentary, semi-spontaneous, L2 accents"),
        ("MLCommons/peoples_speech", "clean", "train", ps[0] if ps else None, ps[1] if ps else None, 25000, 98, "conversational / public meetings / talks"),
        ("edinburghcstr/ami", "ihm", "train", 56426, 67.7, 20000, 24, "meetings, spontaneous, mostly non-native"),
    ]
    evals = [
        ("openslr/librispeech_asr", "all", "test.clean", 2620, 5.4),
        ("openslr/librispeech_asr", "all", "test.other", 2939, 5.3),
        ("fsicoli/common_voice_17_0", "en", "test", 16393, 27.0),
        ("facebook/voxpopuli", "en", "test", 1842, 4.9),
        ("google/fleurs", "en_us", "test", 647, 1.77),
        ("edinburghcstr/ami", "ihm", "test (full-corpus-ASR SC: EN2002, ES2004, IS1009, TS3003; 16 meetings) -- short segments + long-form windows rebuilt from the official Mix-Headset recordings", 12643, 8.7),
    ]
    out = {
        "revisions": REV,
        "target_model": {"id": "Qwen/Qwen3-ASR-1.7B", "revision": REV["Qwen/Qwen3-ASR-1.7B"], "license": "Apache-2.0"},
        "train": [
            {"dataset": d, "revision": REV[d], "config": c, "split": s, "role": "train", "license": LIC[d],
             "available_clips_after_filters": ac, "available_hours_after_filters": ah,
             "planned_clips": pc, "planned_hours_approx": ph, "domain": dom,
             "filters": FILTERS["common"] + FILTERS.get(d, [])}
            for d, c, s, ac, ah, pc, ph, dom in train
        ],
        "eval": [
            {"dataset": d, "revision": REV[d], "config": c, "split": s, "role": "eval", "license": LIC[d],
             "clips": n, "hours": h, "never_used_in_training": True}
            for d, c, s, n, h in evals
        ],
        "excluded": [
            {"dataset": "MLCommons/peoples_speech", "config": "clean_sa / dirty / dirty_sa", "reason": "share-alike (clean_sa, dirty_sa) or not in approved list (dirty)"},
            {"dataset": "google/fleurs", "config": "en_us", "split": "train / validation", "reason": "FLEURS is eval-only in this project"},
            {"dataset": "facebook/voxpopuli", "config": "en_accented", "split": "test", "reason": "held out; optional extra eval set"},
            {"dataset": "openslr/librispeech_asr", "split": "validation.clean / validation.other", "reason": "held out (dev); usable for pilot acceptance checks, never trained on"},
        ],
    }
    out["totals"] = {"planned_train_clips": sum(t[5] for t in train), "planned_train_hours_approx": sum(t[6] for t in train)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
