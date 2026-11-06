"""Common Voice 17 English: speaker (client_id) and sentence overlap between train/dev/test.

Inputs: transcript/en/{train,dev,test}.tsv from fsicoli/common_voice_17_0 @ 8262c16b (metadata only).
Usage: python cv_split_check.py ../../tmp/cv
"""
import csv
import json
import sys

d = sys.argv[1]
csv.field_size_limit(1 << 30)
S = {}
for s in ("train", "dev", "test"):
    rows = list(csv.DictReader(open(f"{d}/{s}.tsv", newline="", encoding="utf-8"), delimiter="\t",
                               quoting=csv.QUOTE_NONE))
    S[s] = {"rows": len(rows), "clients": {r["client_id"] for r in rows},
            "sentences": {r["sentence_id"] for r in rows}, "paths": {r["path"] for r in rows}}
out = {s: {"rows": v["rows"], "speakers": len(v["clients"]), "sentences": len(v["sentences"])} for s, v in S.items()}
for e in ("dev", "test"):
    out[f"train_vs_{e}"] = {k: len(S["train"][k] & S[e][k]) for k in ("clients", "sentences", "paths")}
out["dev_vs_test"] = {k: len(S["dev"][k] & S["test"][k]) for k in ("clients", "sentences", "paths")}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{d}/cv_split_check.json", "w"), indent=1)
