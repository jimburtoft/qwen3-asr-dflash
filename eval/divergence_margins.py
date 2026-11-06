# SPDX-License-Identifier: Apache-2.0
"""Where served greedy outputs with and without the draft differ, how close was the target's decision?

For each clip whose two transcripts differ: tokenize both transcripts as the target's response
(``language X<asr_text>`` + text), find the first differing token, run the HF target (teacher forcing,
fp32 and bf16) on the common prefix and report the logit margin between the two candidate tokens and
the top-2 margin at that position. A small margin (bf16-scale) means a numerical near-tie, a large
margin would point at a real verify-path problem.

    python eval/divergence_margins.py --plain plain.json --spec spec.json --shards "<eval shard glob>"
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.qwen3_asr_target import ASR_TEXT_ID, load_hf_qwen3_asr, load_processor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    P = {c["clip_id"]: c for c in json.load(open(a.plain))["clips"]}
    S = {c["clip_id"]: c for c in json.load(open(a.spec))["clips"]}
    diff = [k for k in P if k in S and P[k]["text"] != S[k]["text"]]
    audio = {}
    for f in glob.glob(a.shards):
        t = pq.read_table(f, columns=["clip_id", "audio_pcm16"])
        for cid, pcm in zip(t.column("clip_id").to_pylist(), t.column("audio_pcm16").to_pylist()):
            if cid in diff:
                audio[cid] = pcm
    proc = load_processor()
    tok = proc.tokenizer
    models = {dt: load_hf_qwen3_asr(getattr(torch, dt)) for dt in ("float32", "bfloat16")}
    rows = []
    for cid in diff:
        x = np.frombuffer(audio[cid], dtype="<i2").astype(np.float32) / 32768.0
        inp = proc.apply_transcription_request(audio=x, sampling_rate=16000)
        prompt = inp["input_ids"][0].tolist()
        res = {"clip_id": cid}
        with torch.no_grad():
            out = models["float32"].generate(**inp, max_new_tokens=256, do_sample=False)
        gen = out[0, len(prompt):].tolist()
        lang_head = gen[: gen.index(ASR_TEXT_ID) + 1] if ASR_TEXT_ID in gen else []
        tp = tok.encode(P[cid]["text"], add_special_tokens=False)
        ts = tok.encode(S[cid]["text"], add_special_tokens=False)
        k = next((i for i in range(min(len(tp), len(ts))) if tp[i] != ts[i]), min(len(tp), len(ts)))
        if k >= min(len(tp), len(ts)):
            res["note"] = "texts differ only in length/whitespace after tokenization"
            rows.append(res)
            continue
        prefix = prompt + lang_head + tp[:k]
        res.update({"pos": k, "tok_plain": tok.decode([tp[k]]), "tok_spec": tok.decode([ts[k]]),
                    "hf_fp32_greedy_text": tok.decode([t for t in gen[len(lang_head):] if t < 151643])[:200]})
        for dt, m in models.items():
            ids = torch.tensor([prefix])
            fx = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                  "input_features": inp["input_features"].to(getattr(torch, dt)),
                  "input_features_mask": inp["input_features_mask"]}
            with torch.no_grad():
                lg = m(**fx).logits[0, -1].float()
            top = torch.topk(lg, 2)
            res[dt] = {"logit_plain_tok": round(float(lg[tp[k]]), 3), "logit_spec_tok": round(float(lg[ts[k]]), 3),
                       "margin_plain_minus_spec": round(float(lg[tp[k]] - lg[ts[k]]), 3),
                       "top1": tok.decode([int(top.indices[0])]), "top2_margin": round(float(top.values[0] - top.values[1]), 3)}
        res["hf_fp32_matches"] = ("plain" if res["float32"]["margin_plain_minus_spec"] > 0 else "spec")
        print(json.dumps(res, ensure_ascii=False), flush=True)
        rows.append(res)
    m32 = [abs(r["float32"]["margin_plain_minus_spec"]) for r in rows if "float32" in r]
    summ = {"n_diff": len(diff), "analysed": len(m32), "abs_margin_fp32_median": float(np.median(m32)) if m32 else None,
            "abs_margin_fp32_max": max(m32) if m32 else None,
            "hf_fp32_agrees_with_plain": sum(r.get("hf_fp32_matches") == "plain" for r in rows),
            "hf_fp32_agrees_with_spec": sum(r.get("hf_fp32_matches") == "spec" for r in rows)}
    print("SUMMARY " + json.dumps(summ))
    if a.out:
        json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
