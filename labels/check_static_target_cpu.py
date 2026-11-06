# SPDX-License-Identifier: Apache-2.0
"""CPU check: static Qwen3ASRTarget == HF Qwen3ASRForConditionalGeneration.

Per clip: prompt ids (ours vs the HF processor), audio embeddings, target-layer features and
logits (teacher-forced over prompt + HF greedy transcript), and our static-cache greedy decode vs
HF ``generate``. Also writes the records as a label file for ``train/train_cpu_ref.py``
(``--out``), with features from the HF model (the reference).

    python labels/check_static_target_cpu.py --parquet <librispeech parquet> --n 4 [--out x.pt]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import TARGET_ID, TARGET_REVISION, DraftConfig  # noqa: E402
from model.qwen3_asr_target import (  # noqa: E402
    EOS_IDS, Qwen3ASRTarget, audio_token_count, build_prompt_ids, load_hf_qwen3_asr, load_processor,
)


def iter_parquet(path, n, max_dur=35.0, min_dur=1.0, skip=0):
    import pyarrow.parquet as pq
    rows = pq.read_table(path, columns=["id", "audio", "text"]).to_pylist()
    k = 0
    for r in rows[skip:]:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if not (min_dur <= len(a) / sr <= max_dur):
            continue
        yield r["id"], a, sr, r["text"]
        k += 1
        if k >= n:
            return


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--language", default=None, help="force language (default: auto, as served)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default="openslr/librispeech_asr")
    ap.add_argument("--split", default="train.clean.100")
    ap.add_argument("--license", default="CC-BY-4.0")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dt = getattr(torch, args.dtype)
    proc = load_processor()
    hf = load_hf_qwen3_asr(dt)
    tgt = Qwen3ASRTarget(hf).eval()
    cfg = DraftConfig()
    L = cfg.target_layer_ids
    recs, summary = [], []
    for cid, a, sr, ref in iter_parquet(args.parquet, args.n, skip=args.skip):
        t0 = time.time()
        x = proc.apply_transcription_request(audio=a, language=args.language, sampling_rate=sr)
        x = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in x.items()}
        ids_hf = x["input_ids"][0]
        n_frames = int(x["input_features_mask"][0].sum())
        n_aud = audio_token_count(n_frames)
        my_ids = build_prompt_ids(proc.tokenizer, n_aud, language=args.language)
        prompt_ok = my_ids == ids_hf.tolist()
        P = ids_hf.numel()
        # HF greedy
        out = hf.generate(**x, max_new_tokens=args.max_new, do_sample=False)
        resp = out[0, P:]
        hit = [i for i, t in enumerate(resp.tolist()) if t in EOS_IDS]
        if hit:
            resp = resp[: hit[0] + 1]
        full = torch.cat([ids_hf, resp])[None]
        # HF teacher-forced reference features
        fx = dict(x)
        fx["input_ids"] = full
        fx["attention_mask"] = torch.ones_like(full)
        o = hf(**fx, output_hidden_states=True, use_cache=False)
        ref_feats = torch.cat([o.hidden_states[l + 1][0] for l in L], -1).float()
        ref_logits = o.logits[0].float()
        ref_audio = hf.model.get_audio_features(x["input_features"], x["input_features_mask"]).pooler_output.float()
        # ours
        mel = x["input_features"][0]
        mw, nv = Qwen3ASRTarget.windows_from_mel(mel, n_frames)
        enc = tgt.encode_windows(mw, nv)
        aud = Qwen3ASRTarget.gather_audio(enc, nv.tolist())
        a_rel = ((aud.float() - ref_audio).norm() / ref_audio.norm()).item()
        span = tgt.audio_span(full[0])
        emb = tgt.build_embeds(full, aud[None], span)
        feats, logits = tgt.forward_full(emb, L, want_logits=True)
        f_rel = ((feats[0].float() - ref_feats).norm() / ref_feats.norm()).item()
        l_rel = ((logits[0].float() - ref_logits).norm() / ref_logits.norm()).item()
        tf = (logits[0, P - 1:-1].argmax(-1) == full[0, P:]).float().mean().item()
        # our static-cache greedy decode
        S = 1 << (P + args.max_new + 8 - 1).bit_length()
        tgt.alloc_cache(1, S, dt, "cpu")
        pemb = emb[:, :P]
        am = tgt.prefill_chunk(pemb, torch.arange(P))
        tok = am[:, -1]
        mine = [int(tok)]
        pos = P
        while mine[-1] not in EOS_IDS and len(mine) < args.max_new:
            tok = tgt.decode(tok, torch.tensor([pos]))
            mine.append(int(tok))
            pos += 1
        g_ok = mine == resp.tolist()
        hyp = proc.tokenizer.decode(resp.tolist(), skip_special_tokens=True)
        row = dict(clip_id=cid, dur=round(len(a) / sr, 2), windows=int(nv.numel()), P=P, R=int(resp.numel()),
                   prompt_ids_match=prompt_ok, audio_rel=a_rel, feat_rel=f_rel, logit_rel=l_rel,
                   tf_argmax_match=tf, static_greedy_identical=g_ok, sec=round(time.time() - t0, 1), hyp=hyp[:100])
        print(json.dumps(row), flush=True)
        summary.append(row)
        recs.append(dict(clip_id=cid, dataset=args.dataset, split=args.split, license=args.license,
                         duration_s=len(a) / sr, ids=full[0].to(torch.int32), prompt_len=P,
                         feats=ref_feats.to(torch.bfloat16), ref_text=ref, hyp_text=hyp))
    if args.out:
        torch.save(dict(records=recs, target_layer_ids=L, target=TARGET_ID, target_revision=TARGET_REVISION),
                   args.out)
        Path(args.out).with_suffix(".check.json").write_text(json.dumps(summary, indent=1))
    ok = all(r["prompt_ids_match"] and r["static_greedy_identical"] for r in summary)
    print("ALL_OK" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
