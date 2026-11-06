# SPDX-License-Identifier: Apache-2.0
"""CPU check: batched ragged-prompt greedy (prefill_rows + decode_rows) == HF greedy per clip.

Uses a label file written by check_static_target_cpu.py (ids = prompt + HF greedy response) and
the same LibriSpeech parquet for the audio.

    python labels/check_batched_cpu.py --labels work/ls_test16.pt --parquet <test.clean parquet> --batch 8
"""

import argparse
import io
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.qwen3_asr_target import EOS_IDS, Qwen3ASRTarget, load_hf_qwen3_asr, load_processor  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--mode", default="left", choices=["left", "right"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dt = getattr(torch, a.dtype)
    import pyarrow.parquet as pq
    rows = {r["id"]: r for r in pq.read_table(a.parquet, columns=["id", "audio"]).to_pylist()}
    recs = torch.load(a.labels, weights_only=False)["records"]
    proc = load_processor()
    tgt = Qwen3ASRTarget(load_hf_qwen3_asr(dt)).eval()
    res = []
    for b0 in range(0, len(recs), a.batch):
        chunk = recs[b0:b0 + a.batch]
        embs, P, want = [], [], []
        for r in chunk:
            x, sr = sf.read(io.BytesIO(rows[r["clip_id"]]["audio"]["bytes"]), dtype="float32")
            fx = proc.feature_extractor(x, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
            nf = int(fx["attention_mask"][0].sum())
            mw, nv = Qwen3ASRTarget.windows_from_mel(fx["input_features"][0].to(dt), nf)
            aud = Qwen3ASRTarget.gather_audio(tgt.encode_windows(mw, nv), nv.tolist())
            ids = r["ids"].long()
            p = int(r["prompt_len"])
            e = tgt.build_embeds(ids[None, :p], aud[None], tgt.audio_span(ids[:p]))[0]
            embs.append(e); P.append(p); want.append(ids[p:].tolist())
        B, Pm = len(embs), max(P)
        max_new = max(len(w) for w in want) + 2
        tgt.alloc_cache(B, Pm + max_new + 1, dt, "cpu")
        if a.mode == "right":
            x = torch.stack([F.pad(e, (0, 0, 0, Pm - e.shape[0])) for e in embs])
            tok = tgt.prefill_rows(x, torch.tensor(P) - 1)
            pos = torch.tensor(P)
            step = lambda t, p: tgt.decode_rows(t, p)  # noqa: E731
        else:
            x = torch.stack([F.pad(e, (0, 0, Pm - e.shape[0], 0)) for e in embs])
            pad = Pm - torch.tensor(P)
            tok = tgt.prefill_left(x, pad)
            pos = torch.tensor([Pm])
            step = lambda t, p: tgt.decode_left(t, p, pad)  # noqa: E731
        out = [[int(t)] for t in tok]
        for _ in range(max_new - 1):
            tok = step(tok, pos)
            pos = pos + 1
            for j in range(B):
                out[j].append(int(tok[j]))
        for j, r in enumerate(chunk):
            s = out[j]
            k = next((i for i, t in enumerate(s) if t in EOS_IDS), len(s) - 1)
            ok = s[:k + 1] == want[j]
            res.append(ok)
            print(json.dumps({"clip_id": r["clip_id"], "P": P[j], "R": len(want[j]), "identical": ok}), flush=True)
    print("BATCHED_ALL_OK" if all(res) else f"BATCHED_MISMATCH {res.count(False)}/{len(res)}")


if __name__ == "__main__":
    main()
