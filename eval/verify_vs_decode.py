# SPDX-License-Identifier: Apache-2.0
"""Is the verify step's argmax the same as plain decode's argmax for the same prefix? (static target, trn2)

For one clip and its no-draft greedy response r: run the static Qwen3-ASR target (bf16) on device in two ways
over the same cache contents --
  * decode: 1 token per step (prefix lengths P .. P+n-1)
  * verify: one 16-token chunk over positions [t, t+16) (prefix + 15 next tokens), like a DFlash verify
    step with a draft that guessed right
and compare argmax / top-2 margins at each position. Shows whether multi-token verify graphs alone change
the greedy decision (independent of vllm-neuron's runner) and by how much.

    python eval/verify_vs_decode.py --clip-shard <parquet> --clip-id <id> [--device neuron]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.device import add_device_arg, get_device, maybe_compile  # noqa: E402
from model.qwen3_asr_target import (  # noqa: E402
    EOS_IDS, Qwen3ASRTarget, audio_token_count, build_prompt_ids, load_hf_qwen3_asr, load_processor,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-shard", required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--block", type=int, default=16)
    add_device_arg(ap)
    a = ap.parse_args()
    dev = get_device(a.device)
    dt = torch.bfloat16
    t = pq.read_table(a.clip_shard).to_pylist()
    row = next(r for r in t if r["clip_id"] == a.clip_id)
    proc = load_processor()
    x = np.frombuffer(row["audio_pcm16"], dtype="<i2").astype(np.float32) / 32768.0
    fe = proc.feature_extractor(x, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    nf = int(fe["attention_mask"][0].sum())
    tgt = Qwen3ASRTarget(load_hf_qwen3_asr(dt), max_pos=1100).eval().to(dev)
    mw, nv = Qwen3ASRTarget.windows_from_mel(fe["input_features"][0], nf)
    with torch.no_grad():
        aud = Qwen3ASRTarget.gather_audio(tgt.encode_windows(mw.to(dt).to(dev), nv.to(dev)).cpu(), nv.tolist())
    pids = build_prompt_ids(proc.tokenizer, audio_token_count(nf))
    P, C, S = len(pids), 128, 1024

    def logits_chunk(x, pos):
        cos = F.embedding(pos, tgt.cos).to(x.dtype)[None]
        sin = F.embedding(pos, tgt.sin).to(x.dtype)[None]
        ar = torch.arange(tgt.S_cache, device=x.device)
        mask = (ar[None, :] <= pos[:, None])[None, None]
        h = x
        for i, layer in enumerate(tgt.layers):
            q, k, v = layer._qkv(h, cos, sin)
            tgt.kc[i].index_copy_(2, pos, k)
            tgt.vc[i].index_copy_(2, pos, v)
            h = layer._mlp(layer._attn_out(h, q, tgt.kc[i], tgt.vc[i], mask))
        return F.linear(torch.rms_norm(h, (tgt.H,), tgt.norm, tgt.eps), tgt.lm_head)[0].float()

    chunk = maybe_compile(logits_chunk, dev)
    with torch.no_grad():
        # 1) greedy with 1-token chunks (same graph as decode at C=1 would be; here: prefill C, then 1-token steps)
        ids = torch.tensor(pids)
        emb_p = tgt.build_embeds(ids[None].to(dev), aud[None].to(dev), Qwen3ASRTarget.audio_span(ids))[0]
        tgt.alloc_cache(1, S, dt, dev)
        n_ch = (P + C - 1) // C
        embp = torch.cat([emb_p, emb_p.new_zeros(n_ch * C - P, tgt.H)])
        for c in range(n_ch):
            lg = chunk(embp[None, c * C:(c + 1) * C], torch.arange(c * C, (c + 1) * C, device=dev))
        nxt = int(lg[(P - 1) - (n_ch - 1) * C].argmax())
        resp, dec_lg = [nxt], []
        one = maybe_compile(lambda e, p: logits_chunk(e, p), dev)
        while resp[-1] not in EOS_IDS and len(resp) < 200:
            p = P + len(resp) - 1
            lg1 = one(F.embedding(torch.tensor([[resp[-1]]], device=dev), tgt.embed), torch.tensor([p], device=dev))[0]
            dec_lg.append(lg1.cpu())
            resp.append(int(lg1.argmax()))
        # 2) verify chunks: cache = prompt; then blocks [anchor + next 15 tokens of resp] -> argmax at each slot
        tgt.alloc_cache(1, S, dt, dev)
        for c in range(n_ch):
            chunk(embp[None, c * C:(c + 1) * C], torch.arange(c * C, (c + 1) * C, device=dev))
        B = a.block
        out = []
        for s0 in range(0, len(resp) - 1, B):
            toks = resp[s0:s0 + B]
            n = len(toks)
            toks = toks + [EOS_IDS[1]] * (B - n)
            pos = torch.arange(P + s0, P + s0 + B, device=dev)
            lgv = chunk(F.embedding(torch.tensor([toks], device=dev), tgt.embed), pos).cpu()
            for j in range(n):
                k = s0 + j          # predicts resp[k + 1]
                if k >= len(dec_lg):
                    break
                ld, lv = dec_lg[k], lgv[j]
                top_d = torch.topk(ld, 2)
                out.append({"k": k, "decode_argmax": int(ld.argmax()), "verify_argmax": int(lv.argmax()),
                            "decode_top2_margin": round(float(top_d.values[0] - top_d.values[1]), 3),
                            "max_abs_logit_diff": round(float((ld - lv).abs().max()), 3)})
    flips = [o for o in out if o["decode_argmax"] != o["verify_argmax"]]
    print(json.dumps({"clip": a.clip_id, "positions": len(out), "argmax_flips": len(flips), "flips": flips[:10],
                      "median_max_abs_logit_diff": float(np.median([o["max_abs_logit_diff"] for o in out])),
                      "text": proc.tokenizer.decode([t for t in resp if t not in EOS_IDS])[:120]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
