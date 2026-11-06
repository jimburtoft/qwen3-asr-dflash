# SPDX-License-Identifier: Apache-2.0
"""Upstream vLLM: Qwen3-ASR-1.7B target + this DFlash draft on REAL audio (CPU container).

Runs N held-out clips through upstream vLLM twice (plain greedy, then DFlash greedy speculative
decoding with K = 15) and reports: token-identical outputs (losslessness), spec-decode acceptance
(mean accepted tokens per draft step + 1 from vLLM's spec-decode metrics) and the draft config vLLM
resolved. Audio comes from the corpus eval shards (16 kHz PCM16).

Run inside vllm/vllm-openai-cpu with this repo at /work and the corpus at /corpus:
    python3 /work/eval/vllm_upstream_asr_check.py --draft /ckpt/release --shard /corpus/eval/.../shard-00000.parquet --n 8
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "16")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def patch_knorm():
    """vLLM CPU platform: the grouped ops.rms_norm in DFlashQwen3Model._normalize_context_k applies only
    layer 0's k_norm weight to all layers (the stacked [L, hd] weight is not indexed per layer on CPU).
    Replace it with a per-layer loop (what the XPU branch did in earlier versions). Lossless either way."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.models import qwen3_dflash as qd

    def per_layer(self, all_k):
        out = torch.empty_like(all_k)
        for i in range(all_k.shape[0]):
            ops.rms_norm(out[i], all_k[i].contiguous(), self._k_norm_weights[i].contiguous(), self._rms_norm_eps)
        return out
    for n in dir(qd):
        c = getattr(qd, n)
        if isinstance(c, type) and hasattr(c, "_normalize_context_k"):
            c._normalize_context_k = per_layer


def counters(llm):
    out = {}
    try:
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                v = getattr(m, "value", None)
                if v is None and hasattr(m, "values"):
                    v = list(m.values)
                out[m.name] = v
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--revision", default="7278e1e70fe206f11671096ffdd38061171dd6e5")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--shard", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--patch-knorm", type=int, default=0)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    if a.patch_knorm:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # patch must reach the engine process
        patch_knorm()

    clips = []
    for sh in a.shard:
        t = pq.read_table(sh, columns=["clip_id", "duration_s", "audio_pcm16"]).to_pylist()
        clips += [r for r in t if r["duration_s"] <= a.max_dur][: a.n]
    prompt = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
              "<|im_end|>\n<|im_start|>assistant\n")
    reqs = [{"prompt": prompt, "multi_modal_data": {"audio": (
        np.frombuffer(c["audio_pcm16"], dtype="<i2").astype(np.float32) / 32768.0, 16000)}} for c in clips]
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    res = {"clips": [c["clip_id"] for c in clips]}
    for spec in (False, True):
        kw = dict(model=a.target, revision=a.revision, dtype="bfloat16", max_model_len=2048, enforce_eager=True,
                  seed=0, limit_mm_per_prompt={"audio": 1}, disable_log_stats=False, max_num_seqs=1)
        if spec:
            kw["speculative_config"] = {"method": "dflash", "model": a.draft, "num_speculative_tokens": a.k}
        llm = LLM(**kw)
        outs = llm.generate(reqs, sp)
        key = "spec" if spec else "plain"
        res[key] = [list(o.outputs[0].token_ids) for o in outs]
        res[key + "_text"] = [o.outputs[0].text for o in outs]
        if spec:
            res["metrics"] = counters(llm)
            dm = llm.llm_engine.vllm_config.speculative_config.draft_model_config.hf_config
            res["draft_cfg"] = {"dflash_config": getattr(dm, "dflash_config", None),
                                "block_size": getattr(dm, "block_size", None),
                                "rope_parameters": getattr(dm, "rope_parameters", None)}
        del llm
    res["identical"] = [p == s for p, s in zip(res["plain"], res["spec"])]
    m = res.get("metrics", {})
    acc = m.get("vllm:spec_decode_num_accepted_tokens") or m.get("vllm:spec_decode_num_accepted_tokens_total")
    drafts = m.get("vllm:spec_decode_num_drafts") or m.get("vllm:spec_decode_num_drafts_total")
    if acc and drafts:
        res["mean_accepted_plus_bonus"] = 1 + float(acc) / float(drafts)
    print("RESULT", json.dumps(res, default=str))


if __name__ == "__main__":
    main()
