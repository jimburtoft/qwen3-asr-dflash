# SPDX-License-Identifier: Apache-2.0
"""Load a release-format draft in upstream vLLM's DFlash path (CPU container) and run it.

FORMAT / loader check, not an acceptance measurement (text prompt, no audio).

Target = Qwen3-ASR's 28-layer text decoder exported as Qwen3ForCausalLM
(tools/extract_text_decoder.py): this checks the draft against the exact target decoder whether
or not the multimodal ``Qwen3ASRForConditionalGeneration`` wrapper exposes aux hidden states.
Checks: config parsing, strict weight loading, the resolved target layers / mask id / block size /
draft RoPE base, and greedy losslessness (spec output == plain output).

Run inside vllm/vllm-openai-cpu with this repo mounted at /work::

    python3 /work/eval/vllm_upstream_load_check.py <target_dir> <draft_dir> [K]
"""

import json
import os
import sys

TARGET = sys.argv[1]
DRAFT = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 15

os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "8")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams  # noqa: E402

PROMPT = "<|im_start|>user\nWrite one sentence about the city of Paris.<|im_end|>\n<|im_start|>assistant\n"
REC: dict = {}


def _hooks():
    """Record what vLLM's DFlash draft actually resolved (rope base, loaded / missing tensors)."""
    import torch
    from vllm.model_executor.models import qwen3_dflash as qd

    cls = [getattr(qd, n) for n in dir(qd) if n.startswith("DFlash") and hasattr(getattr(qd, n), "load_weights")]
    for c in cls:
        orig = c.load_weights

        def lw(self, weights, _orig=orig, _name=c.__name__):
            names = []

            def tap(it):
                for n, w in it:
                    names.append(n)
                    yield n, w
            r = _orig(self, tap(weights))
            params = dict(self.named_parameters())
            REC.setdefault("load", {})[_name] = {
                "ckpt_tensors": len(names), "ckpt_names_sample": sorted(names)[:6],
                "n_params": len(params),
                "param_names_sample": sorted(params)[:6],
                "zero_params": sorted(k for k, v in params.items() if v.dtype.is_floating_point
                                      and v.numel() > 0 and float(v.abs().max()) == 0.0)[:20],
            }
            for n, m in self.named_modules():
                if hasattr(m, "base") and hasattr(m, "rotary_dim") and "rope" not in REC:
                    REC["rope"] = {"module": n, "base": float(m.base), "rotary_dim": int(m.rotary_dim)}
            return r
        c.load_weights = lw


def run(spec):
    kw = dict(model=TARGET, dtype="bfloat16", max_model_len=1024, enforce_eager=True, seed=0)
    if spec:
        kw["speculative_config"] = {"method": "dflash", "model": DRAFT, "num_speculative_tokens": K}
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    outs = llm.generate([PROMPT], sp)
    res = [list(o.outputs[0].token_ids) for o in outs]
    info = {}
    if spec:
        try:
            dm = llm.llm_engine.vllm_config.speculative_config.draft_model_config.hf_config
            info["draft_cfg"] = {"dflash_config": getattr(dm, "dflash_config", None),
                                 "block_size": getattr(dm, "block_size", None),
                                 "rope_theta": getattr(dm, "rope_theta", None),
                                 "rope_parameters": getattr(dm, "rope_parameters", None),
                                 "num_hidden_layers": dm.num_hidden_layers}
        except Exception as e:
            info["cfg_error"] = repr(e)
        try:
            info["metrics"] = {m.name: getattr(m, "value", None) or getattr(m, "values", None)
                               for m in llm.get_metrics() if "spec" in m.name}
        except Exception as e:
            info["metrics_error"] = repr(e)
    del llm
    return res, info


if __name__ == "__main__":
    _hooks()
    plain, _ = run(False)
    spec, info = run(True)
    print("RESULT", json.dumps({"plain": plain, "spec": spec, "lossless": plain == spec, **info, **REC},
                               default=str))
