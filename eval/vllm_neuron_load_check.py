# SPDX-License-Identifier: Apache-2.0
"""Load a release-format draft in public vllm-neuron's DFlash path and check greedy losslessness.

FORMAT / loader check, not an acceptance measurement (text prompt, no audio). Target = the
Qwen3-ASR text decoder exported as Qwen3ForCausalLM (tools/extract_text_decoder.py).

Runs on trn2, or on CPU with ``VLLM_NEURON_CPU_MODE=1`` (the public SDK 2.33 DLAMI's
``aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`` venv)::

    VLLM_NEURON_CPU_MODE=1 python eval/vllm_neuron_load_check.py <target_dir> <draft_dir> [K]

Prints the resolved DFlash config (block size, mask id, target layers, aux capture layers, draft
RoPE base) as seen by vllm-neuron, and spec == plain greedy output.
"""

import json
import os
import sys


TARGET, DRAFT = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 15
PROMPT = os.environ.get("PROMPT", "<|im_start|>user\nWrite one sentence about the city of Paris.<|im_end|>\n<|im_start|>assistant\n")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "48"))
REC: dict = {}


def resolved_config():
    """The draft config exactly as vllm-neuron's DFlash loader parses it (same entry point)."""
    from transformers import AutoConfig
    from vllm_neuron.model.qwen3.config import DFlashConfig
    c = DFlashConfig.from_configs(AutoConfig.from_pretrained(DRAFT))
    return {k: getattr(c, k) for k in ("block_size", "mask_token_id", "target_layer_ids", "rope_theta",
                                       "hidden_size", "intermediate_size", "num_hidden_layers",
                                       "num_attention_heads", "num_key_value_heads", "vocab_size", "rms_norm_eps")}


def run(spec):
    from vllm import LLM, SamplingParams
    kw = dict(model=TARGET, dtype="bfloat16", max_model_len=1024, max_num_seqs=1,
              max_num_batched_tokens=1024, tensor_parallel_size=1, enable_prefix_caching=False,
              additional_config={"neuron_config": {"on_device_sampling_config": {"all_greedy": "true"}}})
    if spec:
        kw["speculative_config"] = {"method": "dflash", "model": DRAFT, "num_speculative_tokens": K}
    llm = LLM(**kw)
    outs = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
    res = [list(o.outputs[0].token_ids) for o in outs]
    del llm
    return res


if __name__ == "__main__":
    REC["vllm_neuron_dflash_config"] = resolved_config()
    which = os.environ.get("RUN", "both")
    plain = run(False) if which in ("both", "plain") else None
    spec = run(True) if which in ("both", "spec") else None
    # the engine core logs "Using DFlash auxiliary layers: (...)" with the capture layers it requested
    print("RESULT", json.dumps({"plain": plain, "spec": spec,
                               "lossless": (plain == spec) if plain and spec else None, **REC}, default=str))
