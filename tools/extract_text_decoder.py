# SPDX-License-Identifier: Apache-2.0
"""Extract Qwen3-ASR's text decoder as a plain Qwen3ForCausalLM checkpoint (dev tool).

Used to exercise DFlash loaders (upstream vLLM, vllm-neuron) against the exact 28-layer target
decoder when the multimodal wrapper does not expose aux hidden states. Writes a local copy only;
the extracted weights are Qwen3-ASR weights (Apache-2.0, Qwen team) and are not part of the draft
release.
"""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import TARGET_ID, TARGET_REVISION  # noqa: E402

PREFIX = "model.language_model."


def main(out: str):
    src = Path(snapshot_download(TARGET_ID, revision=TARGET_REVISION))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    tc = dict(cfg["text_config"])
    rope = tc.get("rope_theta") or (tc.get("rope_parameters") or {}).get("rope_theta")
    assert float(rope) == 1e6, rope
    # Audio positions use the same index in all mRoPE sections, so plain 1-D RoPE is exact.
    for k in ("rope_scaling", "rope_parameters"):
        tc.pop(k, None)
    tc.update({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "torch_dtype": "bfloat16",
               "dtype": "bfloat16", "rope_theta": float(rope),
               "rope_parameters": {"rope_type": "default", "rope_theta": float(rope)},
               "bos_token_id": 151643, "eos_token_id": [151645, 151643], "tie_word_embeddings": True})
    (out / "config.json").write_text(json.dumps(tc, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        if (src / f).exists():
            (out / f).write_bytes((src / f).read_bytes())
    sd = {}
    with safe_open(src / "model.safetensors", "pt") as fh:
        for k in fh.keys():
            if k.startswith(PREFIX):
                sd["model." + k[len(PREFIX):]] = fh.get_tensor(k)
        # Explicit (tied) LM head: some loaders (vllm-neuron's Qwen3) require the tensor.
        sd["lm_head.weight"] = (fh.get_tensor("lm_head.weight") if "lm_head.weight" in fh.keys()
                                else sd["model.embed_tokens.weight"]).clone()
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(out, len(sd))


if __name__ == "__main__":
    main(sys.argv[1])
