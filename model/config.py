# SPDX-License-Identifier: Apache-2.0
"""Draft configuration in the public DFlash release format.

The format is the one read by vLLM's ``qwen3_dflash.py`` and by vllm-neuron's
DFlash drafter: a Qwen3 config with ``architectures = ["DFlashDraftModel"]``,
``block_size`` and a ``dflash_config`` dict carrying ``target_layer_ids`` and
``mask_token_id``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Transformers-native checkpoint (weights identical to Qwen/Qwen3-ASR-1.7B rev 7278e1e7,
# which stores them under ``thinker.*``; checked tensor by tensor).
TARGET_ID = "Qwen/Qwen3-ASR-1.7B-hf"
TARGET_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
TARGET_ORIG_ID = "Qwen/Qwen3-ASR-1.7B"
TARGET_ORIG_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
TARGET_NUM_LAYERS = 28


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Evenly spaced target layers from the 2nd to the 3rd-to-last.

    Same rule as the public z-lab DFlash reference (``utils.build_target_layer_ids``)
    and the paper (Sec. 5, "5 layers uniformly selected between the second layer and
    the third-to-last layer"). For a 28-layer target and 5 features: [1, 7, 13, 19, 25].
    """
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [int(round(start + i * span / (num_draft_layers - 1))) for i in range(num_draft_layers)]


@dataclass
class DraftConfig:
    # Draft transformer (copies the target text-decoder dims, as public DFlash drafts do).
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    # Draft-owned RoPE. 1e6 equals the target's theta and the default that vLLM and
    # vllm-neuron's DFlash config fall back to when transformers>=5 moves rope_theta into
    # rope_parameters, so it resolves identically in every loader (no silent drop).
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 65536
    vocab_size: int = 151936
    hidden_act: str = "silu"
    attention_bias: bool = False
    # DFlash
    block_size: int = 16
    # First embedding row past the tokenizer (last token 151704 "<asr_text>"); untrained, no
    # token string, so text encoding never produces it. Same convention as the public z-lab
    # Qwen3 drafts (151669), which cannot be used here: 151669 is Qwen3-ASR's <|audio_start|>.
    mask_token_id: int = 151705
    target_layer_ids: list[int] = field(
        default_factory=lambda: build_target_layer_ids(TARGET_NUM_LAYERS, 5)
    )
    num_target_layers: int = TARGET_NUM_LAYERS
    target_hidden_size: int = 2048
    # Tokenizer ids of the target (Qwen2 tokenizer): no BOS, EOS <|im_end|>, pad <|endoftext|>
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    @property
    def num_target_features(self) -> int:
        return len(self.target_layer_ids)

    def to_hf_dict(self, dtype: str = "bfloat16") -> dict:
        """Release ``config.json`` (Qwen3 + DFlash fields)."""
        n = self.num_hidden_layers
        return {
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "attention_bias": self.attention_bias,
            "attention_dropout": 0.0,
            "block_size": self.block_size,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "dflash_config": {
                "mask_token_id": self.mask_token_id,
                "target_layer_ids": list(self.target_layer_ids),
            },
            "dtype": dtype,
            "torch_dtype": dtype,
            "head_dim": self.head_dim,
            "hidden_act": self.hidden_act,
            "hidden_size": self.hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": self.intermediate_size,
            "layer_types": ["full_attention"] * n,
            "max_position_embeddings": self.max_position_embeddings,
            "max_window_layers": n,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": n,
            "num_key_value_heads": self.num_key_value_heads,
            "num_target_layers": self.num_target_layers,
            "rms_norm_eps": self.rms_norm_eps,
            # Written both ways: transformers<5 reads rope_theta, >=5 reads rope_parameters.
            # vLLM falls back to 1e6 when neither is present, which would silently break RoPE.
            "rope_theta": self.rope_theta,
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_theta},
            "rope_scaling": None,
            "sliding_window": None,
            "use_sliding_window": False,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_hf_dict(cls, d: dict) -> "DraftConfig":
        dc = d.get("dflash_config", {}) or {}
        rope = d.get("rope_theta")
        if rope is None:
            rope = (d.get("rope_parameters") or {}).get("rope_theta")
        if rope is None:
            raise ValueError("config has no rope_theta / rope_parameters.rope_theta")
        return cls(
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d.get("head_dim", d["hidden_size"] // d["num_attention_heads"]),
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=float(rope),
            max_position_embeddings=d.get("max_position_embeddings", 65536),
            vocab_size=d["vocab_size"],
            block_size=d.get("block_size", dc.get("block_size")),
            mask_token_id=dc.get("mask_token_id", d.get("mask_token_id")),
            target_layer_ids=list(dc.get("target_layer_ids", d.get("target_layer_ids"))),
            num_target_layers=d.get("num_target_layers", TARGET_NUM_LAYERS),
        )

    def save(self, path: str | Path, dtype: str = "bfloat16") -> None:
        Path(path).write_text(json.dumps(self.to_hf_dict(dtype), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "DraftConfig":
        return cls.from_hf_dict(json.loads(Path(path).read_text()))

    def asdict(self) -> dict:
        return asdict(self)
