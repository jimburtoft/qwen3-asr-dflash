# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Qwen3-ASR release contract and the static target's shape helpers (no downloads)."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DraftConfig, build_target_layer_ids  # noqa: E402
from model.dflash import DFlashDraft  # noqa: E402
from model.qwen3_asr_target import (  # noqa: E402
    AUDIO_END_ID, AUDIO_START_ID, AUDIO_TOKEN_ID, TOKENS_PER_WINDOW, audio_token_count, post_cnn_len,
    window_layout,
)
from model.release import load_release, save_release  # noqa: E402


def test_contract_values():
    c = DraftConfig()
    assert build_target_layer_ids(28, 5) == [1, 7, 13, 19, 25] == c.target_layer_ids
    assert c.block_size == 16 and c.rope_theta == 1e6
    assert c.mask_token_id == 151705
    assert c.mask_token_id not in (AUDIO_START_ID, AUDIO_END_ID, AUDIO_TOKEN_ID, 151669)
    assert (c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads) == (2048, 6144, 16, 8)
    d = c.to_hf_dict()
    assert d["dflash_config"] == {"mask_token_id": 151705, "target_layer_ids": [1, 7, 13, 19, 25]}
    assert d["rope_theta"] == d["rope_parameters"]["rope_theta"] == 1e6
    assert DraftConfig.from_hf_dict(json.loads(json.dumps(d))).target_layer_ids == c.target_layer_ids


def test_release_roundtrip(tmp_path):
    c = DraftConfig(num_hidden_layers=1)
    m = DFlashDraft(c)
    save_release(m, tmp_path, dtype=torch.float32)
    m2 = load_release(tmp_path)
    for (k, v), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)
    assert not any(k.startswith(("embed_tokens", "lm_head")) for k in m.state_dict())


def test_audio_token_counts():
    # 13 tokens per full 100-frame chunk; partial chunk -> three stride-2 convs
    assert audio_token_count(100) == 13
    assert audio_token_count(350) == 3 * 13 + int(post_cnn_len(50)) == 3 * 13 + 7
    W, valid = window_layout(3000)                 # 30 s
    assert W == 4 and sum(valid) == audio_token_count(3000) == 390
    assert valid[:3] == [TOKENS_PER_WINDOW] * 3
