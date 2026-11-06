# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DFlash training mask, loss and draft forward (CPU, small dims)."""

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DraftConfig, build_target_layer_ids  # noqa: E402
from model.dflash import DFlashDraft  # noqa: E402
from train.dflash_objective import (  # noqa: E402
    acceptance_stats, build_mask, dflash_loss, pack_batch, position_weights,
)


def tiny_cfg(**kw):
    base = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, vocab_size=97, block_size=8, mask_token_id=5,
                target_layer_ids=[1, 3, 5], num_target_layers=8, target_hidden_size=64,
                rope_theta=10000.0)
    base.update(kw)
    return DraftConfig(**base)


def test_target_layer_ids_rule():
    # z-lab public rule, matches the public Qwen3-8B (36 layers) checkpoint config
    assert build_target_layer_ids(36, 5) == [1, 9, 17, 25, 33]
    assert build_target_layer_ids(30, 5) == [1, 8, 14, 20, 27]


def test_position_weights():
    w = position_weights(16)
    assert w[0] == 0
    assert math.isclose(w[1].item(), 1.0)
    assert math.isclose(w[8].item(), math.exp(-1.0), rel_tol=1e-6)


def naive_mask(anchors, n_valid, ctx_len, Sc, n_blocks, bs):
    B = len(anchors)
    Sq = n_blocks * bs
    m = torch.zeros(B, Sq, Sc + Sq, dtype=torch.bool)
    for b in range(B):
        for qi in range(Sq):
            blk = qi // bs
            for j in range(Sc):
                if blk < n_valid[b] and j < anchors[b][blk] and j < ctx_len[b]:
                    m[b, qi, j] = True
            for kj in range(Sq):
                if kj // bs == blk:
                    m[b, qi, Sc + kj] = True
    return m


def test_mask_matches_naive():
    g = torch.Generator().manual_seed(0)
    B, n_blocks, bs, Sc = 3, 4, 4, 30
    anchors = torch.randint(5, 28, (B, n_blocks), generator=g).sort(-1).values
    n_valid = torch.tensor([4, 2, 0])
    ctx_len = torch.tensor([30, 25, 12])
    m = build_mask(anchors, n_valid, ctx_len, Sc, n_blocks, bs)
    ref = naive_mask(anchors.tolist(), n_valid.tolist(), ctx_len.tolist(), Sc, n_blocks, bs)
    assert torch.equal(m, ref)


def test_pack_labels_and_positions():
    s = torch.arange(100, 120)  # L=20
    P = 6
    pb = pack_batch([s], [P], n_blocks=3, block_size=4, mask_token_id=5,
                    anchors_override=[torch.tensor([6, 10, 17])])
    assert pb.block_ids[0].tolist() == [106, 5, 5, 5, 110, 5, 5, 5, 117, 5, 5, 5]
    assert pb.q_pos[0].tolist() == [6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20]
    # last block: slots 1,2 inside (118,119), slot 3 beyond the end
    assert pb.labels[0].tolist() == [-100, 107, 108, 109, -100, 111, 112, 113, -100, 118, 119, -100]
    assert pb.weights[0, 11] == 0 and pb.weights[0, 0] == 0


@pytest.mark.parametrize("dtype", [torch.float32])
def test_packed_forward_equals_per_block_inference(dtype):
    """Packed + masked training forward == running each block alone the way inference does:
    context = target features of positions < anchor, block attends to itself bidirectionally."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = DFlashDraft(cfg).to(dtype).eval()
    V, H, nf = cfg.vocab_size, cfg.hidden_size, cfg.num_target_features
    emb = torch.randn(V, H, dtype=dtype)
    seqs = [torch.randint(10, V, (40,)), torch.randint(10, V, (27,))]
    P = [12, 9]
    feats = torch.randn(2, 40, nf * H, dtype=dtype)
    pb = pack_batch(seqs, P, n_blocks=5, block_size=cfg.block_size, mask_token_id=cfg.mask_token_id,
                    generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = model(emb[pb.block_ids], feats, pb.q_pos, pb.ctx_pos, pb.mask)
        bs = cfg.block_size
        for b in range(2):
            for i in range(pb.n_blocks):
                sl = slice(i * bs, (i + 1) * bs)
                if pb.weights[b, sl].sum() == 0:
                    continue
                a = int(pb.q_pos[b, sl.start])
                ref = model(emb[pb.block_ids[b:b + 1, sl]], feats[b:b + 1, :a], pb.q_pos[b:b + 1, sl],
                            torch.arange(a)[None], None)
                torch.testing.assert_close(out[b:b + 1, sl], ref, atol=1e-5, rtol=1e-4)


def test_matches_public_zlab_reference():
    """Same weights in the public z-lab reference DFlashDraftModel give the same block output."""
    ref_dir = ROOT / "tmp" / "refs"
    if not (ref_dir / "zlab_dflash.py").exists():
        pytest.skip("public z-lab reference not downloaded to tmp/refs")
    import importlib
    import types
    pkg = types.ModuleType("zlabref")
    pkg.__path__ = [str(ref_dir / "zlabref")]
    (ref_dir / "zlabref").mkdir(exist_ok=True)
    (ref_dir / "zlabref" / "dflash.py").write_text((ref_dir / "zlab_dflash.py").read_text())
    (ref_dir / "zlabref" / "utils.py").write_text((ref_dir / "utils.py").read_text())
    sys.modules["zlabref"] = pkg
    zl = importlib.import_module("zlabref.dflash")
    from transformers import Qwen3Config

    torch.manual_seed(0)
    cfg = tiny_cfg()
    hf = Qwen3Config(**{k: v for k, v in cfg.to_hf_dict("float32").items()
                        if k not in ("architectures", "dtype", "torch_dtype", "rope_parameters")})
    hf._attn_implementation = "eager"
    ref = zl.DFlashDraftModel(hf).eval()
    ours = DFlashDraft(cfg).eval()
    missing, unexpected = ref.load_state_dict(ours.state_dict(), strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    H, nf = cfg.hidden_size, cfg.num_target_features
    a, bs = 17, cfg.block_size
    noise = torch.randn(1, bs, H)
    feats = torch.randn(1, a, nf * H)
    with torch.no_grad():
        o_ref = ref(position_ids=torch.arange(a + bs)[None], noise_embedding=noise,
                    target_hidden=feats, attention_mask=None, is_causal=False)
        o = ours(noise, feats, torch.arange(a, a + bs)[None], torch.arange(a)[None], None)
    torch.testing.assert_close(o, o_ref, atol=1e-5, rtol=1e-4)


def test_loss_and_acceptance():
    torch.manual_seed(0)
    s = torch.randint(10, 50, (30,))
    pb = pack_batch([s], [5], n_blocks=4, block_size=4, mask_token_id=5,
                    generator=torch.Generator().manual_seed(0))
    V = 60
    perfect = torch.full((1, 16, V), -10.0)
    lab = pb.labels.clamp(min=0)
    perfect.scatter_(-1, lab[..., None], 10.0)
    assert dflash_loss(perfect, pb) < 1e-3
    st = acceptance_stats(perfect.argmax(-1), pb)
    assert st["tau_full_blocks"] == 4.0


def test_device_helpers_cpu():
    from model.device import get_device, maybe_compile
    dev = get_device("cpu")
    assert dev.type == "cpu"

    def f(x):
        return x + 1
    assert maybe_compile(f, dev) is f  # eager on CPU
    assert get_device("auto").type in ("neuron", "cuda", "cpu")
