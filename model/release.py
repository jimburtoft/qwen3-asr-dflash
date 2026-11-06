# SPDX-License-Identifier: Apache-2.0
"""Save / load the draft in the public DFlash release layout.

Release directory::

    config.json          Qwen3 + DFlash fields (see model/config.py)
    model.safetensors    fc, hidden_norm, layers.*, norm   (no embed_tokens / lm_head:
                         both loaders alias the target's)
    mask_embedding.pt    optional: {"mask_token_id": int, "embedding": [H]} (vLLM format)
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import DraftConfig
from .dflash import DFlashDraft


def save_release(model: DFlashDraft, out_dir: str | Path, dtype=torch.bfloat16,
                 mask_embedding: torch.Tensor | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(dtype).contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    model.cfg.save(out / "config.json", dtype={torch.bfloat16: "bfloat16", torch.float32: "float32"}[dtype])
    if mask_embedding is not None:
        torch.save({"mask_token_id": model.cfg.mask_token_id,
                    "embedding": mask_embedding.detach().to(dtype).reshape(-1)}, out / "mask_embedding.pt")
    return out


def load_release(path: str | Path, dtype=torch.float32) -> DFlashDraft:
    p = Path(path)
    cfg = DraftConfig.load(p / "config.json")
    m = DFlashDraft(cfg)
    sd = load_file(str(p / "model.safetensors"))
    m.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
    return m.to(dtype)
