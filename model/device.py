# SPDX-License-Identifier: Apache-2.0
"""Device selection and compilation, shared by every script.

``--device auto`` picks Trainium (``neuron``) when torch-neuronx is installed and a device is
visible, else CUDA, else CPU. ``maybe_compile`` uses ``torch.compile(backend="neuron")`` on
Trainium, the default inductor backend on CUDA, and eager on CPU. All graphs are static-shape
(``dynamic=False``) so each shape bucket compiles once.
"""

from __future__ import annotations

import torch

DEVICE_CHOICES = ("auto", "neuron", "cuda", "cpu")


def _neuron_available() -> bool:
    try:
        import torch_neuronx  # noqa: F401
    except ImportError:
        return False
    mod = getattr(torch, "neuron", None)
    try:
        return bool(mod is not None and mod.is_available())
    except Exception:
        return False


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if _neuron_available():
            name = "neuron"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    if name == "neuron":
        import torch_neuronx  # noqa: F401  (registers the "neuron" device and compile backend)
    return torch.device(name)


def maybe_compile(fn, device: torch.device, enabled: bool = True):
    if not enabled or device.type == "cpu":
        return fn
    if device.type == "neuron":
        return torch.compile(fn, backend="neuron", dynamic=False)
    return torch.compile(fn, dynamic=False)


def add_device_arg(ap) -> None:
    ap.add_argument("--device", default="auto", choices=DEVICE_CHOICES,
                    help="auto = neuron if available, else cuda, else cpu")
