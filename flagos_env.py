"""Shared environment, timing and comparison helpers.

Used by ``conv2d_case.py`` and ``petvision_forward.py``.  Nothing here imports
the course project; all inputs are explicit command-line arguments.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

ENV_KEYS = (
    "GEMS_VENDOR",
    "FLAGGEMS_VENDOR",
    "SPINE_TRITON_TARGET_ARCH",
    "SPACEMIT_CACHE",
    "TRITON_CACHE_DIR",
    "GEMS_DEBUG",
)


def _device_sync(device: torch.device) -> str:
    """Drain asynchronous work for ``device`` and report what was used."""
    if device.type == "cpu":
        # CPU tensors are executed eagerly: there is no queue to drain.
        return "none (eager CPU execution)"
    backend = getattr(torch, device.type, None)
    sync = getattr(backend, "synchronize", None)
    if sync is None:
        return f"unavailable for torch.{device.type}"
    sync()
    return f"torch.{device.type}.synchronize()"


def resolve_device(backend: str, requested: str = "auto") -> torch.device:
    """Device the candidate implementation runs on.

    ``flag_gems`` exposes its backend device (``"cpu"`` for the SpacemiT
    backend, ``"cuda"`` etc. for GPU backends); native-reference timings always
    stay on CPU so the reference is comparable and its logits are portable.
    """
    if requested and requested != "auto":
        return torch.device(requested)
    if backend == "flag_gems":
        import flag_gems

        return torch.device(str(flag_gems.device))
    return torch.device("cpu")


def time_calls(fn, warmup: int, repeat: int, device: torch.device) -> dict:
    """Time ``fn`` after ``warmup`` calls; return per-call milliseconds."""
    sync_mode = _device_sync(device)
    for _ in range(max(0, warmup)):
        fn()
    _device_sync(device)
    samples = []
    for _ in range(repeat):
        _device_sync(device)
        start = time.perf_counter()
        fn()
        _device_sync(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "warmup": warmup,
        "count": repeat,
        "sync": sync_mode,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p50_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }


def tensor_metrics(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    """Numerical difference between ``out`` and an fp64 reference."""
    out64 = out.detach().cpu().double()
    ref64 = ref.detach().cpu().double()
    diff = (out64 - ref64).abs()
    scale = ref64.abs().clamp_min(1e-12)
    return {
        "shape": list(out.shape),
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "max_rel_err": (diff / scale).max().item(),
        "ref_abs_max": ref64.abs().max().item(),
        "out_has_nan": bool(torch.isnan(out64).any()),
        "allclose": bool(torch.allclose(out64, ref64, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def collect_env(flaggems_root: str = "") -> dict:
    """Record versions and environment variables relevant to a run."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_mkldnn": bool(torch.backends.mkldnn.is_available()),
        "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:  # pragma: no cover - absence is a valid result
        info["triton"] = f"unavailable: {exc.__class__.__name__}"
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception as exc:  # pragma: no cover
        info["torchvision"] = f"unavailable: {exc.__class__.__name__}"
    try:
        import flag_gems

        info["flag_gems"] = {
            "version": getattr(flag_gems, "__version__", None),
            "vendor": getattr(flag_gems, "vendor_name", None),
            "device": str(getattr(flag_gems, "device", None)),
        }
    except Exception as exc:  # pragma: no cover
        info["flag_gems"] = f"unavailable: {exc.__class__.__name__}"
    if flaggems_root:
        info["flag_gems_root"] = str(Path(flaggems_root).resolve())
        head = Path(flaggems_root) / ".git" / "HEAD"
        info["flag_gems_git_head"] = head.read_text().strip() if head.is_file() else None
    return info


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def vendor_dispatch_probe(device: torch.device) -> dict:
    """Prove that ``aten::conv2d`` is dispatched to the vendor kernel.

    Must be called while FlagGems is enabled.  A grouped convolution is used
    because the SpacemiT kernel explicitly rejects ``groups != 1``; native
    PyTorch would compute it happily, so a raised ``NotImplementedError`` is
    direct evidence that the vendor implementation, not the generic one, ran.
    """
    import torch.nn.functional as F

    gen = torch.Generator(device="cpu").manual_seed(0)
    inp = torch.randn((1, 4, 8, 8), generator=gen, dtype=torch.float32).to(device)
    weight = torch.randn((4, 2, 3, 3), generator=gen, dtype=torch.float32).to(device)
    try:
        F.conv2d(inp, weight, groups=2)
    except NotImplementedError as exc:
        return {
            "vendor_kernel_observed": True,
            "detail": str(exc).splitlines()[0],
        }
    except Exception as exc:  # pragma: no cover - unexpected backend failure
        return {
            "vendor_kernel_observed": False,
            "detail": f"{exc.__class__.__name__}: {exc}".splitlines()[0],
        }
    return {
        "vendor_kernel_observed": False,
        "detail": "groups=2 succeeded, so the generic conv2d kernel is still active",
    }


def read_log_tail(path: str | Path, limit: int = 40) -> list[str]:
    path = Path(path)
    if not path.is_file():
        return []
    lines = [ln.rstrip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    return [ln for ln in lines if ln][-limit:]