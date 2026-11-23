#!/usr/bin/env python3
"""K1 (SpacemiT X60) operator evidence collector.

Run *on the K1 instance* inside the private venv with ``GEMS_VENDOR=spacemit``.
It records, per operator:

* the exact module/function chosen by FlagGems (vendor vs generic),
* a direct call to the vendor function (bypassing registration) compared with
  native PyTorch fp64,
* the registered ``aten`` path and how many times it was actually invoked,
* Triton cache artifacts created during the run,
* warmup+repeat latency with fixed shape/dtype/thread count.

Nothing here writes weights or private paths; JSON goes to ``--out``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (torch.Tensor,)):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    return value


def environment() -> dict:
    info = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "env": {k: os.environ[k] for k in ("GEMS_VENDOR", "SPINE_TRITON_TARGET_ARCH", "TRITON_CACHE_DIR") if k in os.environ},
    }
    try:
        import triton

        info["triton"] = triton.__version__
        info["triton_backend"] = type(triton.runtime.driver.active).__name__
        try:
            info["triton_target"] = str(triton.runtime.driver.active.get_current_target())
        except Exception as exc:
            info["triton_target"] = f"unavailable: {exc!r}"
    except Exception as exc:  # pragma: no cover
        info["triton"] = f"unavailable: {exc!r}"
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception as exc:  # pragma: no cover
        info["torchvision"] = f"unavailable: {exc!r}"
    isa = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("isa"):
                isa = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    info["isa"] = isa
    model = Path("/proc/device-tree/model")
    if model.is_file():
        info["device_tree_model"] = model.read_text(errors="replace").strip("\x00\n")
    return info


def metrics(out: torch.Tensor, ref: torch.Tensor, atol=1e-3, rtol=1e-3) -> dict:
    out64 = out.detach().cpu().double().reshape(-1)
    ref64 = ref.detach().cpu().double().reshape(-1)
    diff = (out64 - ref64).abs()
    return {
        "out_shape": list(out.shape),
        "ref_shape": list(ref.shape),
        "max_abs_err": float(diff.max().item()) if diff.numel() else None,
        "mean_abs_err": float(diff.mean().item()) if diff.numel() else None,
        "allclose": bool(out.shape == ref.shape and torch.allclose(out64, ref64, atol=atol, rtol=rtol)),
        "has_nan": bool(torch.isnan(out64).any()),
    }


def timed(fn, warmup=3, repeat=10) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "warmup": warmup,
        "repeat": repeat,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def cache_snapshot(root: Path, start: float) -> dict:
    try:
        new = [p for p in root.rglob("*") if p.is_file() and p.stat().st_mtime >= start]
    except Exception as exc:  # pragma: no cover
        return {"root": str(root), "error": repr(exc)}
    names = sorted({p.relative_to(root).parts[0] for p in new if p.relative_to(root).parts})
    return {"root": str(root), "new_files": len(new), "top_level_entries": names[:20]}


def dispatch_names(fn) -> dict:
    from torch.utils._python_dispatch import TorchDispatchMode

    class Counter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            key = str(func)
            self.counts[key] = self.counts.get(key, 0) + 1
            return func(*args, **(kwargs or {}))

    counter = Counter()
    with counter:
        fn()
    return dict(sorted(counter.counts.items()))


def with_registered(flag_gems, include: str, attr: str, call, path: Path, bench=None) -> dict:
    """Register only ``include``, wrap the live attr to count real calls."""
    calls = {"n": 0}
    original = getattr(flag_gems, attr)

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    counted.__name__ = getattr(original, "__name__", attr)
    counted.__module__ = getattr(original, "__module__", "")
    setattr(flag_gems, attr, counted)
    record = {"include": include, "registered_attr": attr, "attr_module": counted.__module__}
    try:
        with flag_gems.use_gems(include=[include], record=True, path=str(path)):
            record["out"] = call()
            record["vendor_calls"] = calls["n"]
            if bench is not None:
                record["timing_vendor_ms"] = timed(bench)
    except Exception:
        record["error"] = traceback.format_exc(limit=12)
        record["vendor_calls"] = calls["n"]
    finally:
        setattr(flag_gems, attr, original)
    return record


CONV_CASES = (
    {"name": "c3_s1_p1_bias", "x": (1, 3, 8, 8), "w": (4, 3, 3, 3), "stride": 1, "padding": 1, "dilation": 1, "bias": True},
    {"name": "c3_s2_p1_bias", "x": (1, 3, 9, 7), "w": (4, 3, 3, 3), "stride": 2, "padding": 1, "dilation": 1, "bias": True},
    {"name": "c3_s1_p0_nobias", "x": (1, 3, 8, 8), "w": (4, 3, 3, 3), "stride": 1, "padding": 0, "dilation": 1, "bias": False},
    {"name": "c4_s2_p1_bias", "x": (2, 4, 15, 13), "w": (6, 4, 3, 3), "stride": 2, "padding": 1, "dilation": 1, "bias": True},
    {"name": "resnet_stem_64", "x": (1, 3, 64, 64), "w": (8, 3, 7, 7), "stride": 2, "padding": 3, "dilation": 1, "bias": False, "slow": True},
    {"name": "resnet_stem_256", "x": (1, 3, 256, 256), "w": (64, 3, 7, 7), "stride": 2, "padding": 3, "dilation": 1, "bias": False, "slow": True},
)


def make_conv(case, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(case["x"], generator=gen, dtype=torch.float32)
    w = torch.randn(case["w"], generator=gen, dtype=torch.float32)
    b = torch.randn((case["w"][0],), generator=gen, dtype=torch.float32) if case["bias"] else None
    return x, w, b


def run_direct_conv(flag_gems, out: dict, skip_slow: bool) -> None:
    from flag_gems.runtime.backend._spacemit.ops.conv2d import conv2d as vconv

    sig = str(__import__("inspect").signature(vconv))
    out["direct_conv2d"] = {
        "vendor_module": vconv.__module__,
        "vendor_signature": sig,
        "cases": [],
    }
    for case in CONV_CASES:
        if skip_slow and case.get("slow"):
            continue
        x, w, b = make_conv(case)
        ref = F.conv2d(x.double(), w.double(), None if b is None else b.double(), stride=case["stride"], padding=case["padding"], dilation=case["dilation"]).float()
        rec = {"name": case["name"], "input": list(x.shape), "weight": list(w.shape), "stride": case["stride"], "padding": case["padding"], "dilation": case["dilation"], "bias": b is not None}
        try:
            y = vconv(x, w, b, padding=case["padding"], stride=case["stride"], dilation=case["dilation"], groups=1)
            rec["metrics"] = metrics(y, ref)
            rec["timing"] = timed(lambda: vconv(x, w, b, padding=case["padding"], stride=case["stride"], dilation=case["dilation"], groups=1), warmup=1 if case.get("slow") else 2, repeat=3 if case.get("slow") else 5)
        except Exception:
            rec["error"] = traceback.format_exc(limit=12)
        if case["name"] == "c3_s1_p1_bias":
            # aten order is (input, weight, bias, stride, padding, ...); call it
            # with deliberately different values (stride=2, padding=1) so a
            # stride/padding swap changes the output shape.
            pos = CONV_CASES[1]
            xp, wp, bp = make_conv(pos)
            refp = F.conv2d(xp.double(), wp.double(), bp.double(), stride=2, padding=1).float()
            try:
                y_pos = vconv(xp, wp, bp, pos["stride"], pos["padding"], pos["dilation"], 1)
                rec["aten_positional_call"] = {
                    "input": list(xp.shape), "aten_args": {"stride": 2, "padding": 1},
                    **metrics(y_pos, refp),
                }
            except Exception:
                rec["aten_positional_call"] = {"error": traceback.format_exc(limit=6)}
        out["direct_conv2d"]["cases"].append(rec)


def run_aten_conv(flag_gems, out: dict, out_dir: Path) -> None:
    import torch.nn.functional as F

    x, w, b = make_conv(CONV_CASES[1])
    ref = F.conv2d(x.double(), w.double(), b.double(), stride=2, padding=1).float()
    rec = {
        "case": "c3_s2_p1_bias",
        "generic_module": getattr(flag_gems.conv2d, "__module__", None),
        "generic_signature": str(__import__("inspect").signature(flag_gems.conv2d)),
    }
    try:
        with flag_gems.use_gems(include=["conv2d"], record=True, path=str(out_dir / "baseline_dispatch.log")):
            y = F.conv2d(x, w, b, stride=2, padding=1)
        rec["metrics"] = metrics(y, ref)
    except Exception:
        rec["error"] = traceback.format_exc(limit=12)
    out["aten_conv2d_baseline"] = rec


def count_kernel_runs(vendor_fn, kernel_name: str):
    """Wrap the Triton kernel object actually captured by the vendor function.

    The backend loads its modules as top-level ``_spacemit.ops.*``; importing
    ``flag_gems.runtime.backend._spacemit.ops.*`` would create a second module
    whose kernel object is not the one launched.  ``__globals__`` is the only
    reliable handle.
    """
    kernel = vendor_fn.__globals__[kernel_name]
    target = kernel if hasattr(kernel, "run") else getattr(kernel, "fn", kernel)
    if not hasattr(target, "run"):
        raise AttributeError(f"{type(kernel).__name__} has no runnable kernel")
    original_run = target.run

    def counted_run(*args, **kwargs):
        counted_run.calls += 1
        return original_run(*args, **kwargs)

    counted_run.calls = 0
    target.run = counted_run
    return target, original_run, counted_run


def restore_kernel(target, original_run) -> None:
    target.run = original_run


def run_mean(flag_gems, out: dict, out_dir: Path) -> None:
    # PetVision feeds 256x256 into ResNet-50, so the global-pool input is
    # (1, 2048, 8, 8); (1, 2048, 1, 1) is the already-collapsed variant.
    x = torch.randn((1, 2048, 8, 8), dtype=torch.float32)
    x1 = torch.randn((1, 2048, 1, 1), dtype=torch.float32)
    native = torch.mean(x, dim=[2, 3], keepdim=True)
    ref = torch.mean(x.double(), dim=[2, 3], keepdim=True).float()
    native1 = torch.mean(x1, dim=[2, 3], keepdim=True)
    ref1 = torch.mean(x1.double(), dim=[2, 3], keepdim=True).float()
    out["mean_dim"] = {
        "shape": list(x.shape),
        "shape_1x1": list(x1.shape),
        "native_module": torch.mean.__module__,
        "unregistered_dispatch": dispatch_names(lambda: torch.mean(x, dim=[2, 3], keepdim=True)),
        "all_dispatch": dispatch_names(lambda: torch.mean(x)),
        "adaptive_pool_dispatch": dispatch_names(lambda: F.adaptive_avg_pool2d(x, (1, 1))),
    }
    start = time.time()
    cache_root = Path(os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")))
    rec = {"include": "mean_dim", "dispatch_attr_module": getattr(flag_gems.mean_dim, "__module__", None)}
    y = None
    y1 = None
    try:
        with flag_gems.use_gems(include=["mean_dim"]):
            # First call lets the libtuner pick/compile a configuration.
            torch.mean(x, dim=[2, 3], keepdim=True)
            target, original_run, counted = count_kernel_runs(
                flag_gems.mean_dim, "mean_dim_kernel"
            )
            try:
                y = torch.mean(x, dim=[2, 3], keepdim=True)
            except Exception:
                rec["error"] = traceback.format_exc(limit=12)
            finally:
                restore_kernel(target, original_run)
                rec["kernel_runs"] = counted.calls
            if y is not None:
                y1 = torch.mean(x1, dim=[2, 3], keepdim=True)
                # Timed in the same registered context, after removing the hook.
                rec["timing_vendor_ms"] = timed(lambda: torch.mean(x, dim=[2, 3], keepdim=True))
    except Exception:
        rec.setdefault("error", traceback.format_exc(limit=12))
    # Metrics after the context exits: torch.mean(.mean()) would otherwise be
    # intercepted by the registered op again.
    rec["native_metrics"] = metrics(native, ref)
    rec["timing_native_ms"] = timed(lambda: torch.mean(x, dim=[2, 3], keepdim=True))
    if y is not None:
        rec["vendor_metrics"] = metrics(y, ref)
        rec["vendor_vs_native"] = metrics(y, native)
        rec["one_by_one"] = {
            "native_metrics": metrics(native1, ref1),
            "vendor_metrics": metrics(y1, ref1),
            "vendor_vs_native": metrics(y1, native1),
        }
    rec["triton_cache"] = cache_snapshot(cache_root, start)
    out["mean_dim"].update(rec)


def run_softmax(flag_gems, out: dict, out_dir: Path) -> None:
    x = torch.randn((1, 37), dtype=torch.float32)
    native = torch.softmax(x, dim=-1)
    ref = torch.softmax(x.double(), dim=-1).float()
    out["softmax"] = {
        "shape": list(x.shape),
        "unregistered_dispatch": dispatch_names(lambda: torch.softmax(x, dim=-1)),
    }
    start = time.time()
    cache_root = Path(os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")))
    rec = {"include": "softmax", "dispatch_attr_module": getattr(flag_gems.softmax, "__module__", None)}
    y = None
    try:
        with flag_gems.use_gems(include=["softmax"]):
            torch.softmax(x, dim=-1)
            target, original_run, counted = count_kernel_runs(
                flag_gems.softmax, "softmax_kernel_spacemit"
            )
            try:
                y = torch.softmax(x, dim=-1)
            except Exception:
                rec["error"] = traceback.format_exc(limit=12)
            finally:
                restore_kernel(target, original_run)
                rec["kernel_runs"] = counted.calls
            if y is not None:
                rec["timing_vendor_ms"] = timed(lambda: torch.softmax(x, dim=-1))
    except Exception:
        rec.setdefault("error", traceback.format_exc(limit=12))
    rec["native_metrics"] = metrics(native, ref)
    rec["timing_native_ms"] = timed(lambda: torch.softmax(x, dim=-1))
    if y is not None:
        rec["vendor_metrics"] = metrics(y, ref)
        rec["vendor_vs_native"] = metrics(y, native)
    rec["triton_cache"] = cache_snapshot(cache_root, start)
    out["softmax"].update(rec)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", choices=("all", "conv2d", "mean", "softmax"), default="all")
    parser.add_argument("--skip-slow", action="store_true", help="skip the 64/256 stem cases")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "kind": "k1_ops_check",
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    try:
        import flag_gems
        # Importing flag_gems activates the SpacemiT Triton driver, so collect
        # the environment only afterwards.
        payload["environment"] = environment()

        payload["flag_gems"] = {
            "version": getattr(flag_gems, "__version__", None),
            "vendor": flag_gems.vendor_name,
            "device": str(flag_gems.device),
            "conv2d_module": getattr(flag_gems.conv2d, "__module__", None),
            "mean_dim_module": getattr(flag_gems.mean_dim, "__module__", None),
            "softmax_module": getattr(flag_gems.softmax, "__module__", None),
        }
        if flag_gems.vendor_name != "spacemit":
            raise SystemExit(
                f"GEMS_VENDOR resolved to {flag_gems.vendor_name!r}, expected 'spacemit'"
            )
        if args.only in ("all", "conv2d"):
            run_aten_conv(flag_gems, payload, out_dir)
            run_direct_conv(flag_gems, payload, args.skip_slow)
        if args.only in ("all", "mean"):
            run_mean(flag_gems, payload, out_dir)
        if args.only in ("all", "softmax"):
            run_softmax(flag_gems, payload, out_dir)
    except Exception:
        payload["fatal_error"] = traceback.format_exc(limit=20)

    # A selected operator is a strict check; ``all`` remains a collection mode
    # because it intentionally records known failures alongside passing ops.
    if "softmax" in payload:
        sm = payload["softmax"]
        payload["operator_passed"] = bool(
            sm.get("kernel_runs") == 1
            and sm.get("vendor_metrics", {}).get("allclose")
            and sm.get("vendor_vs_native", {}).get("allclose")
        )
    if "mean_dim" in payload:
        mean_passed = bool(payload["mean_dim"].get("vendor_metrics", {}).get("allclose"))
        payload["mean_dim"]["operator_passed"] = mean_passed
        if args.only == "mean":
            payload["operator_passed"] = mean_passed
    if "direct_conv2d" in payload:
        cases = payload["direct_conv2d"].get("cases", [])
        conv_passed = bool(cases) and all(
            c.get("metrics", {}).get("allclose") for c in cases
        )
        payload["direct_conv2d"]["operator_passed"] = conv_passed
        if args.only == "conv2d":
            payload["operator_passed"] = conv_passed
    collected = "fatal_error" not in payload
    if args.only in ("softmax", "conv2d", "mean"):
        payload["passed"] = bool(collected and payload.get("operator_passed"))
    else:
        # Collection mode preserves the original failure-reproduction contract.
        payload["passed"] = bool(collected)
        payload["failure_reproduction"] = True
    path = out_dir / "k1_ops_check.json"
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(f"passed={payload['passed']} operator_passed={payload.get('operator_passed')}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
