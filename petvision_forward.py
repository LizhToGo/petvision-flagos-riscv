"""Run the PetVision ResNet-50 classifier through the active Conv2d backend.

The architecture below reproduces the course project's final classifier
(``torchvision.resnet50`` backbone, ``fc -> Identity``, ``Linear(2048, 37)``)
without importing the course package.  Weights are read from an explicit
``--weights`` path and are never written or copied.

What is measured:
  * native PyTorch CPU forward (logits + warmup/repeat latency)
  * FlagGems forward with ``--backend flag_gems`` (same tensors, same process)
  * Conv2d-call counts via forward hooks, proving the convolutions are on the
    executed path
  * size/mtime of the checkpoint before and after the run

Usage:
    python petvision_forward.py --backend torch \
        --weights <...>/raw/best.pt --out results/local_cpu
    GEMS_VENDOR=spacemit python petvision_forward.py --backend flag_gems \
        --weights <...>/raw/best.pt --require-vendor spacemit \
        --out results/spacemit_emu
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from flagos_env import (
    collect_env,
    read_log_tail,
    resolve_device,
    time_calls,
    vendor_dispatch_probe,
    write_json,
)


class PetVisionClassifier(nn.Module):
    """Mirror of petvision.models.classifier.ResNetClassifier (resnet50)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        from torchvision.models import resnet50

        self.backbone = resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(in_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def checkpoint_inventory(checkpoint: dict) -> dict:
    state = checkpoint["model_state"]
    conv_configs = {}
    conv_layers = 0
    for key, value in state.items():
        if not key.endswith(".weight") or value.ndim != 4:
            continue
        if not (key.startswith("backbone.") or ".backbone." in key):
            continue
        conv_layers += 1
        out_ch, in_ch, kh, kw = value.shape
        config = f"weight={list(value.shape)}"
        conv_configs[config] = conv_configs.get(config, 0) + 1
    return {
        "state_dict_entries": len(state),
        "conv_weights_4d": conv_layers,
        "conv_weight_shapes": dict(sorted(conv_configs.items())),
    }


def module_inventory(model: nn.Module) -> dict:
    conv_layers = 0
    conv_transpose_layers = 0
    grouped = 0
    configs = {}
    for module in model.modules():
        if isinstance(module, nn.ConvTranspose2d):
            conv_transpose_layers += 1
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        conv_layers += 1
        if module.groups != 1:
            grouped += 1
        key = (
            f"kernel={list(module.kernel_size)} stride={list(module.stride)} "
            f"padding={list(module.padding)} dilation={list(module.dilation)} "
            f"groups={module.groups} bias={module.bias is not None}"
        )
        entry = configs.setdefault(
            key,
            {"count": 0, "in_channels": [], "out_channels": []},
        )
        entry["count"] += 1
        entry["in_channels"].append(module.in_channels)
        entry["out_channels"].append(module.out_channels)
    return {
        "conv2d_layers": conv_layers,
        "conv_transpose2d_layers": conv_transpose_layers,
        "grouped_conv2d_layers": grouped,
        "conv2d_configs": configs,
    }


def load_model(weights: Path):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    class_names = list(checkpoint["class_names"])
    model = PetVisionClassifier(len(class_names))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, class_names


def logits_metrics(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    # Always compare on CPU so cross-device runs work and logits are portable.
    out = out.detach().cpu().reshape(-1).double()
    ref = ref.detach().cpu().reshape(-1).double()
    diff = (out - ref).abs()
    top1_out = int(out.argmax())
    top1_ref = int(ref.argmax())
    order_out = torch.argsort(out, descending=True)[:5].tolist()
    order_ref = torch.argsort(ref, descending=True)[:5].tolist()
    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "allclose": bool(torch.allclose(out, ref, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "top1_index_backend": top1_out,
        "top1_index_reference": top1_ref,
        "top1_agree": top1_out == top1_ref,
        "top5_overlap": len(set(order_out) & set(order_ref)),
        "top5_backend": order_out,
        "top5_reference": order_ref,
        "logits_backend": [round(v, 6) for v in out.tolist()],
        "logits_reference": [round(v, 6) for v in ref.tolist()],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="course checkpoint, read-only")
    parser.add_argument("--backend", choices=("torch", "flag_gems"), default="torch")
    parser.add_argument(
        "--enable-scope",
        choices=("conv2d", "all"),
        default="conv2d",
        help="register only conv2d or the whole FlagGems op table",
    )
    parser.add_argument(
        "--enable-ops",
        default="",
        help="comma separated FlagGems include list (overrides --enable-scope), "
        "e.g. mean.dim or conv2d",
    )
    parser.add_argument(
        "--count-dispatch",
        action="store_true",
        help="count ATen dispatch events for one extra forward (not timed)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch.set_num_threads() value; 0 keeps the default",
    )
    parser.add_argument(
        "--image",
        default="",
        help="optional JPEG/PNG classified with the course preprocessing; "
        "mutually exclusive with the random seed input",
    )
    parser.add_argument("--image-size", type=int, default=0, help="0 = use checkpoint value")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto = flag_gems.device for --backend flag_gems, otherwise cpu",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--require-vendor", default="")
    parser.add_argument("--flaggems-root", default="")
    parser.add_argument(
        "--reference-logits",
        default="",
        help="optional JSON logits file from another environment to compare against",
    )
    parser.add_argument(
        "--save-logits",
        default="",
        help="write this run's native logits as JSON usable by --reference-logits",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def _dispatch_name(func) -> str:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return str(func)
    return f"{schema.name}.{schema.overload_name or 'default'}"


class DispatchCounter(TorchDispatchMode):
    """Record ATen dispatch keys of one forward without changing the math."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        key = _dispatch_name(func)
        self.counts[key] = self.counts.get(key, 0) + 1
        return func(*args, **(kwargs or {}))


def count_dispatch(forward) -> dict:
    counter = DispatchCounter()
    with counter:
        forward()
    return dict(sorted(counter.counts.items()))


def count_kernel(vendor_fn, kernel_name: str) -> tuple:
    """Wrap the real Triton kernel object captured by the vendor function."""
    kernel = vendor_fn.__globals__[kernel_name]
    target = kernel if hasattr(kernel, "run") else getattr(kernel, "fn", kernel)
    original = target.run
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    target.run = counted
    return target, original, calls


def image_input(image_path: Path, image_size: int) -> tuple[torch.Tensor, dict]:
    """Course preprocessing: RGB, BILINEAR resize to 256x256, /255, ImageNet norm."""
    from PIL import Image
    from torchvision.transforms import functional as TF

    raw = image_path.read_bytes()
    image = Image.open(image_path).convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    tensor = TF.pil_to_tensor(image).float() / 255.0
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(3, 1, 1)
    meta = {
        "kind": "image",
        "name": image_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "resize": [image_size, image_size],
        "resample": "BILINEAR",
        "color": "RGB",
        "normalize": "imagenet",
    }
    return (tensor - mean) / std, meta


def topk_summary_from_probs(probs: torch.Tensor, class_names: list[str], k: int = 5) -> dict:
    """Summarize the first sample only (documented in the JSON as sample=0)."""
    flat = probs.detach().cpu().double().reshape(-1)
    order = torch.argsort(flat, descending=True)[:k]
    return {
        "sample": 0,
        "top1_index": int(order[0]),
        "top1_class": class_names[int(order[0])],
        "top1_probability": float(flat[order[0]]),
        "topk": [
            {
                "index": int(i),
                "class": class_names[int(i)],
                "probability": float(flat[i]),
            }
            for i in order
        ],
    }


def topk_summary(logits: torch.Tensor, class_names: list[str], k: int = 5) -> dict:
    # Logits are (batch, num_classes); show the first sample, never a flattened
    # batch (that would shift class indices and mix normalizations).
    first = logits.detach().cpu().double()[0]
    return topk_summary_from_probs(torch.softmax(first, dim=-1), class_names, k)


def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.backend, args.device)
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.is_file():
        raise SystemExit(f"weights not found: {weights_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stat_before = weights_path.stat()
    payload = {
        "schema": 1,
        "kind": "petvision_resnet50_forward",
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": collect_env(args.flaggems_root),
        "weights": {
            "path": str(weights_path),
            "size_bytes": stat_before.st_size,
            "mtime_ns_before": stat_before.st_mtime_ns,
            "opened_read_only": True,
        },
        "config": {
            "backend": args.backend,
            "device": str(device),
            "enable_scope": args.enable_scope,
            "enable_ops": args.enable_ops,
            "batch": args.batch,
            "image_size_arg": args.image_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
            "input_kind": "image" if args.image else "seeded randn",
        },
        "runtime": {
            "torch_num_threads": torch.get_num_threads(),
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "cpu_count": os.cpu_count(),
        },
    }

    model, checkpoint, class_names = load_model(weights_path)
    model.to(device)
    image_size = args.image_size or int(checkpoint.get("image_size", 256))
    payload["checkpoint"] = {
        "architecture": checkpoint.get("architecture"),
        "image_size": image_size,
        "mode": checkpoint.get("mode"),
        "precision": checkpoint.get("precision"),
        "num_classes": len(class_names),
        "class_names": class_names,
    }
    payload["checkpoint"]["state_dict"] = checkpoint_inventory(checkpoint)
    payload["model"] = module_inventory(model)

    if args.image and args.batch != 1:
        raise SystemExit("--image supports exactly one image (batch=1)")
    if args.image:
        image_tensor, input_meta = image_input(Path(args.image), image_size)
        # image_input returns CHW; the classifier expects NCHW.
        inputs = image_tensor.unsqueeze(0)
        payload["input"] = {
            **input_meta,
            "shape": list(inputs.shape),
            "device": str(device),
        }
    else:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        inputs = torch.randn(
            (args.batch, 3, image_size, image_size),
            generator=generator,
            dtype=torch.float32,
        )
        payload["input"] = {
            "kind": "seeded randn",
            "shape": list(inputs.shape),
            "seed": args.seed,
            "device": str(device),
        }
    inputs = inputs.to(device)

    call_counts = {}

    def hook(_module, _inputs, _output):
        call_counts["conv2d_calls"] = call_counts.get("conv2d_calls", 0) + 1

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(hook)

    def forward():
        with torch.no_grad():
            return model(inputs)

    try:
        with torch.no_grad():
            native_logits = forward()
        payload["native_torch"] = {
            "timing": time_calls(forward, args.warmup, args.repeat, device),
            "device": str(device),
            "conv2d_calls_total": call_counts.get("conv2d_calls", 0),
            "conv2d_calls_per_forward": call_counts.get("conv2d_calls", 0)
            / (1 + args.warmup + args.repeat),
        }
        call_counts.clear()
        native_probs = torch.softmax(native_logits, dim=1)
        if args.count_dispatch:
            payload["native_torch"]["dispatch_counts"] = count_dispatch(forward)
            call_counts.clear()
    except Exception:
        payload["native_torch"] = {"error": traceback.format_exc(limit=8)}
        write_json(out_dir / "petvision_forward.json", payload)
        raise

    reference_logits = native_logits
    if args.reference_logits:
        ref_payload = json.loads(Path(args.reference_logits).read_text(encoding="utf-8"))
        reference_logits = torch.tensor(ref_payload["logits"], dtype=torch.float32)
        payload["native_torch"]["reference_logits_file"] = str(args.reference_logits)

    payload["native_torch"]["summary"] = topk_summary(native_logits, class_names)
    payload["native_torch"]["compared_to"] = (
        "external reference logits" if args.reference_logits else "same-process native PyTorch"
    )
    if args.reference_logits:
        payload["native_torch"]["metrics"] = logits_metrics(
            native_logits, reference_logits, args.atol, args.rtol
        )
        if args.batch > 1:
            payload["native_torch"]["metrics"]["batch_note"] = (
                "allclose/logits compare all rows flattened; top1_* are computed "
                "on the flattened tensor (legacy field semantics)"
            )
    else:
        payload["native_torch"]["metrics"] = {
            "note": "no --reference-logits given; this run is its own reference"
        }
        payload["native_torch"]["top1_index"] = int(native_logits.argmax())

    if args.save_logits:
        logits_path = Path(args.save_logits)
        write_json(
            logits_path,
            {
                "kind": "petvision_logits",
                "image_size": image_size,
                "batch": args.batch,
                "seed": args.seed,
                "input": payload["input"],
                "logits": [float(v) for v in native_logits.detach().cpu().reshape(-1)],
            },
        )
        payload["native_torch"]["logits_file"] = str(logits_path.resolve())

    if args.backend == "flag_gems":
        import flag_gems

        payload["vendor"] = flag_gems.vendor_name
        payload["impl"] = {
            "flag_gems_version": getattr(flag_gems, "__version__", None),
            "conv2d_module": getattr(flag_gems.conv2d, "__module__", None),
            "mean_dim_module": getattr(flag_gems.mean_dim, "__module__", None),
            "softmax_module": getattr(flag_gems.softmax, "__module__", None),
        }
        if args.require_vendor and flag_gems.vendor_name != args.require_vendor:
            payload["error"] = (
                f"vendor mismatch: {flag_gems.vendor_name!r} != {args.require_vendor!r}"
            )
            write_json(out_dir / "petvision_forward.json", payload)
            raise SystemExit(payload["error"])

        dispatch_log = out_dir / "dispatch.log"
        if args.enable_ops:
            scope = [tok.strip() for tok in args.enable_ops.split(",") if tok.strip()]
        else:
            scope = ["conv2d"] if args.enable_scope == "conv2d" else None
        payload["config"]["registered_scope"] = scope if scope is not None else "all"
        softmax_only = scope is not None and set(scope) == {"softmax"}
        if softmax_only:
            # The model logits are already produced by the native path above.
            # The only selected operator is the terminal 37-class softmax, so
            # do not run/record/time a FlagGems model forward: register softmax
            # just around the probability computation and reuse native logits.
            softmax_kernel_runs = None
            try:
                with flag_gems.use_gems(include=["softmax"], record=False):
                    torch.softmax(native_logits, dim=1)  # warm autotuning/compile
                    softmax_hook = count_kernel(
                        flag_gems.softmax, "softmax_kernel_spacemit"
                    )
                    try:
                        gem_probs = torch.softmax(native_logits, dim=1)
                    finally:
                        target, original, calls = softmax_hook
                        target.run = original
                        softmax_kernel_runs = len(calls)
                probs_diff = (gem_probs.detach() - native_probs.detach()).abs()
                payload["flag_gems"] = {
                    "device": str(device),
                    "model_path": "native_torch logits reused; no FlagGems model forward",
                    "conv2d_calls_total": payload["native_torch"].get(
                        "conv2d_calls_total"
                    ),
                    "conv2d_calls_per_forward": payload["native_torch"].get(
                        "conv2d_calls_per_forward"
                    ),
                    # Same meaning as the full-forward path: logits vs the
                    # external reference (or the native logits themselves).
                    "metrics": logits_metrics(
                        native_logits, reference_logits, args.atol, args.rtol
                    ),
                    "summary": topk_summary(native_logits, class_names),
                    "probabilities": {
                        "softmax_kernel_runs": softmax_kernel_runs,
                        "max_abs_err_vs_native": float(probs_diff.max().item()),
                        "allclose": bool(
                            torch.allclose(
                                gem_probs,
                                native_probs,
                                atol=args.atol,
                                rtol=args.rtol,
                            )
                        ),
                        "top1": topk_summary_from_probs(gem_probs, class_names),
                        "top1_agree": int(gem_probs.argmax())
                        == int(native_probs.argmax()),
                    },
                    "timing": {
                        "note": "softmax-only: model time is native_torch.timing; "
                        "post-processing is not re-timed here (see k1_ops_check / A/B scripts)"
                    },
                }
            except Exception:
                payload["flag_gems"] = {"error": traceback.format_exc(limit=12)}
            payload["dispatch_log_tail"] = []
        else:
            mean_kernel_runs = None
            softmax_kernel_runs = None
            gem_conv_calls = None
            try:
                # Trace/dispatch evidence only: record=True writes a log for the
                # FlagGems side, so timing is done in a separate clean context.
                with flag_gems.use_gems(include=scope, record=True, path=str(dispatch_log)):
                    if scope is None or "conv2d" in scope:
                        payload["dispatch_probe"] = vendor_dispatch_probe(device)
                    kernel_hook = None
                    call_counts.clear()
                    # Warm up autotuning/compilation before counting dispatches.
                    with torch.no_grad():
                        forward()
                    call_counts.clear()
                    if scope is None or "mean_dim" in scope:
                        kernel_hook = count_kernel(flag_gems.mean_dim, "mean_dim_kernel")
                    try:
                        with torch.no_grad():
                            gem_logits = forward()
                        gem_conv_calls = call_counts.get("conv2d_calls", 0)
                        call_counts.clear()
                    finally:
                        if kernel_hook is not None:
                            target, original, calls = kernel_hook
                            target.run = original
                            mean_kernel_runs = len(calls)
                payload["dispatch_log_tail"] = read_log_tail(dispatch_log, limit=60)
                payload["flag_gems"] = {
                    "device": str(device),
                    "conv2d_calls_total": gem_conv_calls,
                    "conv2d_calls_per_forward": gem_conv_calls,
                    "metrics": logits_metrics(
                        gem_logits, reference_logits, args.atol, args.rtol
                    ),
                    "summary": topk_summary(gem_logits, class_names),
                }
                if mean_kernel_runs is not None:
                    payload["flag_gems"]["mean_dim_kernel_runs_per_forward"] = mean_kernel_runs
                if scope is None or "softmax" in scope:
                    # Real classification post-processing: logits -> 37 probabilities.
                    # A dedicated registered context keeps the counted call free of
                    # the dispatch-recording wrapper used above.
                    with flag_gems.use_gems(include=["softmax"], record=False):
                        torch.softmax(gem_logits, dim=1)  # warm autotuning/compile
                        softmax_hook = count_kernel(
                            flag_gems.softmax, "softmax_kernel_spacemit"
                        )
                        try:
                            gem_probs = torch.softmax(gem_logits, dim=1)
                        finally:
                            target, original, calls = softmax_hook
                            target.run = original
                            softmax_kernel_runs = len(calls)
                    probs_diff = (gem_probs.detach() - native_probs.detach()).abs()
                    payload["flag_gems"]["probabilities"] = {
                        "softmax_kernel_runs": softmax_kernel_runs,
                        "max_abs_err_vs_native": float(probs_diff.max().item()),
                        "allclose": bool(
                            torch.allclose(gem_probs, native_probs, atol=args.atol, rtol=args.rtol)
                        ),
                        "top1": topk_summary_from_probs(gem_probs, class_names),
                        "top1_agree": int(gem_probs.argmax()) == int(native_probs.argmax()),
                    }
                # Same registered ops, no record/logging, one warmup then timing.
                with flag_gems.use_gems(include=scope, record=False):
                    with torch.no_grad():
                        forward()
                    payload["flag_gems"]["timing"] = time_calls(
                        forward, args.warmup, args.repeat, device
                    )
                    if args.count_dispatch:
                        payload["flag_gems"]["dispatch_counts"] = count_dispatch(forward)
            except Exception:
                payload["flag_gems"] = {"error": traceback.format_exc(limit=12)}
            payload["dispatch_log_tail"] = read_log_tail(dispatch_log, limit=60)

    stat_after = weights_path.stat()
    payload["weights"]["mtime_ns_after"] = stat_after.st_mtime_ns
    payload["weights"]["unchanged"] = (
        stat_after.st_size == stat_before.st_size
        and stat_after.st_mtime_ns == stat_before.st_mtime_ns
    )

    checks = {"native_forward_ok": "error" not in payload.get("native_torch", {})}
    native_metrics = payload.get("native_torch", {}).get("metrics", {})
    if args.reference_logits:
        checks["reference_allclose"] = bool(native_metrics.get("allclose", False))
        checks["reference_top1_agree"] = bool(native_metrics.get("top1_agree", False))
    if args.backend == "flag_gems":
        gems = payload.get("flag_gems", {})
        checks["flag_gems_forward_ok"] = "error" not in gems
        checks["flag_gems_allclose"] = bool(gems.get("metrics", {}).get("allclose", False))
        checks["flag_gems_top1_agree"] = bool(
            gems.get("metrics", {}).get("top1_agree", False)
        )
        registered = payload["config"].get("registered_scope")
        if isinstance(registered, list) and "mean_dim" in registered:
            checks["mean_dim_kernel_invoked"] = bool(
                gems.get("mean_dim_kernel_runs_per_forward", 0)
            )
        if isinstance(registered, list) and "softmax" in registered:
            probs = gems.get("probabilities", {})
            checks["softmax_kernel_invoked_once"] = (
                probs.get("softmax_kernel_runs") == 1
            )
            checks["softmax_probs_match"] = bool(probs.get("allclose", False))
            checks["softmax_top1_agree"] = bool(probs.get("top1_agree", False))
    payload["checks"] = checks
    payload["passed"] = all(checks.values())

    json_path = write_json(out_dir / "petvision_forward.json", payload)
    print(f"wrote {json_path}")
    native = payload["native_torch"]
    native_err = native["metrics"].get("max_abs_err")
    native_err_text = "self-reference" if native_err is None else f"{native_err:.3e}"
    print(
        "  native torch: "
        f"{native['timing']['median_ms']:.2f} ms median, "
        f"{native['conv2d_calls_per_forward']:.0f} conv2d calls/forward, "
        f"max_abs_err={native_err_text}"
    )
    summary = native.get("summary", {})
    print(
        f"  top1: {summary.get('top1_class')} "
        f"(p={summary.get('top1_probability', 0):.6f})"
    )
    if args.backend == "flag_gems":
        gems = payload.get("flag_gems", {})
        if "error" in gems:
            print(f"  flag_gems: ERROR {gems['error'].splitlines()[-1]}")
        elif "median_ms" in gems.get("timing", {}):
            print(
                "  flag_gems: "
                f"{gems['timing']['median_ms']:.2f} ms median, "
                f"{gems['conv2d_calls_per_forward']:.0f} conv2d calls/forward, "
                f"max_abs_err={gems['metrics']['max_abs_err']:.3e}, "
                f"top1_agree={gems['metrics']['top1_agree']}"
            )
        else:
            probs = gems.get("probabilities", {})
            print(
                "  flag_gems: softmax-only on native logits; "
                f"kernel_runs={probs.get('softmax_kernel_runs')}, "
                f"probs_max_abs_err={probs.get('max_abs_err_vs_native'):.3e}, "
                f"top1_agree={probs.get('top1_agree')}; "
                "model time = native_torch.timing"
            )
    if not payload["passed"]:
        print(f"  failed checks: {[k for k, v in payload['checks'].items() if not v]}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())