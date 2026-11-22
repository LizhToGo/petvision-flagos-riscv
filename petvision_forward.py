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
import json
import sys
import traceback
from pathlib import Path

import torch
from torch import nn

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


def main(argv=None):
    args = parse_args(argv)
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
            "batch": args.batch,
            "image_size_arg": args.image_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
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

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(
        (args.batch, 3, image_size, image_size),
        generator=generator,
        dtype=torch.float32,
    )
    inputs = inputs.to(device)
    payload["input"] = {"shape": list(inputs.shape), "seed": args.seed, "device": str(device)}

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
    except Exception:
        payload["native_torch"] = {"error": traceback.format_exc(limit=8)}
        write_json(out_dir / "petvision_forward.json", payload)
        raise

    reference_logits = native_logits
    if args.reference_logits:
        ref_payload = json.loads(Path(args.reference_logits).read_text(encoding="utf-8"))
        reference_logits = torch.tensor(ref_payload["logits"], dtype=torch.float32)
        payload["native_torch"]["reference_logits_file"] = str(args.reference_logits)

    payload["native_torch"]["compared_to"] = (
        "external reference logits" if args.reference_logits else "same-process native PyTorch"
    )
    if args.reference_logits:
        payload["native_torch"]["metrics"] = logits_metrics(
            native_logits, reference_logits, args.atol, args.rtol
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
        }
        if args.require_vendor and flag_gems.vendor_name != args.require_vendor:
            payload["error"] = (
                f"vendor mismatch: {flag_gems.vendor_name!r} != {args.require_vendor!r}"
            )
            write_json(out_dir / "petvision_forward.json", payload)
            raise SystemExit(payload["error"])

        dispatch_log = out_dir / "dispatch.log"
        scope = ["conv2d"] if args.enable_scope == "conv2d" else None
        try:
            with flag_gems.use_gems(include=scope, record=True, path=str(dispatch_log)):
                payload["dispatch_probe"] = vendor_dispatch_probe(device)
                with torch.no_grad():
                    gem_logits = forward()
                payload["flag_gems"] = {
                    "timing": time_calls(
                        forward, args.warmup, args.repeat, device
                    ),
                    "device": str(device),
                    "conv2d_calls_total": call_counts.get("conv2d_calls", 0),
                    "conv2d_calls_per_forward": call_counts.get("conv2d_calls", 0)
                    / (1 + args.warmup + args.repeat),
                    "metrics": logits_metrics(
                        gem_logits, reference_logits, args.atol, args.rtol
                    ),
                }
        except Exception:
            payload["flag_gems"] = {"error": traceback.format_exc(limit=12)}
        payload["dispatch_log_tail"] = read_log_tail(dispatch_log, limit=60)

    stat_after = weights_path.stat()
    payload["weights"]["mtime_ns_after"] = stat_after.st_mtime_ns
    payload["weights"]["unchanged"] = (
        stat_after.st_size == stat_before.st_size
        and stat_after.st_mtime_ns == stat_before.st_mtime_ns
    )

    ok = True
    if "error" in payload.get("native_torch", {}):
        ok = False
    if args.backend == "flag_gems":
        gems = payload.get("flag_gems", {})
        ok = ok and "error" not in gems and gems.get("metrics", {}).get("allclose", False)
    payload["passed"] = bool(ok)

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
    if args.backend == "flag_gems":
        gems = payload.get("flag_gems", {})
        if "error" in gems:
            print(f"  flag_gems: ERROR {gems['error'].splitlines()[-1]}")
        else:
            print(
                "  flag_gems: "
                f"{gems['timing']['median_ms']:.2f} ms median, "
                f"{gems['conv2d_calls_per_forward']:.0f} conv2d calls/forward, "
                f"max_abs_err={gems['metrics']['max_abs_err']:.3e}, "
                f"top1_agree={gems['metrics']['top1_agree']}"
            )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())