"""Minimal Conv2d correctness + latency case for FlagGems backends.

Reference: float64 CPU convolution computed with native PyTorch.
Candidate: the active ``aten::conv2d`` implementation, either native PyTorch or
FlagGems after ``flag_gems.use_gems()`` has been entered.

The case set covers the parameters PetVision actually uses (3x3/stride 1/pad 1
inside U-Net and ResNet bottlenecks, the 7x7/stride 2/pad 3 ResNet-50 stem) plus
bias/no-bias, stride 2 and dilation 2 variants.

Usage examples:
    python conv2d_case.py --backend torch --out results/local_cpu
    GEMS_VENDOR=spacemit python conv2d_case.py --backend flag_gems \
        --require-vendor spacemit --out results/spacemit_emu
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from flagos_env import (
    collect_env,
    read_log_tail,
    resolve_device,
    tensor_metrics,
    time_calls,
    vendor_dispatch_probe,
    write_json,
)

# Deliberately small shapes plus one PetVision-realistic stem shape.  The
# small shapes keep an emulated RISC-V run in the seconds range.
CASES = (
    {
        "name": "unet_conv3x3_pad1_bias",
        "input": (1, 3, 32, 32),
        "weight": (8, 3, 3, 3),
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "bias": True,
    },
    {
        "name": "stride2_pad1_bias_odd_input",
        "input": (1, 4, 17, 15),
        "weight": (6, 4, 3, 3),
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "bias": True,
    },
    {
        "name": "valid_nopad_nobias_batch2",
        "input": (2, 3, 19, 19),
        "weight": (5, 3, 3, 3),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "bias": False,
    },
    {
        "name": "dilation2_pad2_bias",
        "input": (1, 3, 15, 15),
        "weight": (4, 3, 3, 3),
        "stride": 1,
        "padding": 2,
        "dilation": 2,
        "bias": True,
    },
    {
        "name": "petvision_resnet50_stem_7x7_s2_pad3",
        "input": (1, 3, 256, 256),
        "weight": (64, 3, 7, 7),
        "stride": 2,
        "padding": 3,
        "dilation": 1,
        "bias": False,
        "large": True,
    },
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "flag_gems"), default="torch")
    parser.add_argument(
        "--enable-scope",
        choices=("conv2d", "all"),
        default="conv2d",
        help="flag_gems.use_gems(include=['conv2d']) or the full op table",
    )
    parser.add_argument("--cases", default="", help="comma separated substrings")
    parser.add_argument("--skip-large", action="store_true", help="drop 256x256 stem")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument(
        "--device",
        default="auto",
        help="auto = flag_gems.device for --backend flag_gems, otherwise cpu",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--require-vendor", default="")
    parser.add_argument("--flaggems-root", default="")
    parser.add_argument("--out", required=True, help="output directory for JSON/logs")
    return parser.parse_args(argv)


def selected_cases(args):
    wanted = [item.strip() for item in args.cases.split(",") if item.strip()]
    cases = []
    for case in CASES:
        if args.skip_large and case.get("large"):
            continue
        if wanted and not any(token in case["name"] for token in wanted):
            continue
        cases.append(case)
    if not cases:
        raise SystemExit(f"no case matched --cases={args.cases!r}")
    return cases


def make_tensors(case, dtype, seed, device):
    # Inputs are always drawn on CPU so the values are identical on every device.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    inp = torch.randn(case["input"], generator=gen, dtype=torch.float32).to(dtype)
    weight = torch.randn(case["weight"], generator=gen, dtype=torch.float32).to(dtype)
    bias = None
    if case["bias"]:
        bias = torch.randn((case["weight"][0],), generator=gen, dtype=torch.float32).to(dtype)
    return inp.to(device), weight.to(device), bias if bias is None else bias.to(device)


def fp64_reference(case, inp, weight, bias):
    # The reference is always computed on CPU in float64.
    return F.conv2d(
        inp.detach().cpu().double(),
        weight.detach().cpu().double(),
        None if bias is None else bias.detach().cpu().double(),
        stride=case["stride"],
        padding=case["padding"],
        dilation=case["dilation"],
        groups=1,
    ).to(torch.float32)


def run_case(case, args, device):
    inp, weight, bias = make_tensors(case, getattr(torch, args.dtype), args.seed, device)
    ref = fp64_reference(case, inp, weight, bias)

    def call():
        return F.conv2d(
            inp,
            weight,
            bias,
            stride=case["stride"],
            padding=case["padding"],
            dilation=case["dilation"],
        )

    record = {
        "name": case["name"],
        "input_shape": list(case["input"]),
        "weight_shape": list(case["weight"]),
        "stride": case["stride"],
        "padding": case["padding"],
        "dilation": case["dilation"],
        "bias": case["bias"],
        "dtype": args.dtype,
        "device": str(device),
        "output_shape": list(ref.shape),
    }
    with torch.no_grad():
        out = call()
        record["native_torch"] = {
            "metrics": tensor_metrics(out, ref, args.atol, args.rtol),
            "timing": time_calls(call, args.warmup, args.repeat, device),
        }
    return record


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.backend, args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "kind": "conv2d_case",
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": collect_env(args.flaggems_root),
        "config": {
            "backend": args.backend,
            "device": str(device),
            "enable_scope": args.enable_scope,
            "cases": [c["name"] for c in selected_cases(args)],
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "cases": [],
    }

    for case in selected_cases(args):
        try:
            payload["cases"].append(run_case(case, args, device))
        except Exception:
            payload["cases"].append(
                {"name": case["name"], "error": traceback.format_exc(limit=6)}
            )

    if args.backend == "flag_gems":
        import flag_gems

        vendor = flag_gems.vendor_name
        payload["vendor"] = vendor
        payload["impl"] = {
            "flag_gems_version": getattr(flag_gems, "__version__", None),
            "flag_gems_device": str(flag_gems.device),
            "conv2d_module": getattr(flag_gems.conv2d, "__module__", None),
            "conv2d_name": getattr(flag_gems.conv2d, "__name__", None),
        }
        if args.require_vendor and vendor != args.require_vendor:
            payload["error"] = (
                f"vendor mismatch: flag_gems.vendor_name={vendor!r}, "
                f"--require-vendor={args.require_vendor!r}"
            )
            write_json(out_dir / "conv2d.json", payload)
            raise SystemExit(payload["error"])

        dispatch_log = out_dir / "dispatch.log"
        scope = ["conv2d"] if args.enable_scope == "conv2d" else None
        with flag_gems.use_gems(include=scope, record=True, path=str(dispatch_log)):
            payload["dispatch_probe"] = vendor_dispatch_probe(device)
            payload["impl"]["active_device"] = str(device)
            for record in payload["cases"]:
                case = next(c for c in CASES if c["name"] == record["name"])
                if "error" in record:
                    continue
                inp, weight, bias = make_tensors(
                    case, getattr(torch, args.dtype), args.seed, device
                )
                ref = fp64_reference(case, inp, weight, bias)

                def call():
                    return F.conv2d(
                        inp,
                        weight,
                        bias,
                        stride=case["stride"],
                        padding=case["padding"],
                        dilation=case["dilation"],
                    )

                try:
                    with torch.no_grad():
                        out = call()
                        direct = flag_gems.conv2d(
                            inp,
                            weight,
                            bias,
                            padding=case["padding"],
                            stride=case["stride"],
                            dilation=case["dilation"],
                        )
                        record["backend"] = {
                            "metrics": tensor_metrics(out, ref, args.atol, args.rtol),
                            "direct_call_max_abs_diff": float(
                                (out.detach().cpu() - direct.detach().cpu()).abs().max()
                            ),
                            "timing": time_calls(
                                call, args.warmup, args.repeat, device
                            ),
                        }
                except Exception:
                    record["backend"] = {"error": traceback.format_exc(limit=8)}
        payload["dispatch_log_tail"] = read_log_tail(dispatch_log)

    failures = []
    for record in payload["cases"]:
        if "error" in record:
            failures.append(f"{record['name']}: {record['error'].splitlines()[-1]}")
            continue
        check = record.get("backend") or record["native_torch"]
        if "error" in check:
            failures.append(f"{record['name']}: {check['error'].splitlines()[-1]}")
        elif not check["metrics"]["allclose"]:
            failures.append(
                f"{record['name']}: max_abs_err={check['metrics']['max_abs_err']:.3e}"
            )

    payload["passed"] = not failures
    payload["failures"] = failures
    json_path = write_json(out_dir / "conv2d.json", payload)
    print(f"wrote {json_path}")
    for record in payload["cases"]:
        status = "ERROR"
        detail = ""
        if "error" not in record:
            check = record.get("backend") or record["native_torch"]
            if "error" in check:
                detail = check["error"].splitlines()[-1]
            else:
                metrics = check["metrics"]
                timing = check["timing"]
                status = "PASS" if metrics["allclose"] else "FAIL"
                detail = (
                    f"max_abs_err={metrics['max_abs_err']:.3e} "
                    f"median={timing['median_ms']:.3f}ms"
                )
        else:
            detail = record["error"].splitlines()[-1]
        print(f"  [{status}] {record['name']}: {detail}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())