#!/usr/bin/env python3
"""Alternating native / vendor-softmax A/B on the real PetVision checkpoint.

Loads the model and image once, warms both paths, then runs ``--pairs``
alternating rounds (order flips each pair).  Model-forward and 37-class softmax
milliseconds are recorded separately.  No dispatch hooks, no record=True, so the
samples are the same code path used for timing in the report.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import time
from pathlib import Path

import torch

from petvision_forward import image_input, load_model


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return (time.perf_counter() - start) * 1e3, out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    import flag_gems

    model, checkpoint, class_names = load_model(Path(args.weights))
    model.eval()
    image_size = int(checkpoint.get("image_size", 256))
    tensor, meta = image_input(Path(args.image), image_size)
    inputs = tensor.unsqueeze(0)

    def native_pair():
        with torch.no_grad():
            model_ms, logits = timed(lambda: model(inputs))
        softmax_ms, probs = timed(lambda: torch.softmax(logits, dim=1))
        return model_ms, softmax_ms, probs

    def vendor_pair():
        with flag_gems.use_gems(include=["softmax"], record=False):
            with torch.no_grad():
                model_ms, logits = timed(lambda: model(inputs))
            softmax_ms, probs = timed(lambda: torch.softmax(logits, dim=1))
        return model_ms, softmax_ms, probs

    with torch.no_grad():
        model(inputs)
    with flag_gems.use_gems(include=["softmax"], record=False):
        with torch.no_grad():
            model(inputs)
        torch.softmax(torch.randn(1, 37), dim=1)

    samples = []
    ref_probs = None
    for i in range(args.pairs):
        if i % 2 == 0:
            n_model, n_soft, n_probs = native_pair()
            v_model, v_soft, v_probs = vendor_pair()
            order = "native_first"
        else:
            v_model, v_soft, v_probs = vendor_pair()
            n_model, n_soft, n_probs = native_pair()
            order = "vendor_first"
        if ref_probs is None:
            ref_probs = n_probs
        samples.append(
            {
                "pair": i,
                "order": order,
                "native_model_ms": n_model,
                "vendor_model_ms": v_model,
                "native_softmax_ms": n_soft,
                "vendor_softmax_ms": v_soft,
                "probs_max_abs_diff": float((n_probs - v_probs).abs().max().item()),
            }
        )

    def med(key):
        return statistics.median(s[key] for s in samples)

    payload = {
        "schema": 1,
        "kind": "k1_alternating_ab",
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "threads": torch.get_num_threads(),
            "pairs": args.pairs,
            "image": meta,
            "model": "PetVision ResNet-50 37-class",
            "registered_scope": ["softmax"],
            "hooks": False,
            "record": False,
        },
        "samples": samples,
        "medians": {
            "native_model_ms": med("native_model_ms"),
            "vendor_model_ms": med("vendor_model_ms"),
            "native_softmax_ms": med("native_softmax_ms"),
            "vendor_softmax_ms": med("vendor_softmax_ms"),
            "max_probs_diff": max(s["probs_max_abs_diff"] for s in samples),
        },
        "note": (
            "The model forward is native ATen in both arms; only the 37-class "
            "softmax differs. Model-forward deltas are not attributable to the "
            "operator."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["medians"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
