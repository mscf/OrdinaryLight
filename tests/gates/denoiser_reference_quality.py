"""Evaluate captured denoiser signals against high-sample ground truth.

The gate is renderer-neutral.  NRD is optional and, when requested, is loaded
through the separately built ``ordinarylight_nrd`` module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.denoising.reference import NrdRelaxReference


def _metrics_payload(metrics):
    return None if metrics is None else asdict(metrics)


def _load_sequence(directory):
    paths = sorted(directory.glob("frame-*.npz"))
    if not paths:
        raise ValueError(f"no frame-*.npz captures found in {directory}")
    truth_path = directory / "ground_truth.npy"
    if not truth_path.is_file():
        raise ValueError(f"missing {truth_path}")
    return tuple(ol.DenoiserSignals.load(path) for path in paths), np.load(
        truth_path, allow_pickle=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-nrd", action="store_true")
    parser.add_argument("--accept-baseline", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args(argv)
    signals, truth = _load_sequence(args.capture)
    reference = None
    if args.with_nrd:
        reference = NrdRelaxReference()
        if not reference.available:
            raise RuntimeError(
                "--with-nrd requested, but ordinarylight_nrd is not installed"
            )
    result = ol.evaluate_denoiser_sequence(
        signals, truth, reference_denoiser=reference,
    )
    payload = {
        "schema": 1,
        "capture": str(args.capture),
        "frames": len(signals),
        "extent": list(signals[0].extent),
        "portable": _metrics_payload(result.portable),
        "reference": _metrics_payload(result.reference),
        "portable_against_reference": _metrics_payload(
            result.portable_against_reference
        ),
        "reference_implementation": result.reference_implementation,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")

    if args.accept_baseline:
        if args.baseline is None:
            raise ValueError("--accept-baseline requires --baseline")
        reason = os.environ.get("ORDINARYLIGHT_QUALITY_OVERRIDE_REASON")
        if not reason:
            raise ValueError(
                "accepting a baseline requires "
                "ORDINARYLIGHT_QUALITY_OVERRIDE_REASON"
            )
        baseline = ol.DenoiserQualityBaseline(
            args.scene or args.capture.name, result.portable, args.tolerance,
        )
        baseline.save(args.baseline)
        return 0
    if args.baseline:
        ol.DenoiserQualityBaseline.load(args.baseline).require(result.portable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
