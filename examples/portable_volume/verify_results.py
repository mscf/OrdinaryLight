"""Compare browser_probe.mjs captures with native execution of exported packages."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ordinarylight.portable import PortablePackage
from ordinaryscience.rt_density import rt_histogram_reference
from ordinaryscience.vector_volume import reprepare_volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Demo output directory containing browser.json"
    )
    args = parser.parse_args()
    root = args.directory
    browser = json.loads((root / "browser.json").read_text())
    required = {
        "render", "live update", "presentation independence", "same-runtime restore",
        "fresh-runtime restore", "invalid restore atomicity", "structural guard",
        "same-device replacement", "failed replacement recovery",
    }
    if not required.issubset(browser["checks"]) or browser["gpuErrors"]:
        raise ValueError("Browser checks are incomplete or GPU errors were reported")
    if set(browser.get("rejectionChecks", [])) != {
        "unsupported schema", "corrupt payload", "missing feature", "excessive limit"
    }:
        raise ValueError("Package rejection checks are incomplete")
    initial = PortablePackage.read(root / "initial")
    updated = reprepare_volume(
        initial, browser["snapshot"], {"parameters": {"seed": 9}}
    )
    replacement = PortablePackage.read(
        root / "packages" / browser["replacementSnapshot"]["package_id"]
    )
    report = dict(
        browser_checks=browser["checks"],
        rejection_checks=browser["rejectionChecks"],
        browser_adapter=browser.get("browserAdapter"),
        gpu_errors=browser["gpuErrors"],
        comparisons={},
    )
    for label, package in [
        ("initial", initial),
        ("updated", updated),
        ("replacement", replacement),
    ]:
        manifest = package.manifest
        resources = {r["id"]: r for r in manifest["resources"]}
        with package.native() as runtime:
            runtime.dispatch()
            native = {
                name: runtime.read(resource)
                for name, resource in manifest["outputs"].items()
            }
            report.setdefault("native_adapter", dict(runtime.device.adapter.info))
        comparison = {}
        for name, actual in native.items():
            description = resources[manifest["outputs"][name]]
            dtype = {"f32": "<f4", "i32": "<i4", "u32": "<u4"}[description["dtype"]]
            expected = np.frombuffer(bytes(browser[label][name]), dtype=dtype).reshape(
                description["shape"]
            )
            np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
            if name in ("density", "excluded"):
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(
                    actual, expected, atol=1e-5, rtol=1e-5, equal_nan=True
                )
            finite = np.isfinite(actual)
            comparison[name] = dict(
                elements=actual.size,
                nonfinite=int((~finite).sum()),
                max_absolute_error=float(
                    np.max(
                        np.abs(
                            actual[finite].astype(float)
                            - expected[finite].astype(float)
                        ),
                        initial=0,
                    )
                ),
            )
        science = manifest["science"]
        reference, excluded = rt_histogram_reference(
            native[science["output"]],
            science["edges"],
            (len(science["y"]["samples"]), len(science["x"]["samples"])),
        )
        np.testing.assert_array_equal(native["density"], reference)
        np.testing.assert_array_equal(native["excluded"], excluded)
        assert reference.sum() + excluded.sum() == native[science["output"]].size
        report["comparisons"][label] = comparison
    pixels = np.asarray(browser["pixels"], dtype=np.uint8).reshape(
        browser["height"], browser["width"], 4
    )
    Image.fromarray(pixels).save(root / "browser-volume.png")
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
