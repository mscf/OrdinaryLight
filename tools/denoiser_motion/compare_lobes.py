"""Compare paired live captures from live_edges, including optics regions."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tests.gates.relax_motion_quality import _luminance
from tools.denoiser_motion.analyze_edges import analyze


def compare(baseline, candidate, output):
    configurations = [
        json.loads((path / "configuration.json").read_text())
        for path in (baseline, candidate)
    ]
    if configurations[0].get("evaluated_lobes") or not configurations[1].get("evaluated_lobes"):
        raise ValueError("expected disabled baseline and enabled candidate")
    if {k: v for k, v in configurations[0].items() if k != "evaluated_lobes"} != {
        k: v for k, v in configurations[1].items() if k != "evaluated_lobes"
    }:
        raise ValueError("capture configurations differ")
    reference = np.load(baseline / "reference.npy")
    other_reference = np.load(candidate / "reference.npy")
    # Require the same reference so reference noise cannot favor either mode.
    np.testing.assert_array_equal(reference, other_reference)
    report = {
        "configuration": configurations,
        "reference_image_exact": True,
        "baseline": analyze(baseline),
        "evaluated": analyze(candidate),
        "regions": {},
    }
    if configurations[0]["scene"] == "optics":
        from tools.denoiser_motion.run import fixture
        scene, _ = fixture()
        identities = np.stack([
            np.load(baseline / f"floor3-guides-{i:03d}.npz")["identity"][..., 0]
            for i in range(len(reference))
        ])
        ref_luma = _luminance(reference)
        offset = 0
        for mesh in scene.render_meshes:
            if mesh.name:
                mask = identities == offset
                values = {}
                for floor in (1, 3):
                    values[f"floor{floor}"] = {
                        name: float(np.sqrt(np.mean(
                            (_luminance(np.load(path / f"floor{floor}.npy"))[mask]
                             - ref_luma[mask]) ** 2
                        )))
                        for name, path in (("baseline", baseline), ("evaluated", candidate))
                    }
                report["regions"][mesh.name] = values
            offset += len(mesh.indices)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    height, width = reference.shape[1:3]
    canvas = Image.new("RGB", (width * 3, (height + 28) * 2))
    draw = ImageDraw.Draw(canvas)
    for row, floor in enumerate((1, 3)):
        arrays = [
            reference,
            np.load(baseline / f"floor{floor}.npy"),
            np.load(candidate / f"floor{floor}.npy"),
        ]
        labels = ("Independent reference", f"Baseline / floor {floor}", f"Evaluated / floor {floor}")
        for col, (frames, label) in enumerate(zip(arrays, labels)):
            rgb = np.maximum(frames[-1, ..., :3], 0)
            rgb = np.uint8(np.clip((rgb / (1 + rgb)) ** (1 / 2.2), 0, 1) * 255)
            y = row * (height + 28)
            canvas.paste(Image.fromarray(rgb), (col * width, y + 28))
            draw.text((col * width + 5, y + 7), label, fill="white")
    canvas.save(output / "comparison.png")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.output)


if __name__ == "__main__":
    main()
