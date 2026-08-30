"""Capture deterministic raster frames for a copied showcase camera pose."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SHADE_ROOT = ROOT.parent / "ordinaryshade"
if (SHADE_ROOT / "ordinaryshade").is_dir():
    sys.path.insert(0, str(SHADE_ROOT))

import ordinarylight as ol
from ordinarylight.integrations.workbench import discover_showcases
from ordinarylight.outputs import to_sdr


def _extent(value: str) -> tuple[int, int]:
    try:
        width, height = (int(item) for item in value.lower().split("x", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT") from error
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return width, height


def _pose(value: str) -> dict:
    candidate = Path(value)
    text = candidate.read_text() if candidate.is_file() else value
    result = json.loads(text)
    required = {"showcase", "position", "target"}
    if not required <= result.keys():
        raise ValueError(f"pose requires {sorted(required)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, help="copied JSON or JSON file")
    parser.add_argument(
        "--resolution", action="append", type=_extent,
        default=None, help="repeatable WIDTHxHEIGHT (default: 720p, 1080p, 4K)",
    )
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--ray-steps", type=int, default=24)
    parser.add_argument(
        "--diagnostic-channel", default="off",
        choices=(
            "off", "hit", "uv", "depth-delta", "confidence", "object-id",
            "depth-trace",
        ),
        help="replace optical shading with a raw SSR diagnostic channel",
    )
    parser.add_argument("--output", type=Path, default=Path("raster_pose_capture"))
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")

    pose = _pose(args.pose)
    catalog = ROOT / "ordinarylight" / "showcases" / "catalog"
    showcases = discover_showcases((catalog,))
    showcase = next(
        (item for item in showcases if item.id == pose["showcase"]), None,
    )
    if showcase is None:
        raise KeyError(f"unknown showcase {pose['showcase']!r}")
    scene = showcase.create_scene()
    camera = ol.PerspectiveCamera(
        pose["position"], pose["target"],
        up=pose.get("up", (0.0, 1.0, 0.0)),
        vertical_fov_degrees=float(pose.get("vertical_fov_degrees", 45.0)),
    )
    settings = dict(showcase.renderer)
    default_material = settings.get("material_program") or ol.builtin_material
    program = ol.RasterProgram.scene(
        target="spirv", validate=False,
        material_programs=scene.material_programs(default_material),
        material_modifier=settings.get(
            "material_modifier", settings.get("material_hook"),
        ),
    )
    settings.update(
        optical_quality="screen-space",
        screen_space_ray_steps=args.ray_steps,
        optical_debug_view=args.diagnostic_channel,
    )
    config = ol.RasterConfig(
        state=ol.RasterState(cull_mode="none"),
        ambient_light=float(settings.pop("ambient_light", 0.08)),
        **settings,
    )
    renderer = ol.renderers.raster.VulkanRasterRenderer(program, config=config)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "pose": pose, "frames": args.frames,
        "diagnostic_channel": args.diagnostic_channel, "captures": [],
    }
    first_sdr_frames = {}
    try:
        for width, height in args.resolution or (
            (1280, 720), (1920, 1080), (3840, 2160),
        ):
            frames = [
                renderer.render_frame(scene, camera, width, height)
                for _ in range(args.frames)
            ]
            reference = frames[0]
            hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
            maximum_error = max(
                float(np.max(np.abs(frame - reference))) for frame in frames[1:]
            )
            stem = f"{width}x{height}"
            first_sdr_frames[(width, height)] = to_sdr(reference)
            for index, frame in enumerate(frames):
                if args.diagnostic_channel != "off":
                    np.save(
                        args.output / f"{stem}-frame-{index:02d}.npy", frame,
                    )
                Image.fromarray(to_sdr(frame), "RGB").save(
                    args.output / f"{stem}-frame-{index:02d}.png",
                )
            report["captures"].append({
                "resolution": [width, height], "hashes": hashes,
                "unique_hashes": len(set(hashes)),
                "maximum_absolute_frame_difference": maximum_error,
                "changed_pixels": [
                    int(np.count_nonzero(np.any(frame != reference, axis=-1)))
                    for frame in frames[1:]
                ],
            })
    finally:
        renderer.close()
    reference_extent = max(
        first_sdr_frames, key=lambda extent: extent[0] * extent[1],
    )
    reference_image = Image.fromarray(
        first_sdr_frames[reference_extent], "RGB",
    )
    for capture in report["captures"]:
        extent = tuple(capture["resolution"])
        candidate = first_sdr_frames[extent].astype(np.float32) / 255.0
        resized_reference = np.asarray(reference_image.resize(
            extent, Image.Resampling.LANCZOS,
        ), np.float32) / 255.0
        capture["sdr_rmse_vs_largest_resolution"] = float(np.sqrt(
            np.mean((candidate - resized_reference) ** 2),
        ))
    report["cross_resolution_reference"] = list(reference_extent)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
