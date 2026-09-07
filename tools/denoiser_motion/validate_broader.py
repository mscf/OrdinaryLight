"""Live broader-filter validation: object motion, camera stops, textured detail."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.showcases.rooms import (
    animate_object_motion_room, build_object_motion_room, build_textured_room,
)
from tests.gates.relax_motion_quality import _camera, _config, _gradient, _luminance, evaluate_sequence
from tools.denoiser_motion.live_edges import read_guides


PHASES = [0.0] * 3 + [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 7


def summarize(path):
    ref = np.load(path / "reference.npy")
    split = np.load(path / "reference-split.npy")
    ref_luma = _luminance(ref)
    guides = [np.load(path / f"guides-{i:03d}.npz") for i in range(len(ref))]
    identities = np.stack([g["identity"][..., 0] for g in guides])
    kind = json.loads((path / "configuration.json").read_text())["kind"]
    # Use the fixture's packed triangle range, not normal sign: the floor's
    # stored geometric normal can point downward.
    floor = identities == (8 if kind == "detail" else 0)
    ref_gradient = np.stack([_gradient(frame) for frame in ref_luma])
    detail = floor & (ref_gradient > 0.02)
    ref_dx = np.diff(ref_luma, axis=2, append=ref_luma[:, :, -1:])
    ref_dy = np.diff(ref_luma, axis=1, append=ref_luma[:, -1:, :])
    disoccluded = np.zeros_like(identities, bool)
    if kind == "object":
        # Fixture packs four room quads and one emitter before moving sphere.
        disoccluded[1:] = (identities[:-1] == 10) & (identities[1:] != 10)
    revealed = disoccluded.any(axis=0) & (identities[-1] != 10)
    report = {
        "kind": kind,
        "reference_split_log_luma_rmse": float(np.sqrt(np.mean(
            (_luminance(split[:, 0]) - _luminance(split[:, 1])) ** 2
        ))),
        "disoccluded_pixel_count": int(disoccluded.sum()),
        "detail_pixel_count": int(detail.sum()),
        "metrics": {},
    }
    for name in ("default", "broader"):
        frames = np.load(path / f"{name}.npy")
        luma = _luminance(frames)
        error = luma - ref_luma
        gradient = np.stack([_gradient(frame) for frame in luma])
        dx = np.diff(luma, axis=2, append=luma[:, :, -1:])
        dy = np.diff(luma, axis=1, append=luma[:, -1:, :])
        report["metrics"][name] = {
            "whole_sequence": evaluate_sequence(ref, frames),
            "moving": evaluate_sequence(ref[3:8], frames[3:8]),
            "stopped": evaluate_sequence(ref[8:], frames[8:]),
            "per_frame_log_luma_rmse": [
                float(np.sqrt(np.mean(frame ** 2))) for frame in error
            ],
            "disocclusion_log_luma_rmse": (
                float(np.sqrt(np.mean(error[disoccluded] ** 2)))
                if disoccluded.any() else None
            ),
            "stopped_revealed_log_luma_rmse": (
                float(np.sqrt(np.mean(error[8:, revealed] ** 2)))
                if revealed.any() else None
            ),
            "stopped_revealed_per_frame_rmse": (
                [float(np.sqrt(np.mean(frame[revealed] ** 2))) for frame in error[8:]]
                if revealed.any() else None
            ),
            "floor_detail_gradient_rmse": (
                float(np.sqrt(np.mean((gradient[detail] - ref_gradient[detail]) ** 2)))
                if detail.any() else None
            ),
            "floor_detail_signed_gradient_gain": (
                float(np.sum((dx * ref_dx + dy * ref_dy)[detail])
                      / np.sum((ref_dx ** 2 + ref_dy ** 2)[detail]))
                if detail.any() else None
            ),
            "floor_detail_signed_gain_by_phase": {
                phase: (
                    float(np.sum((dx * ref_dx + dy * ref_dy)[indices][detail[indices]])
                          / np.sum((ref_dx ** 2 + ref_dy ** 2)[indices][detail[indices]]))
                    if detail[indices].any() else None
                )
                for phase, indices in (("moving", slice(3, 8)), ("stopped", slice(8, None)))
            },
            "floor_detail_gradient_ratio": (
                float(gradient[detail].mean() / ref_gradient[detail].mean())
                if detail.any() else None
            ),
        }
    (path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for guide in guides:
        guide.close()
    return report


def capture(output, kind):
    settings = SimpleNamespace(
        width=320, height=180, frames=len(PHASES), reference_samples=64,
        bounces=8, atrous_iterations=3, history_floor=3,
    )
    factory = build_textured_room if kind == "detail" else build_object_motion_room

    def pose(scene, phase):
        if kind == "object":
            animate_object_motion_room(scene, 1.5 * phase)
            return _camera()
        return _camera(math.pi + 0.25 * (phase - 0.5))

    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(320, 180, "Broader filter validation", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, weight in (("default", 4.0), ("broader", 2.0)):
            scene = factory()
            config = replace(
                _config(settings, reference=False),
                denoiser_signal_capture=True, denoiser_color_weight=weight,
                denoiser_sampled_indirect=False,
            )
            frames = []
            with ol.VulkanGlfwPresenter(window, config=config) as presenter:
                for index, phase in enumerate(PHASES):
                    camera = pose(scene, phase)
                    presenter.present_wavefront(scene, camera, 320, 180)
                    frames.append(presenter.capture_wavefront_hdr())
                    if name == "default":
                        np.savez_compressed(
                            output / f"guides-{index:03d}.npz",
                            **read_guides(presenter._core),
                        )
            np.save(output / f"{name}.npy", frames)
            print(f"{kind}: captured {name}", flush=True)
        scene = factory()
        cache = {}
        with ol.VulkanGlfwPresenter(
            window, config=_config(settings, reference=True),
        ) as presenter:
            for index, phase in enumerate(sorted(set(PHASES))):
                camera = pose(scene, phase)
                batches = []
                for batch in range(8):
                    presenter._core.wavefront_frame_sequence = 1000 + index * 8 + batch
                    presenter.present_wavefront(scene, camera, 320, 180)
                    batches.append(presenter.capture_wavefront_hdr()[..., :3])
                cache[phase] = (
                    np.mean(batches, axis=0),
                    np.array([np.mean(batches[::2], axis=0), np.mean(batches[1::2], axis=0)]),
                )
                print(f"{kind}: reference {index + 1}/6", flush=True)
        np.save(output / "reference.npy", [cache[p][0] for p in PHASES])
        np.save(output / "reference-split.npy", [cache[p][1] for p in PHASES])
        (output / "configuration.json").write_text(json.dumps({
            "kind": kind, "width": 320, "height": 180, "phases": PHASES,
            "history_floor": 3, "evaluated_lobes": False, "color_weights": [4, 2],
            "reference_samples": 512, "reference_seed_offset": 1000,
            "reference_reused_for_identical_pose": True,
        }, indent=2) + "\n")
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("object", "camera", "detail"), required=True)
    args = parser.parse_args()
    capture(args.output, args.kind)


if __name__ == "__main__":
    main()
