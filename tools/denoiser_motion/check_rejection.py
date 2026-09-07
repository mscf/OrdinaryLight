"""GPU contract checks for short-history motion policy; no ray tracer required."""

from dataclasses import replace
import json

import numpy as np

import ordinarylight as ol
from ordinarylight.targets.vulkan.core import _relax_temporal_policy
from .replay import ShaderReplay


def frame(index, left, *, light=1.0, cut=False):
    h, w = 24, 40
    radiance = np.full((h, w, 4), 0.1, np.float32)
    radiance[..., 3] = 1
    radiance[6:18, left : left + 8, :3] = light
    normal = np.zeros((h, w, 4), np.float32)
    normal[..., 2] = 1
    normal[..., 3] = 0.2
    depth = np.full((h, w), 2, np.float32)
    depth[6:18, left : left + 8] = 1
    material = np.zeros((h, w), np.uint32)
    material[6:18, left : left + 8] = 1
    motion = np.zeros((h, w, 2), np.float32)
    if index:
        motion[6:18, left : left + 8, 0] = -6
    return ol.DenoiserSignals(
        radiance,
        np.zeros_like(radiance),
        normal,
        depth,
        motion,
        material,
        ol.DenoiserFrameInfo(np.eye(4), np.eye(4), index, camera_cut=cut),
    )


def check():
    config = ol.RendererConfig(denoiser_motion_history_floor=3)
    policy = _relax_temporal_policy(config, 10)
    gpu = ShaderReplay(40, 24)
    checks = []
    try:
        first = frame(0, 4)
        gpu.denoise(first, first.view_z, policy)
        second = frame(1, 10)
        lobes, lengths = gpu.denoise(second, second.view_z, policy)
        accepted = gpu.last_acceptance[0]
        assert np.all(accepted[8:16, 5:9] == 0), (
            "newly revealed background retained foreground history"
        )
        assert np.all(lengths[0][8:16, 5:9] == 1)
        np.testing.assert_allclose(lobes[0][8:16, 5:9], 0.1, atol=0.001)
        assert np.all(accepted[8:16, 11:17] == 1), "moving surface lost valid history"
        assert np.all(lengths[0][8:16, 11:17] == 2)
        checks += [
            "disocclusion rejects old foreground",
            "moving surface retains valid history",
        ]
        # A light switches off on otherwise identical geometry/motion.
        third = frame(2, 10, light=0)
        third = replace(third, motion=np.zeros_like(third.motion))
        lobes, lengths = gpu.denoise(third, third.view_z, policy)
        assert np.all(gpu.last_acceptance[0][8:16, 12:16] == 0)
        np.testing.assert_allclose(lobes[0][8:16, 12:16], 0, atol=0.001)
        checks.append("lighting change rejects stale radiance")
        cut = replace(third, frame=replace(third.frame, camera_cut=True))
        gpu.denoise(cut, cut.view_z, policy)
        assert not np.any(gpu.last_acceptance)
        checks.append("camera cut rejects all history")
        for _ in range(6):
            _, lengths = gpu.denoise(third, third.view_z, policy)
        assert max(np.max(x) for x in lengths) == 3
        checks.append("validated fast-motion history remains bounded at three")
        # A dark one-pixel surface surrounded by a different bright surface.
        # Foreign geometry must not inflate the local mean/variance and clamp
        # this pixel toward the neighbor's light when its own light switches off.
        radiance = np.full_like(first.diffuse_radiance_hit_distance, 100)
        radiance[..., 3] = 1
        radiance[12, 20, :3] = 1
        material = np.zeros_like(first.material_id)
        material[12, 20] = 1
        boundary = replace(
            first,
            diffuse_radiance_hit_distance=radiance,
            material_id=material,
            motion=np.zeros_like(first.motion),
            frame=replace(first.frame, camera_cut=True),
        )
        gpu.denoise(boundary, boundary.view_z, policy)
        radiance = radiance.copy()
        radiance[12, 20, :3] = 0
        boundary = replace(
            boundary,
            diffuse_radiance_hit_distance=radiance,
            frame=replace(boundary.frame, camera_cut=False),
        )
        lobes, _ = gpu.denoise(boundary, boundary.view_z, policy)
        assert gpu.last_acceptance[0][12, 20] == 0
        np.testing.assert_allclose(lobes[0][12, 20], 0, atol=0.001)
        checks.append("foreign surface cannot contaminate boundary history clamp")

    finally:
        gpu.close()
    return checks


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
