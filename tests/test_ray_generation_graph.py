"""Primary rays, tile indexing and cold-state ABI without a renderer or scene."""

from contextlib import ExitStack
import os
import struct
import numpy as np
import pytest
from ordinarylight.runtime import VulkanRuntime, VulkanRayGeneration
from ordinarylight.wavefront import RAY_DTYPE, HOT_PATH_STATE_DTYPE


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_tiled_primary_rays_and_state():
    with VulkanRuntime() as runtime, ExitStack() as stack:
        jitter = struct.unpack("f", struct.pack("I", 0x38003800))[0]
        camera = stack.enter_context(
            runtime.buffer(
                64,
                data=struct.pack(
                    "16f", 1, 2, 3, 4, 0, 0, -1, 0, 1, 0, 0, jitter, 0, 1, 0, 0
                ),
            )
        )
        rays = stack.enter_context(runtime.buffer(16 + 6 * 48))
        paths = stack.enter_context(runtime.buffer(6 * 48))
        media = stack.enter_context(runtime.buffer(6 * 64))
        stage = stack.enter_context(
            VulkanRayGeneration(
                runtime, camera=camera, rays=rays, paths=paths, media=media, capacity=6
            )
        )
        for capture in (False, True):
            stage.operation(
                extent=(7, 5),
                tile_origin=(2, 1),
                tile_extent=(3, 2),
                sample_index=1,
                sample_count=2,
                capture_secondary=capture,
            ).execute(runtime).wait()
            raw = rays.read()
            np.testing.assert_array_equal(
                np.frombuffer(raw, np.uint32, count=3), [6, 6, 0]
            )
            result = np.frombuffer(raw, RAY_DTYPE, count=6, offset=16)
            np.testing.assert_array_equal(result["path_index"], np.arange(6))
            np.testing.assert_allclose(
                result["origin_tmin"], np.tile([1, 2, 3, 0.001], (6, 1))
            )
            pixels = np.array([(x, y) for y in range(1, 3) for x in range(2, 5)])
            ndc = 2 * (pixels + 0.5) / [7, 5] - 1
            expected = np.column_stack([ndc[:, 0] * 7 / 5, -ndc[:, 1], -np.ones(6)])
            expected /= np.linalg.norm(expected, axis=1)[:, None]
            np.testing.assert_allclose(
                result["direction_tmax"][:, :3], expected, atol=1e-6
            )
            state = np.frombuffer(paths.read(), HOT_PATH_STATE_DTYPE)
            np.testing.assert_array_equal(
                state["metadata"][:, 0], pixels[:, 1] * 7 + pixels[:, 0]
            )
            np.testing.assert_array_equal(state["metadata"][:, 1], (4 << 8) | 1)
            np.testing.assert_array_equal(
                state["metadata"][:, 3], 257 | (8 if capture else 0)
            )
            np.testing.assert_array_equal(state["radiance"], 0)
            np.testing.assert_array_equal(
                np.frombuffer(media.read(), np.float32).reshape(6, 16)[:, 0], 1
            )
        with pytest.raises(ValueError, match="tile"):
            stage.operation(extent=(7, 5))
        with pytest.raises(RuntimeError, match="borrowers"):
            rays.close()
