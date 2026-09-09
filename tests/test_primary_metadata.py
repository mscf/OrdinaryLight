from contextlib import ExitStack
import os
import numpy as np
import pytest
import vulkan as vk
import ordinarylight as ol
from ordinarylight.wavefront import (
    prepare_primary_metadata,
    HOT_PATH_STATE_DTYPE,
    SECONDARY_PATH_STATE_DTYPE,
)
from ordinarylight.runtime import VulkanPrimaryMetadata, prepare_primary_metadata_shader


def fixture():
    scene = ol.Scene()
    for roughness in (0.25, 0.75):
        scene.add_mesh(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 2]],
            ol.Material(roughness=roughness),
        )
    return scene


def test_prepared_scene_metadata():
    scene = fixture()
    prepared = prepare_primary_metadata(scene)
    words = np.frombuffer(prepared.data, np.uint32).reshape(-1, 4)
    assert prepared.triangle_count == 2
    np.testing.assert_array_equal(words[:, 0], scene.triangle_material_ids())
    np.testing.assert_array_equal(words[:, 1], scene.triangle_instance_ids())
    np.testing.assert_array_equal(words[:, 2].view(np.float32), [0.25, 0.75])


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_primary_metadata_updates_captured_state():
    scene = fixture()
    prepared = prepare_primary_metadata(scene)
    spirv = prepare_primary_metadata_shader()
    with ol.VulkanRuntime() as runtime, ExitStack() as stack:

        def buffer(data):
            return stack.enter_context(runtime.buffer(len(data), data=data))

        paths = np.zeros(2, HOT_PATH_STATE_DTYPE)
        paths["metadata"][:, 0] = [0, 1]
        secondary = np.zeros(2, SECONDARY_PATH_STATE_DTYPE)
        secondary["primary_position"][:, 3] = 1
        secondary["primary_geometry"].view(np.uint32)[:, 0] = [1, 0]
        state = buffer(secondary.tobytes())
        image = stack.enter_context(runtime.image(2, 1, format=vk.VK_FORMAT_R32_UINT))
        stage = stack.enter_context(
            VulkanPrimaryMetadata(
                runtime,
                paths=buffer(paths.tobytes()),
                secondary_paths=state,
                metadata=buffer(prepared.data),
                material=image,
                capacity=2,
                triangle_count=2,
                spirv=spirv,
            )
        )
        stage.operation(path_count=2).execute(runtime).wait()
        result = np.frombuffer(state.read(), SECONDARY_PATH_STATE_DTYPE)
        np.testing.assert_array_equal(result["primary_position"][:, 3], [1.75, 1.25])
        np.testing.assert_array_equal(
            result["primary_geometry"].view(np.uint32)[:, 3],
            scene.triangle_instance_ids()[::-1],
        )
        np.testing.assert_array_equal(
            result["primary_geometry"].view(np.uint32)[:, 0], [1, 0]
        )
