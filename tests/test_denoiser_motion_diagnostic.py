"""CPU checks for the optics diagnostic's capture and measurement contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import ordinarylight as ol
from ordinarylight.targets.vulkan.core import VulkanRayQueryCore
from tools.denoiser_motion.run import previous_view_depth, region_error


def test_offscreen_frame_index_reaches_stochastic_camera_state():
    executor = Mock()
    core = SimpleNamespace(
        scene_tlas=object(),
        scene_vertex_buffer=object(),
        config=SimpleNamespace(wavefront_tile_capacity=64),
        wavefront_executor=executor,
        _replace_wavefront_executor_if_strategy_changed=lambda: None,
    )
    camera = ol.PerspectiveCamera(position=(0, 0, -3), target=(0, 0, 0))
    for frame_index in (7, 19):
        VulkanRayQueryCore.trace_wavefront_tile(
            core,
            camera,
            8,
            8,
            frame_index=frame_index,
            readback=False,
        )
        assert executor.update_camera.call_args.kwargs["frame_sequence"] == frame_index
    executor.update_camera.reset_mock()
    VulkanRayQueryCore.trace_wavefront_tile(
        core,
        camera,
        8,
        8,
        frame_index=21,
        upload_camera=False,
        readback=False,
    )
    executor.update_camera.assert_not_called()


def test_previous_depth_uses_previous_camera_and_background_mask():
    camera = ol.PerspectiveCamera(position=(0, 0, -3), target=(0, 0, 0))
    positions = np.array([[[0, 0, 2], [1, 0, 2]]], np.float32)
    np.testing.assert_array_equal(
        previous_view_depth(positions, camera, np.array([[True, False]])),
        [[5, 0]],
    )


def test_region_error_is_local_and_absent_region_is_not_perfect_score():
    reference = np.zeros((2, 2, 3), np.float32)
    candidate = reference.copy()
    candidate[0, 0] = 1
    assert region_error(reference, candidate, np.zeros((2, 2), bool)) is None
    mask = np.array([[True, False], [False, False]])
    assert region_error(reference, candidate, mask) > 0.6
    assert region_error(reference, candidate, ~mask) == 0


def test_nrd_split_camera_preserves_transform_and_column_major_layout(tmp_path):
    import struct
    import ordinarylight_nrd
    from ordinarylight.raster import camera_matrix
    from tools.denoiser_motion.run import camera_sequence, camera
    from tests.test_denoising_signals import signals

    cameras = [camera(-0.3), camera(0.3)]
    matrices = camera_sequence(cameras, 5, 3)
    for index, entry in enumerate(matrices):
        np.testing.assert_allclose(
            entry["view_to_clip"] @ entry["world_to_view"],
            camera_matrix(cameras[index], 5, 3),
            atol=2e-6,
        )
    path = tmp_path / "capture.bin"
    ordinarylight_nrd._write_sequence(path, [signals()], matrices[:1])
    data = path.read_bytes()
    assert struct.unpack_from("<I", data, 8)[0] == 2
    # Header 24 bytes; frame metadata 28 bytes; then four column-major matrices.
    actual = np.frombuffer(data, dtype="<f4", count=64, offset=52).reshape(4, 4, 4)
    for index, name in enumerate(
        (
            "view_to_clip",
            "previous_view_to_clip",
            "world_to_view",
            "previous_world_to_view",
        )
    ):
        np.testing.assert_array_equal(actual[index].T, matrices[0][name])


def test_geometry_edge_mask_uses_guides_without_wrapping_or_noise():
    from tools.denoiser_motion.analyze_edges import geometry_edges

    normal = np.zeros((8, 10, 4), np.float32)
    normal[..., 2] = 1
    guide = dict(
        normal_roughness=normal,
        view_z=np.ones((8, 10, 1), np.float32),
        identity=np.zeros((8, 10, 1), np.uint32),
    )
    assert not geometry_edges(guide).any()
    guide["identity"][:, :3] = 1
    edge = geometry_edges(guide)
    assert edge[:, 2:4].all()
    assert not edge[:, :2].any()
    assert not edge[:, 4:].any()
    guide["view_z"][:] = 0
    guide["normal_roughness"][:] = 0
    guide["identity"][:] = 0
    assert not geometry_edges(guide).any()


def test_capture_uses_live_probability_partition_instead_of_sampled_lobe():
    from ordinarylight.targets.vulkan.api import _VulkanGlobalIlluminationEngine
    from ordinarylight.wavefront import SECONDARY_PATH_STATE_DTYPE

    state = np.zeros((1, 4), dtype=SECONDARY_PATH_STATE_DTYPE)
    state["primary_radiance"][..., :3] = 1.0
    state["primary_radiance"][..., 3] = (0.25, 0.25, 0.0, 1.0)
    # Identical radiance must have identical live partitions regardless of
    # which BSDF branch was sampled (the previous capture code did not).
    state["primary_throughput"][..., 3] = (1.0, 2.0, 1.0, 2.0)
    state["primary_position"][..., 3] = 1.0
    state["position_valid"][..., 0] = 3.0
    state["position_valid"][..., 1] = 4.0
    state["position_valid"][..., 3] = (1.0, 0.0, 1.0, 1.0)
    radiance = np.full((1, 4, 4), 8.0, np.float32)
    result = {
        "radiance": radiance,
        "denoiser_path_signals": state,
        "depth": np.ones((1, 4), np.float32),
        "normal": np.zeros((1, 4, 3), np.float32),
        "primitive_id": np.zeros((1, 4), np.uint32),
        "primary_position": np.zeros((1, 4, 3), np.float32),
        "primary_barycentric": np.zeros((1, 4, 2), np.float32),
    }
    engine = SimpleNamespace(
        config=SimpleNamespace(denoiser_signal_capture=True, wavefront_tile_capacity=4, denoiser_sampled_indirect=False),
        _core=SimpleNamespace(
            upload_window_scene=Mock(), trace_wavefront_tile=Mock(return_value=result),
        ),
    )
    capture = _VulkanGlobalIlluminationEngine.capture_denoiser_raw(
        engine, object(), object(), 4, 1, frame_index=9,
    )
    diffuse = capture["path_state"]["diffuse_radiance_hit_distance"]
    specular = capture["path_state"]["specular_radiance_hit_distance"]
    np.testing.assert_array_equal(diffuse[0, :, 0], [6, 6, 8, 0])
    np.testing.assert_array_equal(specular[0, :, 0], [2, 2, 0, 8])
    np.testing.assert_array_equal(diffuse[..., :3] + specular[..., :3], radiance[..., :3])
    np.testing.assert_array_equal(diffuse[0, :, 3], [5, 0, 5, 5])
    np.testing.assert_array_equal(specular[..., 3], diffuse[..., 3])
    # The raw executor records remain available unchanged to their owner.
    np.testing.assert_array_equal(state["diffuse_radiance_hit_distance"], 0)
    engine.config.denoiser_sampled_indirect = True
    capture = _VulkanGlobalIlluminationEngine.capture_denoiser_raw(
        engine, object(), object(), 4, 1,
    )
    diffuse = capture["path_state"]["diffuse_radiance_hit_distance"]
    specular = capture["path_state"]["specular_radiance_hit_distance"]
    np.testing.assert_array_equal(diffuse[0, :, 0], [7.75, 0.75, 8, 0])
    np.testing.assert_array_equal(specular[0, :, 0], [0.25, 7.25, 0, 8])
    np.testing.assert_array_equal(diffuse[..., :3] + specular[..., :3], radiance[..., :3])
    np.testing.assert_array_equal(diffuse[0, :, 3], [5, 0, 5, 0])
    np.testing.assert_array_equal(specular[0, :, 3], [0, 0, 0, 5])
    state["specular_radiance_hit_distance"][..., :3] = 1
    state["specular_radiance_hit_distance"][..., 3] = -1
    capture = _VulkanGlobalIlluminationEngine.capture_denoiser_raw(
        engine, object(), object(), 4, 1,
    )
    diffuse = capture["path_state"]["diffuse_radiance_hit_distance"]
    specular = capture["path_state"]["specular_radiance_hit_distance"]
    np.testing.assert_array_equal(diffuse[0, :, 0], [7, 0, 7, 0])
    np.testing.assert_array_equal(specular[0, :, 0], [1, 8, 1, 8])
