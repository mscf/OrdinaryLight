from types import SimpleNamespace

import numpy as np
import pytest

import ordinarylight as ol
from ordinarylight.integrations.workbench import OrbitCamera

from ordinarylight.integrations import raster_workbench as viewer


def _showcase(identifier, counter):
    def create_scene():
        counter.append(identifier)
        scene = ol.Scene()
        scene.add_mesh(
            ((-1, 0, 0), (1, 0, 0), (0, 1, 0)),
            ((0, 1, 2),), ol.Material(),
        )
        return scene

    return SimpleNamespace(
        id=identifier,
        create_scene=create_scene,
        camera=OrbitCamera(
            target=(0, 0, 0), radius=4, height=2,
        ),
        renderer={},
    )


def test_parity_viewer_lists_rendering_targets():
    assert tuple(key for _title, key in viewer.TARGETS) == (
        "vulkan-raster", "wavefront-gi", "wavefront-gi-fast", "webgpu-raster",
    )


def test_parity_viewer_exposes_gi_volume_showcases():
    catalog = {showcase.id: showcase for showcase in viewer._catalog()}

    assert {
        "volume",
        "multi-volume",
        "volume-scattering",
        "volume-multiple-scattering",
    }.issubset(catalog)
    assert "gi-feature" in catalog["volume"].tags
    assert "raster-feature" in catalog["volume"].tags
    assert catalog["volume"].description
    assert catalog["volume"].camera.target == (-0.1, 1.45, -0.7)
    assert catalog["volume"].renderer["volume_rendering"] == "ray-march"
    assert catalog["multi-volume"].renderer["volume_rendering"] == "ray-march"


def test_optional_scene_light_toggle_preserves_authored_intensity():
    from ordinarylight.showcases.raster_features import (
        build_material_program_room_scene,
    )

    scene = build_material_program_room_scene()
    entry = scene.metadata["optional_scene_lights"][0]
    light = scene.get_light(entry["id"])
    viewer._set_optional_scene_lights(scene, False)
    assert light.intensity == 0.0
    viewer._set_optional_scene_lights(scene, True)
    assert light.intensity == entry["intensity"]


def test_material_program_room_uses_shared_screen_space_optics():
    catalog = {showcase.id: showcase for showcase in viewer._catalog()}
    settings = catalog["raster-material-program-room"].renderer

    assert settings["optical_quality"] == "screen-space"
    assert settings["screen_space_ray_steps"] == 64
    assert settings["screen_space_optical_layers"] == 4


def test_target_switch_preserves_scene_and_camera_controller():
    created = []
    showcase = _showcase("one", created)
    scene, controller, active = viewer._preserved_view(
        showcase, None, None, None,
    )
    controller.orbit(0.4, -0.2)
    camera_before = controller.camera()

    retained_scene, retained_controller, retained_active = viewer._preserved_view(
        showcase, scene, controller, active,
    )
    assert retained_scene is scene
    assert retained_controller is controller
    assert retained_active == "one"
    assert retained_controller.camera() == camera_before
    assert created == ["one"]


def test_feature_switch_replaces_scene_and_camera_controller():
    created = []
    first = _showcase("one", created)
    second = _showcase("two", created)
    scene, controller, active = viewer._preserved_view(first, None, None, None)
    new_scene, new_controller, new_active = viewer._preserved_view(
        second, scene, controller, active,
    )
    assert new_scene is not scene
    assert new_controller is not controller
    assert new_active == "two"
    assert created == ["one", "two"]


def test_gi_viewer_config_uses_quality_wavefront_defaults():
    showcase = SimpleNamespace(renderer={})
    config = viewer._gi_config(showcase, present=True)
    assert config.samples_per_pixel == 2
    assert config.max_bounces == 8
    assert config.wavefront_restir_di is True
    assert config.wavefront_restir_reservoirs == 2
    assert config.wavefront_restir_candidates == 4
    assert config.wavefront_restir_history_limit == 20
    assert config.wavefront_restir_shared_primary is True
    assert config.wavefront_tile_capacity == 524288
    assert config.wavefront_restir_spatial_reuse is False
    assert config.progressive_accumulation is True
    assert config.temporal_history is True
    assert config.denoiser_enabled is True
    assert config.denoiser_iterations == 3
    assert config.present_mode == "mailbox"
    assert config.direct_swapchain_storage is True
    assert config.wavefront_hdr_capture is False


def test_gi_viewer_config_accepts_restir_reservoir_count():
    showcase = SimpleNamespace(renderer={})
    config = viewer._gi_config(showcase, restir_reservoirs=8)
    assert config.wavefront_restir_reservoirs == 8


def test_gi_viewer_config_can_present_raw_gi_without_denoiser_history():
    showcase = SimpleNamespace(renderer={})
    config = viewer._gi_config(
        showcase, denoiser_enabled=False, denoiser_iterations=5,
    )
    assert config.denoiser_enabled is False
    assert config.progressive_accumulation is False
    assert config.temporal_history is False
    assert config.denoiser_iterations == 5


def test_gi_viewer_config_enables_explicit_hdr_capture():
    showcase = SimpleNamespace(renderer={})
    config = viewer._gi_config(showcase, present=True, capture=True)
    assert config.wavefront_hdr_capture is True


def test_gi_temporal_variance_report_detects_two_alternating_chains():
    base = np.zeros((3, 4, 4), dtype=np.float32)
    first = base.copy()
    second = base.copy()
    first[..., 0] = 0.25
    second[..., 0] = 0.75
    report = viewer._gi_temporal_variance_report(
        [first, second, first, second, first, second]
    )
    assert report["mean_adjacent_rmse"] > 0.0
    assert report["mean_lag_two_rmse"] == 0.0
    assert report["lag_two_to_adjacent_rmse_ratio"] == 0.0
    assert report["alternating_history_signature"] is True


def test_gi_viewer_config_honors_showcase_bounce_count():
    showcase = SimpleNamespace(renderer={"max_bounces": 12})

    config = viewer._gi_config(showcase)

    assert config.max_bounces == 12


def test_direct_gi_uses_native_surface_extent_to_avoid_recreation_loop():
    selected = (3840, 2160)
    native_client = (3840, 2130)
    assert viewer._direct_render_extent(
        "wavefront-gi", selected, native_client,
    ) == native_client
    assert viewer._direct_render_extent(
        "vulkan-raster", selected, native_client,
    ) == selected


def test_raster_internal_extent_matches_native_surface_aspect_ratio():
    extent = viewer._surface_aspect_extent((1920, 1080), (1200, 900))
    assert extent == (1440, 1080)
    assert extent[0] / extent[1] == 1200 / 900


def test_camera_pose_json_round_trips_into_validated_camera():
    showcase_id, camera = viewer._camera_pose_from_json(
        '{"showcase":"optical-screen-rough-reflection",'
        '"position":[-9.795,1.33,0.12],"target":[0,1.1,0],'
        '"up":[0,1,0],"vertical_fov_degrees":45}'
    )

    assert showcase_id == "optical-screen-rough-reflection"
    assert camera.position == (-9.795, 1.33, 0.12)
    assert camera.target == (0, 1.1, 0)
    assert camera.vertical_fov_degrees == 45.0
    restored = ol.ArcballCameraController.from_camera(camera).camera()
    assert restored.position == pytest.approx(camera.position)
    assert restored.target == pytest.approx(camera.target)
    assert restored.up == pytest.approx(camera.up)
    assert restored.vertical_fov_degrees == camera.vertical_fov_degrees


@pytest.mark.parametrize(
    "payload, message",
    (
        ("not json", "invalid camera-pose JSON"),
        ('{"position":[0,0,1],"target":[0,0,0]}', "showcase"),
        (
            '{"showcase":"x","position":[0,0,1],"target":[0,0,0],'
            '"up":[1,0,0]}',
            "up vector",
        ),
    ),
)
def test_camera_pose_json_rejects_invalid_viewer_payloads(payload, message):
    with pytest.raises(ValueError, match=message):
        viewer._camera_pose_from_json(payload)


def test_camera_pose_argument_accepts_inline_json_and_file(tmp_path):
    payload = (
        '{"showcase":"optical-screen-rough-reflection",'
        '"position":[-9,1,0],"target":[0,1,0]}'
    )
    inline_id, inline_camera = viewer._camera_pose_argument(payload)
    path = tmp_path / "pose.json"
    path.write_text(payload, encoding="utf-8")
    file_id, file_camera = viewer._camera_pose_argument(path)

    assert inline_id == file_id == "optical-screen-rough-reflection"
    assert inline_camera == file_camera


def test_planar_mirror_guides_are_opt_in_and_limited_to_static_showcase():
    showcase = SimpleNamespace(renderer={}, id="planar-mirror-guides")
    assert not viewer._gi_config(showcase).denoiser_planar_mirror_guides
    assert viewer._gi_config(showcase, denoiser_planar_mirror_guides=True).denoiser_planar_mirror_guides
    other = SimpleNamespace(renderer={}, id="glossy-glass")
    assert not viewer._gi_config(other, denoiser_planar_mirror_guides=True).denoiser_planar_mirror_guides


def test_gi_viewer_honors_showcase_restir_override():
    assert viewer._gi_config(SimpleNamespace(renderer={})).wavefront_restir_di
    settings = SimpleNamespace(renderer={"wavefront_restir_di": False})
    assert not viewer._gi_config(settings).wavefront_restir_di


def test_transmission_motion_cap_is_opt_in():
    showcase = SimpleNamespace(renderer={}, id="glass-detail-motion")
    assert not viewer._gi_config(showcase).denoiser_transmission_motion_cap
    assert viewer._gi_config(
        showcase, denoiser_transmission_motion_cap=True,
    ).denoiser_transmission_motion_cap


def test_custom_inline_is_opt_in_and_limited_to_validated_showcases():
    for name in viewer.CUSTOM_INLINE_SHOWCASES:
        showcase = SimpleNamespace(id=name, renderer={})
        default = viewer._gi_config(showcase)
        enabled = viewer._gi_config(showcase, custom_inline=True, shared_primary=False)
        assert not default.wavefront_custom_inline
        assert default.wavefront_execution_strategy == "wavefront"
        assert enabled.wavefront_custom_inline
        assert enabled.wavefront_execution_strategy == "hybrid"
        assert enabled.wavefront_hybrid_inline_bounces == 3
    unsupported = viewer._gi_config(SimpleNamespace(id="volume", renderer={}), custom_inline=True)
    assert not unsupported.wavefront_custom_inline


def test_gi_render_scale_preserves_native_default_and_sample_budget():
    showcase = SimpleNamespace(id="glass-detail-camera", renderer={})
    native = viewer._gi_config(showcase)
    scaled = viewer._gi_config(showcase, render_scale=.5)
    assert native.wavefront_render_scale == 1.0
    assert scaled.wavefront_render_scale == .5
    assert scaled.wavefront_restir_reservoirs == native.wavefront_restir_reservoirs


def test_cubic_upscale_is_opt_in_and_rejects_unknown_filters():
    showcase = SimpleNamespace(id="glass-detail-camera", renderer={})
    assert viewer._gi_config(showcase).wavefront_upscale_filter == "bilinear"
    assert viewer._gi_config(
        showcase, render_scale=.5, upscale_filter="clamped-cubic",
    ).wavefront_upscale_filter == "clamped-cubic"
    with pytest.raises(ValueError, match="wavefront_upscale_filter"):
        viewer._gi_config(showcase, upscale_filter="unknown")


def test_fsr1_is_opt_in_and_rejects_bypassed_reconstruction_features():
    showcase = SimpleNamespace(id="glass-detail-camera", renderer={})
    config = viewer._gi_config(showcase, render_scale=.5, upscale_filter="fsr1")
    assert config.wavefront_upscale_filter == "fsr1"
    for option in ("wavefront_temporal_reconstruction", "stationary_accumulation",
                   "wavefront_diffuse_filter"):
        with pytest.raises(ValueError, match="fsr1 currently requires"):
            ol.RendererConfig(
                wavefront_upscale_filter="fsr1", progressive_accumulation=True,
                **{option: True},
            )


def test_low_bounce_gi_limits_paths_without_lowering_resolution():
    showcase = SimpleNamespace(renderer={"max_bounces": 8})
    regular = viewer._gi_config(showcase)
    fast = viewer._gi_config(
        showcase, low_bounce=True, path_spp=8, shared_primary=False,
    )
    assert regular.max_bounces == 8
    assert regular.samples_per_pixel == 2
    assert fast.max_bounces == 4
    assert fast.samples_per_pixel == 1
    assert fast.wavefront_restir_shared_primary
    assert fast.wavefront_restir_reservoirs == regular.wavefront_restir_reservoirs
    assert fast.wavefront_render_scale == regular.wavefront_render_scale == 1.0
    assert fast.denoiser_enabled


def test_low_bounce_readback_uses_gi_renderer(monkeypatch):
    monkeypatch.setattr(viewer.ol, "Renderer", lambda **options: options)
    result = viewer._renderer(
        SimpleNamespace(renderer={}), None, "wavefront-gi-fast", True, 512,
    )
    assert result["renderer_preference"] == "gi"
    assert result["config"].max_bounces == 4
    assert result["config"].samples_per_pixel == 1
    assert viewer._direct_render_extent(
        "wavefront-gi-fast", (1280, 720), (2052, 1764),
    ) == (2052, 1764)
