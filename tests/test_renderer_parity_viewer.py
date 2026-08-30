from types import SimpleNamespace

import ordinarylight as ol
from ordinarylight.integrations.workbench import OrbitCamera

from tools import raster_feature_viewer as viewer


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


def test_parity_viewer_lists_all_three_rendering_targets():
    assert tuple(key for _title, key in viewer.TARGETS) == (
        "vulkan-raster", "wavefront-gi", "webgpu-raster",
    )


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


def test_gi_viewer_config_uses_interactive_single_sample_wavefront():
    showcase = SimpleNamespace(renderer={})
    config = viewer._gi_config(showcase, present=True)
    assert config.samples_per_pixel == 1
    assert config.max_bounces == 8
    assert config.progressive_accumulation is False
    assert config.present_mode == "mailbox"
    assert config.direct_swapchain_storage is True


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
