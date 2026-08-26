"""Legacy interactive PyNARI benchmark based on sample02."""

import gc
import math
import random
import time
from collections import deque

import dearpygui.dearpygui as dpg
import numpy as np
import pynari as anari


WIDTH = 800
HEIGHT = 450
TEXTURE_TAG = "anari_color_texture"
IMAGE_TAG = "anari_color_image"
STATUS_TAG = "render_status"
CONTINUOUS_TAG = "continuous_rendering"
ANIMATE_TAG = "animate_camera"
SAMPLES_TAG = "pixel_samples"


def create_scene():
    """Create a smaller version of PyNARI's sample02 RTOW scene."""
    device = anari.newDevice("default")
    objects = []
    surfaces = []
    rng = random.Random(80577)

    def remember(obj):
        objects.append(obj)
        return obj

    def matte(color):
        material = remember(device.newMaterial("matte"))
        material.setParameter("color", anari.FLOAT32_VEC3, color)
        material.commitParameters()
        return material

    def metal(color, roughness):
        material = remember(device.newMaterial("physicallyBased"))
        material.setParameter("baseColor", anari.FLOAT32_VEC3, color)
        material.setParameter("metallic", anari.FLOAT32, 1.0)
        material.setParameter("roughness", anari.FLOAT32, roughness)
        material.commitParameters()
        return material

    def glass(ior=1.5):
        material = remember(device.newMaterial("physicallyBased"))
        material.setParameter("baseColor", anari.FLOAT32_VEC3, (1.0, 1.0, 1.0))
        material.setParameter("transmission", anari.FLOAT32, 1.0)
        material.setParameter("metallic", anari.FLOAT32, 0.0)
        material.setParameter("ior", anari.FLOAT32, ior)
        material.commitParameters()
        return material

    def add_sphere(position, radius, material):
        geometry = remember(device.newGeometry("sphere"))
        positions = remember(
            device.newArray1D(
                anari.FLOAT32_VEC3,
                np.asarray([position], dtype=np.float32),
            )
        )
        geometry.setParameter("vertex.position", anari.ARRAY1D, positions)
        geometry.setParameter("radius", anari.FLOAT32, radius)
        geometry.commitParameters()

        surface = remember(device.newSurface())
        surface.setParameter("geometry", anari.GEOMETRY, geometry)
        surface.setParameter("material", anari.MATERIAL, material)
        surface.commitParameters()
        surfaces.append(surface)

    # Ground and three large reference spheres.
    add_sphere((0.0, -1000.0, 0.0), 1000.0, matte((0.5, 0.5, 0.5)))
    add_sphere((0.0, 1.0, 0.0), 1.0, glass())
    add_sphere((-4.0, 1.0, 0.0), 1.0, matte((0.4, 0.2, 0.1)))
    add_sphere((4.0, 1.0, 0.0), 1.0, metal((0.7, 0.6, 0.5), 0.05))

    # 225 small spheres provide enough geometry/material diversity for a
    # useful interactive workload without sample02's full startup cost.
    for a in range(-7, 8):
        for b in range(-7, 8):
            position = (a + 0.8 * rng.random(), 0.2, b + 0.8 * rng.random())
            choice = rng.random()
            if choice < 0.75:
                color = tuple(rng.random() ** 2 for _ in range(3))
                material = matte(color)
            elif choice < 0.93:
                color = tuple(0.5 * (1.0 + rng.random()) for _ in range(3))
                material = metal(color, 0.2)
            else:
                material = glass()
            add_sphere(position, 0.2, material)

    world = remember(device.newWorld())
    world.setParameterArray1D("surface", anari.SURFACE, surfaces)
    world.commitParameters()

    camera = remember(device.newCamera("perspective"))
    camera.setParameter("aspect", anari.FLOAT32, WIDTH / HEIGHT)
    camera.setParameter("up", anari.FLOAT32_VEC3, (0.0, 1.0, 0.0))
    camera.setParameter("fovy", anari.FLOAT32, math.radians(28.0))

    background_values = np.asarray(
        ((0.9, 0.9, 0.9, 1.0), (0.15, 0.25, 0.8, 1.0)),
        dtype=np.float32,
    ).reshape((2, 1, 4))
    background = remember(device.newArray2D(anari.FLOAT32_VEC4, background_values))

    renderer = remember(device.newRenderer("default"))
    renderer.setParameter("ambientRadiance", anari.FLOAT32, 0.8)
    renderer.setParameter("pixelSamples", anari.INT32, 1)
    renderer.setParameter("background", anari.ARRAY2D, background)
    renderer.commitParameters()

    frame = remember(device.newFrame())
    frame.setParameter("size", anari.UINT32_VEC2, (WIDTH, HEIGHT))
    frame.setParameter("channel.color", anari.DATA_TYPE, anari.UFIXED8_RGBA_SRGB)
    frame.setParameter("renderer", anari.RENDERER, renderer)
    frame.setParameter("camera", anari.CAMERA, camera)
    frame.setParameter("world", anari.WORLD, world)
    frame.commitParameters()

    scene = {
        "device": device,
        "objects": objects,
        "frame": frame,
        "renderer": renderer,
        "camera": camera,
        "angle": 0.0,
    }
    update_camera(scene)
    return scene


def update_camera(scene):
    """Orbit the camera around the scene and recommit it."""
    angle = scene["angle"]
    position = (13.0 * math.cos(angle), 3.0, 13.0 * math.sin(angle))
    target = (0.0, 0.5, 0.0)
    direction = tuple(target[i] - position[i] for i in range(3))
    camera = scene["camera"]
    camera.setParameter("position", anari.FLOAT32_VEC3, position)
    camera.setParameter("direction", anari.FLOAT32_VEC3, direction)
    camera.commitParameters()


def upload_framebuffer(frame):
    rgba = np.asarray(frame.get("channel.color"), dtype=np.uint8)
    rgba = rgba.reshape((HEIGHT, WIDTH, 4))
    texture = np.flipud(rgba).astype(np.float32).ravel() / 255.0
    dpg.set_value(TEXTURE_TAG, texture)


def resize_render_image():
    """Fit the rendered image inside the viewport without distorting it."""
    # Leave room for the window padding, controls, and performance text.
    available_width = max(1, dpg.get_viewport_client_width() - 24)
    available_height = max(1, dpg.get_viewport_client_height() - 92)
    scale = min(available_width / WIDTH, available_height / HEIGHT)
    dpg.configure_item(
        IMAGE_TAG,
        width=max(1, int(WIDTH * scale)),
        height=max(1, int(HEIGHT * scale)),
    )


def release_scene(scene):
    """Release native resources before Python interpreter finalization."""
    for obj in reversed(scene["objects"]):
        obj.release()
    scene["objects"].clear()
    scene.clear()
    gc.collect()


def main():
    print("Creating benchmark scene...")
    scene = create_scene()
    frame_times = deque(maxlen=30)
    state = {"single_frame": True}

    dpg.create_context()
    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(
            WIDTH,
            HEIGHT,
            np.zeros(WIDTH * HEIGHT * 4, dtype=np.float32),
            format=dpg.mvFormat_Float_rgba,
            tag=TEXTURE_TAG,
        )

    def request_frame():
        state["single_frame"] = True

    def update_sample_count():
        samples = dpg.get_value(SAMPLES_TAG)
        scene["renderer"].setParameter("pixelSamples", anari.INT32, samples)
        scene["renderer"].commitParameters()
        request_frame()

    with dpg.window(tag="main_window", label="PyNARI Interactive Benchmark"):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Render once",
                callback=lambda _sender, _data, _user_data=None: request_frame(),
            )
            dpg.add_checkbox(label="Continuous", default_value=True, tag=CONTINUOUS_TAG)
            dpg.add_checkbox(label="Orbit camera", default_value=True, tag=ANIMATE_TAG)
            dpg.add_slider_int(
                label="Samples/pixel",
                min_value=1,
                max_value=64,
                default_value=1,
                width=180,
                tag=SAMPLES_TAG,
                callback=lambda _sender, _data, _user_data=None: update_sample_count(),
            )
        dpg.add_text("Starting...", tag=STATUS_TAG)
        dpg.add_image(
            TEXTURE_TAG,
            width=WIDTH,
            height=HEIGHT,
            tag=IMAGE_TAG,
        )

    dpg.create_viewport(
        title="PyNARI + Dear PyGui Benchmark",
        width=WIDTH + 32,
        height=HEIGHT + 115,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    resize_render_image()

    try:
        while dpg.is_dearpygui_running():
            should_render = dpg.get_value(CONTINUOUS_TAG) or state["single_frame"]
            if should_render:
                state["single_frame"] = False
                if dpg.get_value(ANIMATE_TAG):
                    scene["angle"] += math.radians(0.35)
                    update_camera(scene)

                start = time.perf_counter()
                scene["frame"].render()
                render_seconds = time.perf_counter() - start
                frame_times.append(render_seconds)

                upload_framebuffer(scene["frame"])
                average = sum(frame_times) / len(frame_times)
                dpg.set_value(
                    STATUS_TAG,
                    f"Render: {render_seconds * 1000:.1f} ms  |  "
                    f"30-frame avg: {average * 1000:.1f} ms  |  "
                    f"{1.0 / average:.1f} FPS",
                )

            resize_render_image()
            dpg.render_dearpygui_frame()
    finally:
        dpg.destroy_context()
        release_scene(scene)


if __name__ == "__main__":
    main()
