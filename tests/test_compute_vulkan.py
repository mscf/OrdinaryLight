"""Opt-in real-device coverage for reflected Vulkan compute."""

import importlib.util
import os

import numpy as np
import pytest

import ordinarylight as ol
import ordinaryshade as osh
from ordinarylight.shaders import find_glsl_compiler


@osh.compute(workgroup_size=(4, 1, 1))
def add_one(
    source: osh.storage_buffer(osh.f32, access="read", binding=0),
    destination: osh.storage_buffer(osh.f32, access="write", binding=1),
):
    index = osh.global_invocation_id.x
    if index < osh.u32(8):
        destination[index] = source[index] + osh.f32(1.0)


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") != "1",
    reason="GPU compute validation is opt-in",
)
def test_vulkan_compute_sequence_dispatches_on_raster_device():
    if importlib.util.find_spec("vulkan") is None:
        pytest.skip("vulkan is unavailable")
    compiler = find_glsl_compiler()
    if compiler is None:
        pytest.skip("a GLSL-to-SPIR-V compiler is unavailable")
    compute = osh.compile(
        add_one, target="spirv", spirv_compiler=compiler, validate=True,
    )
    raster = ol.RasterProgram.scene(target="spirv", validate=False)
    renderer = ol.renderers.raster.VulkanRasterRenderer(raster)
    try:
        resources = {
            "source": ol.ComputeBuffer(
                np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            ),
            "destination": ol.ComputeBuffer(
                shape=(2, 2, 2), dtype=np.float32,
            ),
        }
        with ol.VulkanComputeSequence(
            (ol.ComputeStep(compute, (2, 1, 1)),), resources,
            context=renderer,
        ) as sequence:
            sequence.dispatch()
            np.testing.assert_array_equal(
                sequence.read("destination"),
                (np.arange(8, dtype=np.float32) + 1).reshape(2, 2, 2),
            )
            view = sequence.buffer_view("destination")
            assert view.device == renderer.device
            volume = ol.Volume(
                np.zeros((2, 2, 2), np.float32), gpu_source=view,
                material=ol.VolumeMaterial(ol.Texture1D(np.asarray(
                    ((0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 1.0, 0.5)),
                    np.float32,
                ))), value_range=(0.0, 8.0),
            )
            image = ol.Renderer(implementation=renderer).render(
                ol.Scene(volumes=[volume]),
                ol.PerspectiveCamera((0, 0, 3), (0, 0, 0)),
                (32, 32),
            )
            assert image.shape == (32, 32, 4)
    finally:
        renderer.close()
