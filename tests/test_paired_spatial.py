"""Paired spatial filtering must preserve both lobes and reusable graph contracts."""
from contextlib import ExitStack
import os
import numpy as np
import ordinaryshade as osh
import pytest
import vulkan as vk
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ordinarylight.runtime import VulkanRuntime, VulkanKernel, compile_compute
from ordinarylight.runtime import relax


@osh.compute(workgroup_size=(8, 8, 1))
def initialize(
    diffuse: osh.storage_image('rgba16f', access='write', binding=0),
    specular: osh.storage_image('rgba16f', access='write', binding=1),
    normal: osh.storage_image('rgba16f', access='write', binding=2),
    depth: osh.storage_image('r32f', access='write', binding=3),
    material: osh.storage_image('r32ui', access='write', binding=4),
    hdr: osh.storage_image('rgba16f', access='write', binding=5),
):
    p = osh.ivec2(osh.global_invocation_id.xy)
    if p.x >= 19 or p.y >= 13:
        return
    value = osh.f32((p.x * 7 + p.y * 3) % 17) * 0.125
    d = osh.vec3(value, 0.3, 1.0)
    s = osh.vec3(0.1, value, 0.2)
    if p.x == 9 and p.y == 6:
        d = osh.vec3(400.0)
        s = osh.vec3(900.0)
    diffuse.store(p, osh.vec4(d, 3.0))
    specular.store(p, osh.vec4(s, 7.0))
    normal.store(p, osh.vec4(0.0, 0.0, 1.0, 0.5))
    depth.store(p, osh.vec4(0.0 if p.x == 0 else 1.0 + osh.f32(p.x) * 0.001))
    material.store(p, osh.uvec4(osh.u32(p.x / 10)))
    hdr.store(p, osh.vec4(2.0, 3.0, 4.0, 1.0))


@osh.compute(workgroup_size=(8, 8, 1))
def capture(
    diffuse: osh.storage_image('rgba16f', access='read', binding=0),
    specular: osh.storage_image('rgba16f', access='read', binding=1),
    hdr: osh.storage_image('rgba16f', access='read', binding=2),
    result: osh.storage_buffer(osh.vec4, binding=3),
):
    p = osh.ivec2(osh.global_invocation_id.xy)
    if p.x >= 19 or p.y >= 13:
        return
    index = osh.u32(p.y * 19 + p.x) * osh.u32(3)
    result[index] = diffuse.load(p)
    result[index + osh.u32(1)] = specular.load(p)
    result[index + osh.u32(2)] = hdr.load(p)


def use(resource, access):
    return VulkanResourceUse(resource, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             access, vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == 'image' else None)


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='opt-in GPU')
@pytest.mark.parametrize('iterations', range(1, 6))
def test_paired_matches_fallback_with_resize_and_split_operations(monkeypatch, iterations):
    with VulkanRuntime() as runtime, ExitStack() as stack:
        if not relax._supports_paired_filter(runtime):
            pytest.skip('Device lacks seven storage image bindings')
        rgba = vk.VK_FORMAT_R16G16B16A16_SFLOAT
        images = [stack.enter_context(runtime.image(19, 13, format=f)) for f in
                  (rgba, rgba, rgba, vk.VK_FORMAT_R32_SFLOAT, vk.VK_FORMAT_R32_UINT, rgba)]
        resources = {i: VulkanResource.image(image) for i, image in enumerate(images)}
        writer = stack.enter_context(VulkanKernel(runtime, compile_compute(osh.compile(initialize).source), resources))
        result = stack.enter_context(runtime.buffer(19 * 13 * 3 * 16))
        results = []
        for paired in (False, True):
            with monkeypatch.context() as patch:
                patch.setattr(relax, '_supports_paired_filter', lambda runtime: paired)
                stage = stack.enter_context(relax.VulkanRelaxSpatial(runtime, **dict(zip(
                    ('diffuse','specular','normal_roughness','view_z','material','output'),images)), iterations=iterations))
            assert len(stage.passes) == (iterations if paired else iterations * 2) + 1
            bindings = {0: VulkanResource.image(stage.filtered_diffuse),
                        1: VulkanResource.image(stage.filtered_specular), 2: resources[5],
                        3: VulkanResource.buffer(result)}
            reader = stack.enter_context(VulkanKernel(runtime, compile_compute(osh.compile(capture).source), bindings))
            frames = []
            for extent in ((19,13), (13,9), (19,13)):
                graph = VulkanGraph().add('initialize', VulkanPass('init', tuple(use(r,vk.VK_ACCESS_SHADER_WRITE_BIT) for r in resources.values()),writer.bind,(3,2,1)))
                graph.add('filter', stage.filter_operation(extent=extent),after=('initialize',))
                graph.add('compose', stage.compose_operation(extent=extent),after=('filter',))
                graph.add('capture', VulkanPass('read',tuple(use(r,vk.VK_ACCESS_SHADER_WRITE_BIT if i==3 else vk.VK_ACCESS_SHADER_READ_BIT) for i,r in bindings.items()),reader.bind,(3,2,1)),after=('compose',))
                compiled = graph.compile()
                for _ in range(2):
                    compiled.execute(runtime).wait()
                    pixels = np.frombuffer(result.read(),np.float32).reshape(13,19,3,4).copy()
                    w,h = extent
                    pixels = pixels[:h,:w]
                    np.testing.assert_array_equal(pixels[:,:,0,3],3)
                    np.testing.assert_array_equal(pixels[:,:,1,3],7)
                    np.testing.assert_array_equal(pixels[:,0,2],np.tile([2,3,4,1],(h,1)))
                    frames.append(pixels)
            results.append(frames)
        for baseline, candidate in zip(*results):
            np.testing.assert_array_equal(baseline, candidate)
