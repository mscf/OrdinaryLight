"""GPU prototype checks; run with vxl8r/src on PYTHONPATH."""
from contextlib import ExitStack
import numpy as np
import pytest


@pytest.mark.parametrize('slots', (3, 90000))
@pytest.mark.parametrize('budget', (1, 8, 32))
def test_face_budget_identity_weights_and_reuse(slots, budget, monkeypatch):
    import vulkan as vk
    import ordinaryshade as osh
    from ordinarylight.runtime import VulkanRuntime, VulkanKernel, compile_compute
    from ordinarylight.pipeline.graph import reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    from ordinarylight.pipeline.gi import GiBuffer
    from ordinarylight.wavefront import PRIMARY_HIT_IDENTITY_DTYPE
    from vxl8r_render.backends.sparse_average import SparseFaceAverageTarget
    from face_lighting_plan_experiment import FaceLightingPlan
    from profile_face_table_occupancy import fill_image
    width, height = 45, 39
    pixels = width * height
    hits = np.zeros(pixels, PRIMARY_HIT_IDENTITY_DTYPE)
    hits['identity'][:,1] = np.arange(pixels) % 3
    hits['identity'][:,2] = (np.arange(pixels) // 7) % 6
    mask = np.arange(pixels) % 7 != 0
    with ExitStack() as stack:
        runtime = stack.enter_context(VulkanRuntime())
        primary = stack.enter_context(runtime.buffer(hits.nbytes, data=hits))
        hdr = stack.enter_context(runtime.image(width,height,format=vk.VK_FORMAT_R32G32B32A32_SFLOAT))
        compiled = osh.compile(fill_image)
        kernel = stack.enter_context(VulkanKernel(runtime,compile_compute(compiled.source),{0:VulkanResource.image(hdr)}))
        reflected_operation(kernel,compiled.reflection,workgroups=((width+7)//8,(height+7)//8,1)).execute(runtime).wait()
        wrapped = GiBuffer(runtime,primary.buffer,primary.byte_size,primary.usage,primary.require_open,
                           record_dtype=PRIMARY_HIT_IDENTITY_DTYPE)
        target = stack.enter_context(SparseFaceAverageTarget(runtime,hdr,wrapped,
            render_extent=(width,height),output_extent=(width,height),slots=slots))
        plan = stack.enter_context(FaceLightingPlan(target))
        for index, valid in enumerate((mask, np.zeros(pixels,dtype=bool), mask)):
            hits['valid'] = valid
            primary.upload(hits)
            target.operation().execute(runtime).wait()
            def forbidden(*a, **kw):
                raise AssertionError('Sample planning allocated GPU resources')
            with monkeypatch.context() as patch:
                patch.setattr(runtime, 'buffer', forbidden)
                patch.setattr(runtime, 'image', forbidden)
                patch.setattr(VulkanKernel, '__init__', forbidden)
                completion = plan.operation(budget=budget,seed=index).execute(runtime)
            completion.wait()
            counters, samples, ranges = plan.read()
            anchors = np.frombuffer(target.anchors.read(),np.uint32)
            faces, populations = np.unique(anchors[valid],return_counts=True)
            assert int(counters[0]) == sum(min(int(n),budget) for n in populations)
            assert int(counters[1]) == len(faces)
            assert int(counters[2]) == int(valid.sum())
            assert np.count_nonzero(ranges[:,1]) == len(faces)
            assert len(np.unique(samples[:,0])) == len(samples)
            for face, population in zip(faces,populations):
                offset, count = map(int,ranges[face])
                selected = samples[offset:offset+count]
                assert count == min(int(population),budget)
                assert np.all(selected[:,0] < pixels)
                assert np.all(valid[selected[:,0]])
                assert np.all(anchors[selected[:,0]] == face)
                assert np.all(selected[:,1] == face)
                assert np.all(selected[:,3] == population)
                weights = selected[:,2].copy().view(np.float32)
                assert np.all(weights > 0)
                np.testing.assert_allclose(weights.sum(),1.0,atol=1e-6)
        with pytest.raises(ValueError):
            plan.operation(budget=0)
        with pytest.raises(ValueError):
            plan.operation(budget=33)
