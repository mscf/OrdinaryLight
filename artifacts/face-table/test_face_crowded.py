import os
import numpy as np
import pytest
osh = pytest.importorskip("ordinaryshade")


def fill_program(image_format):
    @osh.compute(workgroup_size=(8, 8, 1))
    def fill_hdr(values: osh.storage_buffer(osh.vec4, access='read', binding=0),
                 image: osh.storage_image(image_format, access='write', binding=1)):
        pixel = osh.ivec2(osh.global_invocation_id.xy)
        size = image.size()
        if pixel.x < size.x and pixel.y < size.y:
            image.store(pixel, values[pixel.y * size.x + pixel.x])

    return fill_hdr


pytestmark = pytest.mark.skipif(os.environ.get('VXL8R_TEST_VULKAN') != '1', reason='Opt-in GPU averaging')


@pytest.mark.parametrize("image_format", ["rgba16f", "rgba32f"])
@pytest.mark.parametrize("sparse,compact", [(False,False), (False,True), (True,False), (True,True)])
@pytest.mark.parametrize("single_face", [False, True])
def test_face_average_hdr_identity_padding_scaling_and_reuse(monkeypatch, image_format, sparse, compact, single_face):
    import vulkan as vk
    from ordinarylight.runtime import VulkanRuntime, VulkanKernel, compile_compute
    from ordinarylight.runtime.resources import VulkanBuffer, VulkanCompletion
    from ordinarylight.pipeline.graph import reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    from ordinarylight.wavefront import PRIMARY_HIT_DTYPE
    from vxl8r_render.backends.face_average import FaceAverageTarget
    if sparse:
        from vxl8r_render.backends.sparse_average import SparseFaceAverageTarget as FaceAverageTarget
    from vxl8r_render.backends.native_readback import read_hdr
    rng = np.random.default_rng(713)
    width, height = 45, 39
    out_w, out_h = 91, 61
    data = rng.uniform(0, 60000, (48, 64, 4)).astype('<f4')
    data[..., 3] = 1
    hits = np.zeros(width * height, PRIMARY_HIT_DTYPE)
    hits['position_distance'][:, 3] = 1
    hits['identity'][:, 1] = np.arange(len(hits)) % 3
    hits['identity'][:, 2] = (np.arange(len(hits)) // 7) % 6
    if single_face:
        hits['identity'][:, 1:3] = 0  # All stripes reduce into the same face.
    hits['identity'][:, 3] = 7  # Live recycled slots still participate in averaging.
    hits['identity'][::7] = np.uint32(0xffffffff)
    hits['position_distance'][::7, 3] = -1
    hits['identity'][1::19, 0] = 4  # Foreign instance must not collide with slot IDs.
    initial_hits = hits.copy()
    from ordinarylight.wavefront import PRIMARY_HIT_IDENTITY_DTYPE
    from ordinarylight.pipeline.gi import GiBuffer
    def encoded():
        if not compact:
            return hits
        result = np.zeros(len(hits), PRIMARY_HIT_IDENTITY_DTYPE)
        result['identity'] = hits['identity']
        result['valid'] = hits['position_distance'][:,3] >= 0
        return result
    compiled = osh.compile(fill_program(image_format))
    with VulkanRuntime() as runtime:
        with runtime.buffer(data.nbytes, data=data) as pixels, runtime.buffer(encoded().nbytes, data=encoded()) as primary:
            with runtime.image(64, 48, format=(vk.VK_FORMAT_R16G16B16A16_SFLOAT if image_format == "rgba16f" else vk.VK_FORMAT_R32G32B32A32_SFLOAT)) as hdr:
                with VulkanKernel(runtime, compile_compute(compiled.source), {0: VulkanResource.buffer(pixels), 1: VulkanResource.image(hdr)}) as upload:
                    fill = reflected_operation(upload, compiled.reflection, workgroups=(8, 6, 1))
                    with FaceAverageTarget(runtime, hdr, GiBuffer(runtime, primary.buffer, primary.byte_size, primary.usage,
                                           primary.require_open, record_dtype=PRIMARY_HIT_IDENTITY_DTYPE if compact else PRIMARY_HIT_DTYPE), render_extent=(width, height),
                                           output_extent=(out_w, out_h), slots=70000 if sparse and single_face else 3) as target:
                        for iteration in range(4):
                            data[:] = data.astype(np.float16).astype(np.float32)
                            pixels.upload(data)
                            primary.upload(encoded())
                            fill.execute(runtime).wait()
                            source = data[:height, :width, :3].astype(np.float16).astype(np.float32)
                            expected = source.reshape(-1, 3).copy()
                            valid = (hits['position_distance'][:, 3] >= 0) & (hits['identity'][:, 0] == 0)
                            for slot in range(3):
                                for face in range(6):
                                    mask = valid & (hits['identity'][:, 1] == slot) & (hits['identity'][:, 2] == face)
                                    if mask.any():
                                        expected[mask] = source.reshape(-1, 3)[mask].mean(axis=0)
                            iy = ((np.arange(out_h) + .5) * height / out_h).astype(int)
                            ix = ((np.arange(out_w) + .5) * width / out_w).astype(int)
                            for averaged in (True, False):
                                def forbidden(*args, **kwargs):
                                    raise AssertionError('GPU averaging performed host transfer or allocation')
                                with monkeypatch.context() as patch:
                                    patch.setattr(VulkanBuffer, 'read', forbidden)
                                    patch.setattr(VulkanBuffer, 'upload', forbidden)
                                    patch.setattr(runtime, 'buffer', forbidden)
                                    patch.setattr(runtime, 'image', forbidden)
                                    patch.setattr(VulkanKernel, '__init__', forbidden)
                                    if sparse and averaged in target._command_cache:
                                        patch.setattr(VulkanKernel, 'bind', forbidden)
                                    patch.setattr(VulkanCompletion, 'wait', forbidden)
                                    patch.setattr(vk, 'vkQueueWaitIdle', forbidden)
                                    patch.setattr(vk, 'vkDeviceWaitIdle', forbidden)
                                    completion = target.operation(averaged=averaged).execute(runtime)
                                result = read_hdr(runtime, target.hdr, (out_w, out_h), completion)
                                reference = expected.reshape(height, width, 3) if averaged else source
                                np.testing.assert_allclose(result, reference[iy[:, None], ix[None, :]], rtol=3e-6, atol=.02)
                            # An all-miss frame must not reuse stale face lists or colors.
                            hits['identity'][:] = np.uint32(0xffffffff)
                            hits['position_distance'][:, 3] = -1
                            data[..., :3] *= .1
                            if iteration == 2:
                                hits[:] = initial_hits
                        if sparse:
                            assert set(target._command_cache) == {False, True}
                        with pytest.raises(ValueError, match='one sampled camera ray'):
                            target.operation(averaged=True, sample_count=2)


def test_sparse_average_compact_queue_supports_unique_faces_and_2d_dispatch():
    import vulkan as vk
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel,compile_compute
    from ordinarylight.pipeline.graph import reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    from ordinarylight.wavefront import PRIMARY_HIT_DTYPE
    from vxl8r_render.backends.sparse_average import SparseFaceAverageTarget
    from vxl8r_render.backends.native_readback import read_hdr
    width,height=512,256
    count=width*height
    data=np.random.default_rng(512).uniform(0,2,(height,width,4)).astype(np.float32)
    hits=np.zeros(count,PRIMARY_HIT_DTYPE)
    hits['position_distance'][:,3]=1
    hits['identity'][:,1]=np.arange(count,dtype=np.uint32)
    shader=osh.compile(fill_program('rgba32f'))
    with VulkanRuntime() as runtime,runtime.buffer(data.nbytes,data=data) as pixels,runtime.buffer(hits.nbytes,data=hits) as primary:
        with runtime.image(width,height,format=vk.VK_FORMAT_R32G32B32A32_SFLOAT) as hdr:
            with VulkanKernel(runtime,compile_compute(shader.source),{0:VulkanResource.buffer(pixels),1:VulkanResource.image(hdr)}) as fill:
                reflected_operation(fill,shader.reflection,workgroups=((width+7)//8,(height+7)//8,1)).execute(runtime).wait()
                with SparseFaceAverageTarget(runtime,hdr,primary,render_extent=(width,height),output_extent=(width,height),slots=count) as target:
                    completion=target.operation().execute(runtime)
                    actual=read_hdr(runtime,target.hdr,(width,height),completion)
                    np.testing.assert_allclose(actual,data[...,:3],rtol=1e-6,atol=1e-6)
                    work=np.frombuffer(target.worklist.read(),np.uint32)
                    used=int(work[0])
                    assert used==2*count-(count+target.stripes-1)//target.stripes
                    assert len(np.unique(work[4:4+used]))==used
                    args=np.frombuffer(target.arguments.read(),np.uint32)
                    groups=(used+63)//64
                    np.testing.assert_array_equal(args,[min(groups,4096),(groups+4095)//4096,1])
                    assert args[1]>=1
