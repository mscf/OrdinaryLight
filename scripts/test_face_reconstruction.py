"""GPU reconstruction contract: local direct/specular, weighted null draws, fallback."""
from contextlib import ExitStack
from types import SimpleNamespace as NS
import numpy as np
import ordinaryshade as osh


@osh.compute(workgroup_size=(64,1,1))
def initialize(values:osh.storage_buffer(osh.vec4,access='read',binding=0),
               diffuse:osh.storage_image('rgba16f',access='write',binding=1),
               specular:osh.storage_image('rgba16f',access='write',binding=2),
               hdr:osh.storage_image('rgba16f',access='write',binding=3)):
    i=osh.global_invocation_id.x
    if i>=osh.u32(8):return
    pixel=osh.ivec2(i,0)
    diffuse.store(pixel,values[i])
    specular.store(pixel,osh.vec4(osh.f32(i),2.0,3.0,4.0))
    hdr.store(pixel,osh.vec4(9.0))


@osh.compute(workgroup_size=(64,1,1))
def readback(result:osh.storage_buffer(osh.vec4,access='write',binding=0),
             diffuse:osh.storage_image('rgba16f',access='read',binding=1),
             specular:osh.storage_image('rgba16f',access='read',binding=2),
             hdr:osh.storage_image('rgba16f',access='read',binding=3)):
    i=osh.global_invocation_id.x
    if i>=osh.u32(8):return
    pixel=osh.ivec2(i,0)
    result[i]=diffuse.load(pixel)
    result[i+osh.u32(8)]=specular.load(pixel)
    result[i+osh.u32(16)]=hdr.load(pixel)


def test_weighted_reconstruction_preserves_local_terms_and_fallback(monkeypatch):
    import vulkan as vk
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel,compile_compute
    from ordinarylight.pipeline.vulkan import VulkanResource
    from ordinarylight.pipeline.graph import VulkanGraph,reflected_operation
    from vxl8r_render.backends.face_reconstruction import SelectedDiffuseReconstruction
    with ExitStack() as stack:
        own=stack.enter_context
        runtime=own(VulkanRuntime())
        def buffer(data):
            data=np.asarray(data)
            return own(runtime.buffer(data.nbytes,data=data,memory='device'))
        anchors=buffer(np.array([0,0,0,0,1,1,0xffffffff,0xffffffff],np.uint32))
        ranges=buffer(np.array([[0,2],[2,1],[0,0]],np.uint32))
        samples=np.array([[0,0,0,4],[2,0,0,4],[4,1,0,2]],np.uint32)
        samples[:,2]=np.array([.25,.75,1],np.float32).view(np.uint32)
        samples=buffer(samples)
        direct_values=np.ones((8,4),np.float32)
        direct_values[:,:3]=np.arange(8)[:,None]
        direct_values[4:,3]=0
        direct=buffer(direct_values)
        values=direct_values.copy()
        values[0,:3]+=4  # Second selected sample contributes zero: do not renormalize.
        values[:,3]=8
        source=buffer(values);output=buffer(np.zeros((24,4),np.float32))
        images={name:own(runtime.image(8,1,format=vk.VK_FORMAT_R16G16B16A16_SFLOAT)) for name in ('diffuse','specular','hdr')}
        plan=NS(runtime=runtime,pixels=8,ranges=ranges,samples=samples,require_open=lambda:None,
                target=NS(render_extent=(8,1),face_capacity=3,anchors=anchors))
        reconstruction=own(SelectedDiffuseReconstruction(plan,direct,images))
        def operation(shader,storage):
            compiled=osh.compile(shader)
            kernel=own(VulkanKernel(runtime,compile_compute(compiled.source),
                {0:VulkanResource.buffer(storage),**{i:VulkanResource.image(images[n]) for i,n in enumerate(('diffuse','specular','hdr'),1)}}))
            return reflected_operation(kernel,compiled.reflection,workgroups=(1,1,1))
        init=operation(initialize,source);read=operation(readback,output)
        for eligible in (True,False):
            if not eligible:
                direct_values[:,3]=0;direct.upload(direct_values)
            def forbidden(*args,**kwargs):raise AssertionError('Allocation during reconstruction')
            with monkeypatch.context() as patch:
                patch.setattr(runtime,'buffer',forbidden);patch.setattr(runtime,'image',forbidden)
                patch.setattr(VulkanKernel,'__init__',forbidden)
                graph=VulkanGraph().add('init',init).add('reconstruct',reconstruction.operation(),after=('init',))
                graph.add('read',read,after=('reconstruct',)).compile().execute(runtime).wait()
            result=np.frombuffer(output.read(),np.float32).reshape(3,8,4)
            expected=values.copy()
            if eligible:expected[:4,:3]=direct_values[:4,:3]+1
            np.testing.assert_array_equal(result[0],expected)
            np.testing.assert_array_equal(result[1],np.column_stack((np.arange(8),np.full(8,2),np.full(8,3),np.full(8,4))))
            hdr=np.full((8,4),9,np.float32)
            if eligible:
                hdr[:4,:3]=expected[:4,:3]+result[1,:4,:3];hdr[:4,3]=1
            np.testing.assert_array_equal(result[2],hdr)
            np.testing.assert_array_equal(np.frombuffer(reconstruction.invalid.read(),np.uint32),[0 if eligible else 4,2,0])
