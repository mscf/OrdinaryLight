from types import SimpleNamespace as NS

import pytest
import vulkan as vk
from ordinarylight.pipeline.vulkan import VulkanResource
from ordinarylight.runtime.primary import primary_operation
from ordinarylight.wavefront.primary_bindings import primary_bindings


def kernel():
    runtime = object()
    owner = NS(runtime=runtime, require_open=lambda: None)
    bindings, arrays = {}, {}
    for spec in primary_bindings():
        if spec.count > 1:
            arrays[spec.binding] = [
                (
                    VulkanResource(owner, "image", 100),
                    VulkanResource(owner, "sampler", 101),
                )
            ] * spec.count
        else:
            bindings[spec.binding] = VulkanResource(
                owner, spec.kind, spec.binding, 256 if spec.kind == "buffer" else 0
            )
    return NS(
        runtime=runtime,
        require_open=lambda: None,
        bindings=bindings,
        image_arrays={},
        sampled_image_arrays=arrays,
        sampled_image_layouts={},
    )


def test_primary_dispatch_and_aliased_resource_access(monkeypatch):
    k = kernel()
    # A previous-camera alias must retain both resource ranges in the barrier.
    k.bindings[18] = k.bindings[7].byte_range(64, 64)
    # Current/previous reservoirs can share a placeholder; retain read/write access.
    k.bindings[17] = k.bindings[16]
    calls = []
    k.bind = lambda cmd, data: calls.append((cmd, data))
    monkeypatch.setattr(vk, "vkCmdDispatch", lambda *args: calls.append(args))
    operation = primary_operation(k, bytes(176), workgroups=(3, 2, 1))
    stage = operation.passes[0]
    assert (
        stage.workgroups is None
    )  # Recorder dispatches; scheduler must not repeat it.
    assert len([u for u in stage.uses if u.resource.handle == 100]) == 1
    use = next(u for u in stage.uses if u.resource == k.bindings[16])
    assert use.access == vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT
    assert next(u for u in stage.uses if u.resource.handle == 7).resource.size == 256
    stage.record("command")
    assert calls == [("command", bytes(176)), ("command", 3, 2, 1)]


@pytest.mark.parametrize(
    "change", ["missing", "array", "runtime", "kind", "constants", "groups"]
)
def test_primary_rejects_invalid_contract(change):
    k = kernel()
    constants, groups = bytes(176), (1, 1, 1)
    if change == "missing":
        del k.bindings[23]
    elif change == "array":
        k.sampled_image_arrays[29] = []
    elif change == "runtime":
        k.runtime = object()
    elif change == "kind":
        k.bindings[0] = k.bindings[1]
    elif change == "constants":
        constants = bytes(175)
    else:
        groups = (0, 1, 1)
    with pytest.raises(ValueError):
        primary_operation(k, constants, workgroups=groups)


def test_visibility_sample_plane_bounds_and_barrier():
    import struct
    k = kernel()
    owner = k.bindings[1].owner
    k.bindings[33] = VulkanResource(owner, 'buffer', 333, 112 * 5 * 3 * 2)
    constants = struct.pack('8I', 5, 3, 0, 0, 5, 3, 2, 1) + bytes(144)
    operation = primary_operation(k, constants, workgroups=(1, 1, 1))
    use = next(u for u in operation.passes[0].uses if u.resource.handle == 333)
    assert use.access == vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT
    constants = struct.pack('8I', 5, 3, 0, 0, 5, 3, 3, 2) + bytes(144)
    with pytest.raises(ValueError, match='sample plane'):
        primary_operation(k, constants, workgroups=(1, 1, 1))


def test_capture_consumer_replay_orders_cache_access():
    import struct
    from ordinarylight.pipeline.graph import VulkanGraph
    from ordinarylight.pipeline.vulkan import VulkanPass,VulkanResourceUse
    k=kernel()
    k.bindings[33]=VulkanResource(k.bindings[1].owner,'buffer',333,112*15)
    constants=struct.pack('<8I',5,3,0,0,5,3,1,0)+bytes(144)
    capture=primary_operation(k,constants,workgroups=(1,1,1),visibility='capture')
    replay=primary_operation(k,constants,workgroups=(1,1,1),visibility='replay')
    for operation,access in ((capture,vk.VK_ACCESS_SHADER_WRITE_BIT),(replay,vk.VK_ACCESS_SHADER_READ_BIT)):
        assert next(u.access for u in operation.passes[0].uses if u.resource.handle==333)==access
    consumer=VulkanPass('consume',(VulkanResourceUse(k.bindings[33],
        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,vk.VK_ACCESS_SHADER_READ_BIT),),lambda command:None)
    graph=VulkanGraph().add('capture',capture).add('consumer',consumer,after=('capture',)).add('replay',replay,after=('consumer',))
    assert graph.compile().order==('capture','consumer','replay')
    with pytest.raises(ValueError):primary_operation(k,constants,workgroups=(1,1,1),visibility='invalid')
    del k.bindings[33]
    with pytest.raises(ValueError):primary_operation(k,constants,workgroups=(1,1,1),visibility='capture')


@pytest.mark.parametrize('change', ('valid','missing','short','samples','mode'))
def test_selected_diffuse_contract(change):
    import struct
    k=kernel()
    owner=next(iter(k.bindings.values())).owner
    for binding,size in ((33,112),(34,8),(35,16)):
        k.bindings[binding]=VulkanResource(owner,'buffer',binding,size)
    constants=bytearray(176)
    struct.pack_into('<8I',constants,0,1,1,0,0,1,1,1,0)
    visibility='replay'
    if change=='missing':del k.bindings[35]
    if change=='short':k.bindings[34]=VulkanResource(owner,'buffer',34,4)
    if change=='samples':struct.pack_into('<I',constants,24,2)
    if change=='mode':visibility='fused'
    if change=='valid':
        operation=primary_operation(k,constants,workgroups=(1,1,1),visibility=visibility)
        uses={u.resource.handle:u for u in operation.passes[0].uses}
        assert uses[34].access==vk.VK_ACCESS_SHADER_READ_BIT
        assert uses[35].access==vk.VK_ACCESS_SHADER_WRITE_BIT
    else:
        with pytest.raises(ValueError):
            primary_operation(k,constants,workgroups=(1,1,1),visibility=visibility)


@pytest.mark.parametrize('size,valid', ((99,False),(100,True)))
def test_distance_visibility_capacity(size,valid):
    import struct
    k=kernel()
    owner=next(iter(k.bindings.values())).owner
    k.bindings[33]=VulkanResource(owner,'buffer',33,size)
    constants=bytearray(176)
    struct.pack_into('<8I',constants,0,1,1,0,0,1,1,1,0)
    if valid:
        primary_operation(k,constants,workgroups=(1,1,1),visibility='capture',visibility_format='distance')
        with pytest.raises(ValueError):
            primary_operation(k,constants,workgroups=(1,1,1),visibility='capture')
    else:
        with pytest.raises(ValueError):
            primary_operation(k,constants,workgroups=(1,1,1),visibility='capture',visibility_format='distance')


def test_planes_visibility_capacity():
    import struct
    k=kernel()
    owner=next(iter(k.bindings.values())).owner
    constants=bytearray(176)
    struct.pack_into('<8I',constants,0,1,1,0,0,1,1,2,1)
    k.bindings[33]=VulkanResource(owner,'buffer',33,224)
    primary_operation(k,constants,workgroups=(1,1,1),visibility='capture',visibility_format='planes')
    k.bindings[33]=VulkanResource(owner,'buffer',33,223)
    with pytest.raises(ValueError):
        primary_operation(k,constants,workgroups=(1,1,1),visibility='capture',visibility_format='planes')


@pytest.mark.parametrize("size", (1000,1023,1024))
def test_compact_planes_capacity_includes_per_sample_padding(size):
    import struct
    k=kernel()
    owner=next(iter(k.bindings.values())).owner
    constants=bytearray(176)
    struct.pack_into('<8I',constants,0,5,1,0,0,5,1,2,1)
    k.bindings[33]=VulkanResource(owner,'buffer',33,size)
    if size==1024:
        primary_operation(k,constants,workgroups=(1,1,1),visibility='capture',visibility_format='distance_planes')
    else:
        with pytest.raises(ValueError):
            primary_operation(k,constants,workgroups=(1,1,1),visibility='capture',visibility_format='distance_planes')


@pytest.mark.parametrize("extension", (None,"geometry_resources","material_resources"))
def test_capture_omits_transport_outputs_but_preserves_extension_aliases(extension):
    import struct
    from ordinarylight.pipeline.vulkan import VulkanResourceUse
    k=kernel()
    owner=k.bindings[1].owner
    k.bindings[33]=VulkanResource(owner,'buffer',333,112)
    k.bindings[15]=VulkanResource(owner,'buffer',115,256)
    for binding in (30,31,32):
        k.bindings[binding]=VulkanResource(owner,'buffer',binding,256)
    constants=struct.pack('<8I',1,1,0,0,1,1,1,0)+bytes(144)
    rw=vk.VK_ACCESS_SHADER_READ_BIT|vk.VK_ACCESS_SHADER_WRITE_BIT
    if extension:
        setattr(k,extension,NS(uses=(VulkanResourceUse(k.bindings[1],vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,rw),)))
    capture=primary_operation(k,constants,workgroups=(1,1,1),visibility='capture')
    uses={u.resource.handle:u for u in capture.passes[0].uses}
    for binding in (5,6,8,9,16,21,23,30,31):assert binding not in uses
    assert (1 in uses)==bool(extension)
    if extension:assert uses[1].access==rw
    assert uses[333].access==vk.VK_ACCESS_SHADER_WRITE_BIT
    assert uses[115].access==rw
    assert uses[7].access==vk.VK_ACCESS_SHADER_READ_BIT
    replay=primary_operation(k,constants,workgroups=(1,1,1),visibility='replay')
    replay_uses={u.resource.handle:u for u in replay.passes[0].uses}
    assert replay_uses[1].access==rw
    assert replay_uses[333].access==vk.VK_ACCESS_SHADER_READ_BIT


def test_capture_still_validates_unused_descriptor_runtime():
    import struct
    k=kernel()
    k.bindings[33]=VulkanResource(k.bindings[1].owner,'buffer',333,112)
    k.bindings[1]=VulkanResource(NS(runtime=object(),require_open=lambda:None),'buffer',1,256)
    constants=struct.pack('<8I',1,1,0,0,1,1,1,0)+bytes(144)
    with pytest.raises(ValueError,match='share a runtime'):
        primary_operation(k,constants,workgroups=(1,1,1),visibility='capture')


@pytest.mark.parametrize('control_size,index_size,valid',[(16,60,True),(12,60,False),(16,56,False)])
def test_compact_continuation_tile_capacity(control_size,index_size,valid):
    import struct
    k=kernel()
    owner=k.bindings[1].owner
    for binding,size in ((33,112*15),(34,8*15),(35,16*15),(36,control_size),(37,index_size)):
        k.bindings[binding]=VulkanResource(owner,'buffer',binding+1000,size)
    constants=struct.pack('<8I',5,3,0,0,5,3,1,0)+bytes(144)
    if valid:
        operation=primary_operation(k,constants,workgroups=(1,1,1),visibility='replay')
        assert any(u.resource==k.bindings[37] for u in operation.passes[0].uses)
    else:
        with pytest.raises(ValueError,match='complete tile'):
            primary_operation(k,constants,workgroups=(1,1,1),visibility='replay')
