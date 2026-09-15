"""Pass validation must preserve overlap checks without all-pairs comparisons."""
import random
import pytest
import vulkan as vk
from ordinarylight.pipeline.vulkan import VulkanPass,VulkanResource,VulkanResourceUse


def use(kind,handle,offset=0,size=16):
    resource=VulkanResource(None,kind,handle,size if kind=='buffer' else 0,
                            offset=offset if kind=='buffer' else 0)
    return VulkanResourceUse(resource,vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        vk.VK_ACCESS_SHADER_READ_BIT,vk.VK_IMAGE_LAYOUT_GENERAL if kind=='image' else None)


def reference_accepts(uses):
    for i,a in enumerate(uses):
        a=a.resource
        for b in uses[:i]:
            b=b.resource
            if (a.kind,a.handle)==(b.kind,b.handle):
                if a.kind!='buffer' or max(a.offset,b.offset)<min(a.offset+a.size,b.offset+b.size):return False
    return True


@pytest.mark.parametrize('intervals,valid',[
    ([(32,16),(0,16),(16,16)],True),
    ([(32,16),(0,33),(16,16)],False),
    ([(0,64),(16,16)],False),
    ([(0,16),(0,16)],False),
    ([(0,1),(2,1),(1,1)],True),
])
def test_buffer_ranges_allow_adjacency_and_reject_overlap(intervals,valid):
    uses=tuple(use('buffer',1,start,size) for start,size in intervals)
    if valid:VulkanPass('ranges',uses,lambda _:None)
    else:
        with pytest.raises(ValueError,match='duplicate resources'):VulkanPass('ranges',uses,lambda _:None)


@pytest.mark.parametrize('kind',['image','sampler','acceleration_structure'])
def test_non_buffer_duplicates_and_same_handle_different_kinds(kind):
    with pytest.raises(ValueError,match='duplicate resources'):
        VulkanPass('duplicate',(use(kind,1),use(kind,1)),lambda _:None)
    VulkanPass('distinct',(use(kind,1),use('buffer',1)),lambda _:None)


def test_grouped_validation_matches_all_pairs_reference():
    rng=random.Random(164)
    for _ in range(2000):
        uses=tuple(use(rng.choice(('buffer','image','sampler','acceleration_structure')),
                      rng.randrange(20),rng.randrange(50),rng.randrange(1,25))
                   for _ in range(rng.randrange(30)))
        if reference_accepts(uses):VulkanPass('random',uses,lambda _:None)
        else:
            with pytest.raises(ValueError,match='duplicate resources'):VulkanPass('random',uses,lambda _:None)


def test_large_unique_resource_set_keeps_uses_and_recorder():
    uses=tuple(use('acceleration_structure',i) for i in range(1024))
    record=lambda _:None
    stage=VulkanPass('large',uses,record)
    assert stage.uses is uses and stage.record is record
