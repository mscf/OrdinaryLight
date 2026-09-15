"""Native-style dormant placeholders must not become writable resolve scratch."""
from contextlib import ExitStack
from importlib.resources import files
import os
from types import SimpleNamespace as NS
import numpy as np
import pytest
import vulkan as vk
from ordinarylight.runtime.path_resolve import path_resolve_operation
from ordinarylight.pipeline.vulkan import VulkanResource


def fake_kernel():
    owner=NS(require_open=lambda:None)
    bindings={i:VulkanResource(owner,'image' if i==1 else 'buffer',i,256) for i in range(6)}
    bindings[3]=bindings[5]=bindings[2]
    return NS(bindings=bindings,require_open=lambda:None)


@pytest.mark.parametrize('sampled', [False,True])
def test_signal_only_uses_exclude_placeholder_writes(sampled):
    k=fake_kernel()
    op=path_resolve_operation(k,capacity=2,extent=(2,1),reservoir_extent=(1,1),path_count=2,
        capture_secondary=True,sampled_indirect=sampled,seed_reservoirs=False)
    uses={u.resource.handle:u.access for u in op.passes[0].uses}
    assert set(uses)==({0,1} if sampled else {0,1,2})
    if not sampled:assert uses[2]==vk.VK_ACCESS_SHADER_READ_BIT|vk.VK_ACCESS_SHADER_WRITE_BIT
    with pytest.raises(ValueError,match='must not alias'):
        path_resolve_operation(k,capacity=2,extent=(2,1),reservoir_extent=(1,1),path_count=2,
            capture_secondary=True,sampled_indirect=sampled)


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH')!='1',reason='opt-in GPU')
@pytest.mark.parametrize('sampled',[False,True])
def test_aliased_native_placeholders_preserve_first_secondary_record(sampled):
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel
    from ordinarylight.pipeline.graph import VulkanGraph
    from ordinarylight.wavefront import HOT_PATH_STATE_DTYPE,SECONDARY_PATH_STATE_DTYPE
    with VulkanRuntime() as runtime,ExitStack() as stack:
        own=stack.enter_context
        data=np.arange(64,dtype=np.float32).reshape(2,32)/64
        secondary_data=data.view(SECONDARY_PATH_STATE_DTYPE).reshape(2).copy()
        secondary_data['position_valid'][:]=[1,2,3,1]
        secondary_data['normal_pdf'][:]=[0,1,0,.25]
        secondary_data['primary_position'][:]=[1,2,1,1.5]
        secondary_data['primary_radiance'][:]=[.1,.2,.3,.25]
        original=secondary_data.copy()
        paths_data=np.zeros(2,HOT_PATH_STATE_DTYPE)
        paths_data['metadata'][:,0]=[0,1]
        paths_data['radiance'][:,:3]=[[2,4,6],[4,8,12]]
        paths=own(runtime.buffer(paths_data.nbytes,data=paths_data))
        secondary=own(runtime.buffer(secondary_data.nbytes,data=secondary_data))
        camera=own(runtime.buffer(64,data=bytes(64)))
        hdr=own(runtime.image(2,1,format=vk.VK_FORMAT_R16G16B16A16_SFLOAT))
        b=VulkanResource.buffer
        kernel=own(VulkanKernel(runtime,files('ordinarylight.shaders').joinpath('wavefront_path_to_hdr.comp.spv').read_bytes(),
            {0:b(paths),1:VulkanResource.image(hdr),2:b(secondary),3:b(secondary),4:b(camera),5:b(secondary)},push_constant_size=32))
        for sample in range(2):
            op=path_resolve_operation(kernel,capacity=2,extent=(2,1),reservoir_extent=(2,1),path_count=2,
                capture_secondary=True,sampled_indirect=sampled,seed_reservoirs=False,sample_index=sample,sample_count=2)
            VulkanGraph().add('resolve',op).compile().execute(runtime).wait()
            if sample==0 or sampled:assert secondary.read()==original.tobytes()
        if not sampled:
            result=np.frombuffer(secondary.read(),SECONDARY_PATH_STATE_DTYPE)
            for name in original.dtype.names:
                if name not in ('diffuse_radiance_hit_distance','specular_radiance_hit_distance'):
                    np.testing.assert_array_equal(result[name],original[name])
            np.testing.assert_array_equal(result['diffuse_radiance_hit_distance'][:,:3],paths_data['radiance'][:,:3]*.75)
            np.testing.assert_array_equal(result['specular_radiance_hit_distance'][:,:3],paths_data['radiance'][:,:3]*.25)
            np.testing.assert_array_equal(result['diffuse_radiance_hit_distance'][:,3],2)


def test_native_reuse_toggle_changes_cached_dispatch():
    from collections import OrderedDict
    from unittest.mock import Mock,patch
    from ordinarylight.targets.vulkan import path_resolve_graph as adapter
    config=NS(wavefront_indirect_reuse_candidates=False,denoiser_sampled_indirect=True)
    executor=NS(core=NS(runtime=object(),config=config,window_frames=[{}]),capacity=1,
        _denoiser_signals_active=lambda:True,
        path_resolve_stages={0:((),NS(runtime=object()),OrderedDict())})
    with patch.object(adapter,'_NativeResolveKernel',return_value=NS(bindings={})), \
         patch.object(adapter,'path_resolve_operation') as operation, \
         patch.object(adapter,'VulkanGraph',return_value=Mock()):
        for enabled in (False,True,False):
            config.wavefront_indirect_reuse_candidates=enabled
            adapter.record_path_resolve(executor,object(),0,1,1,1)
        assert operation.call_count==2
        assert [call.kwargs['seed_reservoirs'] for call in operation.call_args_list]==[False,True]
