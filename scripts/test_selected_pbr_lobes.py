"""Actual GPU face selections consumed by the separable PBR sampler.

Incident lighting is analytic in this estimator test. No scene rays are traced.
"""
from contextlib import ExitStack
import struct
import numpy as np
import ordinaryshade as osh
import pytest
from ordinarylight.shaders.pbr_lobe_programs import PbrLobeSample,pbrLobeProbability,pbrLobeValue,pbrLobePdf,samplePbrLobe
from test_pbr_lobe_estimator import Constants,incident,quadrature


@osh.compute(workgroup_size=(64,1,1))
def masked_probe(randoms: osh.storage_buffer(osh.vec4,access='read',binding=0),
                 weights: osh.storage_buffer(osh.vec2,access='read',binding=1),
                 results: osh.storage_buffer(PbrLobeSample,access='write',binding=2),
                 full: osh.storage_buffer(PbrLobeSample,access='write',binding=3),
                 pc: osh.push_constants(Constants)):
    i=(osh.workgroup_id.y*osh.num_workgroups.x+osh.workgroup_id.x)*osh.u32(64)+osh.local_invocation_index
    if i>=osh.array_length(randoms):return
    results[i]=samplePbrLobe(pc.base_roughness.rgb,pc.base_roughness.w,pc.view_metallic.w,pc.options.x,
        osh.vec3(0.0,0.0,1.0),-pc.view_metallic.xyz,randoms[i].xyz,weights[i].x>0.0)
    full[i]=samplePbrLobe(pc.base_roughness.rgb,pc.base_roughness.w,pc.view_metallic.w,pc.options.x,
        osh.vec3(0.0,0.0,1.0),-pc.view_metallic.xyz,randoms[i].xyz,True)


@pytest.mark.parametrize('quad_links',(False,True))
@pytest.mark.parametrize('budget',(1,8,32))
def test_gpu_selected_lobes_preserve_energy_and_specular(budget,quad_links):
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel,compile_compute
    from ordinarylight.wavefront import PRIMARY_VISIBILITY_DTYPE
    from ordinarylight.pipeline.graph import VulkanGraph,reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    from vxl8r_render.backends.visible_faces import VisibleFaceGroups
    from vxl8r_render.backends.face_lighting import FaceLightingPlan,SelectedDiffuseSamples
    w,h=512,512;count=w*h
    faces=np.arange(count)//127
    slots=int(faces.max())//6+1
    hits=np.zeros(count,PRIMARY_VISIBILITY_DTYPE)
    hits['identity'][:,1]=faces//6;hits['identity'][:,2]=faces%6
    hits['address'][:,3]=2
    randoms=np.random.default_rng(46823).random((count,4),dtype=np.float32)
    view=np.array([np.sqrt(1-.7*.7),0,.7]);base=(.8,.25,.1)
    compiled=osh.compile(masked_probe,helpers=(pbrLobeProbability,pbrLobeValue,pbrLobePdf,samplePbrLobe))
    with ExitStack() as stack:
        runtime=stack.enter_context(VulkanRuntime())
        source=stack.enter_context(runtime.buffer(hits.nbytes,data=hits,memory='device'))
        groups=stack.enter_context(VisibleFaceGroups(runtime,source,extent=(w,h),slots=slots,quad_links=quad_links))
        plan=stack.enter_context(FaceLightingPlan(groups))
        selection=stack.enter_context(SelectedDiffuseSamples(plan))
        rng=stack.enter_context(runtime.buffer(randoms.nbytes,data=randoms,memory='device'))
        out=stack.enter_context(runtime.buffer(count*32,memory='device'))
        full=stack.enter_context(runtime.buffer(count*32,memory='device'))
        kernel=stack.enter_context(VulkanKernel(runtime,compile_compute(compiled.source),
            {0:VulkanResource.buffer(rng),1:VulkanResource.buffer(selection.weights),2:VulkanResource.buffer(out),3:VulkanResource.buffer(full)},push_constant_size=48))
        push=struct.pack('<12f',*base,.6,*view,0,1,0,0,0)
        sample=reflected_operation(kernel,compiled.reflection,workgroups=(4096,1,1),push_constants=push)
        graph=VulkanGraph().add('groups',groups.operation()).add('plan',plan.operation(budget=budget,seed=budget),after=('groups',))
        graph.add('mask',selection.operation(),after=('plan',)).add('sample',sample,after=('mask',))
        graph.compile().execute(runtime).wait()
        masked=np.frombuffer(out.read(),np.float32).reshape(count,8)
        baseline=np.frombuffer(full.read(),np.float32).reshape(count,8)
        weights=np.frombuffer(selection.weights.read(),np.float32).reshape(count,2)
        assert np.isfinite(masked).all()
        specular=baseline[:,7]==1
        np.testing.assert_array_equal(masked[specular],baseline[specular])
        np.testing.assert_array_equal(masked[~specular & (weights[:,0]==0),4:7],0)
        counts,samples,ranges=plan.read()
        assert np.count_nonzero(weights[:,0])==int(counts[0])
        np.testing.assert_array_equal(weights[samples[:,0],0],samples[:,2].copy().view(np.float32))
        np.testing.assert_array_equal(weights[samples[:,0],1],samples[:,3])
        # Selection weights reconstruct a face mean. Multiply by population
        # only here to compare the population-weighted image mean to quadrature.
        multiplier=np.where(specular,1.,weights[:,0]*weights[:,1])
        spatial=.3+(np.arange(count)%29)/29
        estimate=masked[:,4:7]*incident(masked[:,:3])*multiplier[:,None]*spatial[:,None]
        target=quadrature(base,.6,0,view,1).sum(axis=0)*spatial.mean()
        stderr=estimate.std(axis=0,ddof=1)/np.sqrt(count)
        assert np.all(np.abs(estimate.mean(axis=0)-target)<6*stderr+3e-4),(estimate.mean(axis=0),target,stderr)
        active=np.any(masked[:,4:7]>0,axis=1)
        print(dict(budget=budget,selected=int(counts[0]),pixels=count,nonzero_continuations=int(active.sum()),
                   estimate=estimate.mean(axis=0).tolist(),target=target.tolist(),scene_rays_traced=False))
        # Rebuild an empty frame through the exact same allocations.
        hits['address'][:,3]=0;source.upload(hits)
        empty=VulkanGraph().add('groups',groups.operation()).add('plan',plan.operation(budget=budget),after=('groups',))
        empty.add('mask',selection.operation(),after=('plan',)).compile().execute(runtime).wait()
        np.testing.assert_array_equal(np.frombuffer(selection.weights.read(),np.float32),0)
