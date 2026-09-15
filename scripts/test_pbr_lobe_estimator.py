"""Headless GPU checks against the maintained BRDF and independent quadrature."""
from contextlib import ExitStack
from functools import lru_cache
import struct
import numpy as np
import ordinaryshade as osh
import pytest
from ordinarylight.shaders.pbr_lobe_programs import (
    PbrLobeSample, pbrLobeProbability,pbrLobeValue,pbrLobePdf,samplePbrLobe,pbrLobeMisWeights)
from generate_primary_shaders import ordinarylight_pbr_evaluate


@osh.structure
class Constants:
    base_roughness: osh.vec4
    view_metallic: osh.vec4
    options: osh.vec4


@osh.function
def environment(outgoing: osh.vec3) -> osh.vec3:
    patch=osh.power(osh.maximum(osh.dot(outgoing,osh.vec3(.6,.3,.7416198487095663)),0.0),8.0)
    return osh.vec3(.25+patch,.4+.3*patch,.6+.1*patch)


@osh.compute(workgroup_size=(64,1,1))
def probe(randoms: osh.storage_buffer(osh.vec4,access='read',binding=0),
          results: osh.storage_buffer(PbrLobeSample,access='write',binding=1),
          reference: osh.storage_buffer(osh.vec4,access='write',binding=2),
          pc: osh.push_constants(Constants)):
    i=(osh.workgroup_id.y*osh.num_workgroups.x+osh.workgroup_id.x)*osh.u32(64)+osh.local_invocation_index
    if i>=osh.array_length(randoms):return
    normal=osh.vec3(0.0,0.0,1.0)
    sample=samplePbrLobe(pc.base_roughness.rgb,pc.base_roughness.w,pc.view_metallic.w,
        pc.options.x,normal,-pc.view_metallic.xyz,randoms[i].xyz,randoms[i].w>0.0)
    outgoing=sample.direction_pdf.xyz
    combined=pbrLobeValue(pc.base_roughness.rgb,pc.base_roughness.w,pc.view_metallic.w,normal,pc.view_metallic.xyz,outgoing,False)
    combined=combined+pbrLobeValue(pc.base_roughness.rgb,pc.base_roughness.w,pc.view_metallic.w,normal,pc.view_metallic.xyz,outgoing,True)
    maintained=ordinarylight_pbr_evaluate(pc.base_roughness.rgb,pc.base_roughness.w,pc.view_metallic.w,normal,pc.view_metallic.xyz,outgoing)
    reference[i]=osh.vec4(combined-maintained,0.0)
    if pc.options.y>0.0:
        light_pdf=osh.maximum(outgoing.z,0.0)/3.14159265359
        mis=pbrLobeMisWeights(light_pdf*pc.options.z,sample.direction_pdf.w)
        sample.throughput_lobe.rgb=sample.throughput_lobe.rgb*mis.y
        u=randoms[i].y
        phi=6.28318530718*randoms[i].z
        light_direction=osh.vec3(osh.sqrt(u)*osh.cosine(phi),osh.sqrt(u)*osh.sine(phi),osh.sqrt(1.0-u))
        q_light=light_direction.z/3.14159265359
        probability=pbrLobeProbability(pc.base_roughness.rgb,pc.view_metallic.w)
        q_diffuse=(1.0-probability)*pbrLobePdf(pc.base_roughness.w,normal,pc.view_metallic.xyz,light_direction,False)
        q_specular=probability*pbrLobePdf(pc.base_roughness.w,normal,pc.view_metallic.xyz,light_direction,True)
        diffuse=pbrLobeValue(pc.base_roughness.rgb,pc.base_roughness.w,pc.view_metallic.w,normal,pc.view_metallic.xyz,light_direction,False)
        specular=pbrLobeValue(pc.base_roughness.rgb,pc.base_roughness.w,pc.view_metallic.w,normal,pc.view_metallic.xyz,light_direction,True)
        direct=diffuse*pbrLobeMisWeights(q_light*pc.options.z,q_diffuse).x+specular*pbrLobeMisWeights(q_light*pc.options.z,q_specular).x
        reference[i]=osh.vec4(direct*environment(light_direction)*(light_direction.z/q_light)*osh.mix(pc.options.x,1.0,pc.view_metallic.w),0.0)
    results[i]=sample


@lru_cache(maxsize=1)
def compiled_probe():
    from ordinarylight.runtime import compile_compute
    compiled=osh.compile(probe,helpers=(pbrLobeProbability,pbrLobeValue,pbrLobePdf,samplePbrLobe,ordinarylight_pbr_evaluate,pbrLobeMisWeights,environment))
    return compiled,compile_compute(compiled.source)


def incident(directions):
    # Smooth directional environment; tests more than constant white lighting.
    l=np.array([.6,.3,.7416198487095663])
    patch=np.maximum(directions@l,0)**8
    return np.stack((.25+patch,.4+.3*patch,.6+.1*patch),axis=-1)


def quadrature(base,roughness,metallic,view,occlusion):
    z,weights=np.polynomial.legendre.leggauss(192)
    z=(z+1)/2;weights=weights/2
    phi=(np.arange(768)+.5)*2*np.pi/768
    xx=np.sqrt(1-z*z)[:,None]*np.cos(phi)
    yy=np.sqrt(1-z*z)[:,None]*np.sin(phi)
    zz=np.broadcast_to(z[:,None],xx.shape)
    outgoing=np.stack((xx,yy,zz),axis=-1)
    half=view+outgoing;half/=np.linalg.norm(half,axis=-1,keepdims=True)
    nh=np.maximum(half[...,2],0);vh=np.maximum(half@view,0)
    f0=(1-metallic)*.04+metallic*np.array(base)
    fresnel=f0+(1-f0)*(1-np.clip(vh,0,1))[...,None]**5
    diffuse=(1-fresnel)*(1-metallic)*np.array(base)/np.pi
    a2=max(roughness*roughness,.0009)**2
    d=a2/np.maximum(np.pi*(nh*nh*(a2-1)+1)**2,1e-6)
    nv=view[2]
    gv=2*nv/max(nv+np.sqrt(a2+(1-a2)*nv*nv),1e-6)
    gl=2*zz/np.maximum(zz+np.sqrt(a2+(1-a2)*zz*zz),1e-6)
    specular=fresnel*(d*gv*gl/np.maximum(4*nv*zz,1e-6))[...,None]
    factor=(weights[:,None]*2*np.pi/768*zz)[...,None]*incident(outgoing)*((1-metallic)*occlusion+metallic)
    return np.stack(((diffuse*factor).sum(axis=(0,1)),(specular*factor).sum(axis=(0,1))))


@pytest.mark.parametrize('mis_samples',(0,1,4))
@pytest.mark.parametrize('roughness,metallic,cosine,occlusion',(
    (.25,0.,1.,1.),(.6,0.,.7,.35),(1.,0.,.2,1.),
    (.35,.65,.65,1.),(.6,1.,1.,1.),(.9,1.,.2,1.),
))
def test_lobe_energy_matches_integrated_brdf(roughness,metallic,cosine,occlusion,mis_samples):
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel
    from ordinarylight.pipeline.vulkan import VulkanResource
    from ordinarylight.pipeline.graph import reflected_operation
    base=(.8,.25,.1);view=np.array([np.sqrt(1-cosine*cosine),0,cosine])
    count=1048576
    randoms=np.random.default_rng(14673).random((count,4),dtype=np.float32)
    randoms[:,3]=1
    compiled,spirv=compiled_probe()
    with ExitStack() as stack:
        runtime=stack.enter_context(VulkanRuntime())
        inputs=stack.enter_context(runtime.buffer(randoms.nbytes,data=randoms,memory='device'))
        result=stack.enter_context(runtime.buffer(count*32,memory='device'))
        reference=stack.enter_context(runtime.buffer(count*16,memory='device'))
        kernel=stack.enter_context(VulkanKernel(runtime,spirv,{0:VulkanResource.buffer(inputs),1:VulkanResource.buffer(result),2:VulkanResource.buffer(reference)},push_constant_size=48))
        push=struct.pack('<12f',*base,roughness,*view,metallic,occlusion,int(mis_samples>0),mis_samples,0)
        def run():
            reflected_operation(kernel,compiled.reflection,workgroups=(4096,count//(4096*64),1),push_constants=push).execute(runtime).wait()
            return np.frombuffer(result.read(),np.float32).reshape(count,8).copy()
        output=run()
        assert np.isfinite(output).all()
        error=np.frombuffer(reference.read(),np.float32)
        if not mis_samples:
            assert np.max(np.abs(error))<2e-5
        values=output[:,4:7]*incident(output[:,:3])
        lobe=output[:,7]
        estimates=[]
        expected=quadrature(base,roughness,metallic,view,occlusion)
        if mis_samples:
            combined=values+error.reshape(count,4)[:,:3]
            stderr=combined.std(axis=0,ddof=1)/np.sqrt(count)
            assert np.all(np.abs(combined.mean(axis=0)-expected.sum(axis=0))<6*stderr+3e-4)
        for index in (0,1):
            samples=values*(lobe==index)[:,None]
            mean=samples.mean(axis=0)
            stderr=samples.std(axis=0,ddof=1)/np.sqrt(count)
            if not mis_samples:
                assert np.all(np.abs(mean-expected[index])<6*stderr+3e-4),(mean,expected[index],stderr)
            estimates.append(mean)
        # With diffuse disabled, the exact same random choices preserve every
        # specular sample and return null for diffuse events. No rerouting.
        randoms[:,3]=0;inputs.upload(randoms)
        spec_only=run()
        np.testing.assert_array_equal(spec_only[lobe==1],output[lobe==1])
        np.testing.assert_array_equal(spec_only[lobe==0,4:7],0)
        np.testing.assert_array_equal(spec_only[lobe==0,3],0)
        if roughness==1.:
            # Below-horizon specular draws are null, not cosine fallback draws.
            assert np.count_nonzero((lobe==1)&(output[:,3]==0))>count*.01
        print(dict(mis_samples=mis_samples,roughness=roughness,metallic=metallic,cosine=cosine,estimate=np.array(estimates).tolist(),quadrature=expected.tolist()))
