"""Headless pre-lighting grouping/selection checks; no HDR resources exist."""
from contextlib import ExitStack
import numpy as np
import pytest


@pytest.mark.parametrize('cache_format',('full','distance','planes','distance_planes'))
@pytest.mark.parametrize('coherent',(False,True))
@pytest.mark.parametrize('quad_links',(False,True))
@pytest.mark.parametrize('slots',(3,90000))
@pytest.mark.parametrize('budget',(1,8,32))
def test_visibility_only_grouping_and_plan(slots,budget,monkeypatch,quad_links,coherent,cache_format):
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel
    from ordinarylight.wavefront import PRIMARY_VISIBILITY_DTYPE
    from ordinarylight.pipeline.graph import VulkanGraph
    from vxl8r_render.backends.visible_faces import VisibleFaceGroups
    from face_lighting_plan_experiment import FaceLightingPlan
    w,h=45,39;p=w*h
    records=np.zeros((3,p),PRIMARY_VISIBILITY_DTYPE)
    rank=np.arange(p)//13 if coherent else np.arange(p)
    records['identity'][:,:,1]=rank%3
    records['identity'][:,:,2]=(rank//7)%6
    records[0]['address'][:,3]=(np.arange(p)%7!=0)*2
    records[2]=records[0]
    records[2]['identity'][::11,1]=slots
    records[2]['identity'][::13,2]=6
    records[2]['identity'][::17,0]=1
    # Highest valid slot exercises the direct/hash-domain boundary.
    records[2]['identity'][1]=[0,slots-1,5,0]
    records[2]['address'][1,3]=2
    with ExitStack() as stack:
        runtime=stack.enter_context(VulkanRuntime())
        words=records.view(np.uint32).reshape(3,p,28)
        packed=np.concatenate((words[:,:,3:4],words[:,:,4:]),axis=2).copy()
        planes=words.reshape(3,p,7,4).transpose(0,2,1,3).copy()
        compact=np.zeros((3,6*p+(p+3)//4,4),np.uint32)
        compact[:,:6*p]=planes[:,1:].reshape(3,6*p,4)
        compact[:,6*p:].reshape(3,-1)[:,:p]=words[:,:,3]
        upload=compact if cache_format=='distance_planes' else planes if cache_format=='planes' else packed if cache_format=='distance' else records
        source=stack.enter_context(runtime.buffer(upload.nbytes,data=upload))
        target=stack.enter_context(VisibleFaceGroups(runtime,source,extent=(w,h),slots=slots,quad_links=quad_links,visibility_format=cache_format))
        plan=stack.enter_context(FaceLightingPlan(target))
        for sample in (0,1,2,0):
            identity=records[sample]['identity']
            valid=(records[sample]['address'][:,3]!=0)&(identity[:,0]==0)&(identity[:,1]<slots)&(identity[:,2]<6)
            def forbidden(*a,**kw):raise AssertionError('GPU allocation during operation')
            with monkeypatch.context() as patch:
                patch.setattr(runtime,'buffer',forbidden);patch.setattr(runtime,'image',forbidden)
                patch.setattr(VulkanKernel,'__init__',forbidden)
                grouping=target.operation(sample=sample)
                selection=plan.operation(budget=budget,seed=sample)
                assert all(u.resource.kind=='buffer' for op in (grouping,selection) for stage in op.passes for u in stage.uses)
                graph=VulkanGraph().add('groups',grouping).add('plan',selection,after=('groups',))
                graph.compile().execute(runtime).wait()
            counters,samples,ranges=plan.read()
            anchors=np.frombuffer(target.anchors.read(),np.uint32)
            np.testing.assert_array_equal(anchors!=0xffffffff,valid)
            keys=np.frombuffer(target.keys.read(),np.uint32)
            faces=anchors[valid]
            actual=faces if target.capacity>=slots*6 else keys[faces]//target.stripes
            np.testing.assert_array_equal(actual,identity[valid,1]*6+identity[valid,2])
            unique,pop=np.unique(faces,return_counts=True)
            assert int(counters[2])==int(valid.sum())
            assert int(counters[1])==len(unique)
            assert int(counters[0])==sum(min(int(n),budget) for n in pop)
            assert len(np.unique(samples[:,0]))==len(samples)
            assert np.count_nonzero(ranges[:,1])==len(unique)
            for face,count in zip(unique,pop):
                base,n=map(int,ranges[face]);chosen=samples[base:base+n]
                assert n==min(int(count),budget)
                assert np.all(valid[chosen[:,0]])
                assert np.all(anchors[chosen[:,0]]==face)
                assert np.all(chosen[:,3]==count)
                np.testing.assert_allclose(chosen[:,2].copy().view(np.float32).sum(),1.0,atol=1e-6)
        with pytest.raises(ValueError):target.operation(sample=3)
        with pytest.raises(ValueError):target.operation(sample=-1)
