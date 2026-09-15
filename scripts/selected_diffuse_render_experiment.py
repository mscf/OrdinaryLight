"""Experimental actual ray reduction; homogeneous voxel faces, one sample/pixel.

Uses public compiler/kernel/primary operations and a public GI pipeline builder.
Native recorder interception is confined to the capture diagnostic harness.
"""
from ordinarylight.pipeline.graph import VulkanOperation,VulkanGraph
from ordinarylight.pipeline import RenderStage
from ordinarylight.pipeline.vulkan import VulkanResource
from vxl8r_render.backends.face_lighting import FaceLightingPlan,SelectedDiffuseSamples
from vxl8r_render.backends.visible_faces import VisibleFaceGroups
from vxl8r_render.backends.face_reconstruction import SelectedDiffuseReconstruction


def install(stack,enabled,slots,*,extent,budget=8,profile=False,quad_links=False,replay_workgroup=lambda:(8,8),omit_primary_analytic=lambda:False,single_capture=lambda:False,visibility_format=lambda:"full",conservative_capture=lambda:False,fixed_mask=None,compact_continuations=lambda:False):
    if fixed_mask not in (None,"none","all"):
        raise ValueError("Unknown diagnostic selection mask")
    from primary_visibility_experiment import install as capture_install
    states={};frame_views={};slot_states={};timers={}
    def extra(executor,allocation,slot):
        key=executor,allocation
        if key not in states:
            runtime=executor.core.runtime
            groups=stack.enter_context(VisibleFaceGroups(runtime,allocation,extent=extent,slots=slots(),quad_links=quad_links() if callable(quad_links) else quad_links,visibility_format=visibility_format()))
            plan=stack.enter_context(FaceLightingPlan(groups))
            mask=stack.enter_context(SelectedDiffuseSamples(plan))
            direct=stack.enter_context(runtime.buffer(extent[0]*extent[1]*16,memory='device'))
            reconstruct=stack.enter_context(SelectedDiffuseReconstruction(plan,direct,frame_views[runtime,slot]))
            reconstruct.diagnostic_fixed_mask=fixed_mask
            states[key]=(plan,mask,direct,reconstruct)
            if profile:
                from operation_timing_experiment import OperationTimer
                timers[key]=stack.enter_context(OperationTimer(runtime))
                reconstruct.diagnostic_timer=timers[key]
        slot_states[executor.core.runtime,slot]=states[key]
        plan,mask,direct,reconstruct=states[key]
        return {34:VulkanResource.buffer(mask.weights),35:VulkanResource.buffer(direct)}
    def selected(executor,allocation,sample):
        plan,mask,_,_=states[executor,allocation]
        a=plan.target.operation(sample=sample);b=plan.operation(budget=budget);c=mask.operation()
        if fixed_mask is not None:
            import vulkan as vk
            from ordinarylight.pipeline.vulkan import VulkanPass,VulkanResourceUse
            resource=VulkanResource.buffer(mask.weights)
            value=0 if fixed_mask=="none" else 0x3f800000
            def fill(command):
                vk.vkCmdFillBuffer(command,resource.handle,resource.offset,resource.size,value)
            c=VulkanOperation((VulkanPass('fixed_selection',(
                VulkanResourceUse(resource,vk.VK_PIPELINE_STAGE_TRANSFER_BIT,vk.VK_ACCESS_TRANSFER_WRITE_BIT),),fill),))
        if profile:
            a=timers[executor,allocation].wrap('group',a)
            b=timers[executor,allocation].wrap('plan',b)
            c=timers[executor,allocation].wrap('mask',c)
        return VulkanOperation((*a.passes,*b.passes,*c.passes),validate=mask.require_open)
    capture_install(stack,enabled,full_frame=True,after_capture=selected,
        replay_options={'primary_lobe_selection':True},replay_bindings=extra,
        measure=(lambda e,a,n,o:timers[e,a].wrap(n,o)) if profile else None,
        replay_workgroup=replay_workgroup,omit_primary_analytic=omit_primary_analytic,single_capture=single_capture,visibility_format=visibility_format,conservative_capture=conservative_capture,compact_continuations=compact_continuations)
    def builder(pipeline,frame):
        # Fixed masks test transport-state equivalence before stochastic face
        # reconstruction. Reconstructing from independently selected plan samples
        # would make an identical-shader control differ as well.
        runtime=frame.images['hdr'].runtime
        frame_views[runtime,frame.slot]=frame.images
        if fixed_mask is not None:
            return pipeline
        def reconstruct(context):
            _,_,_,target=slot_states[runtime,frame.slot]
            operation=target.operation()
            if profile:operation=target.diagnostic_timer.wrap('reconstruct',operation)
            frame.record_graph(VulkanGraph().add('selected_diffuse',operation).compile())
        target='gi.snapshot_hdr' if 'gi.snapshot_hdr' in pipeline.stage_names else 'gi.denoise'
        return pipeline.insert_before(target,RenderStage('gi.selected_diffuse',
            reads={'gi.hdr','gi.guides'},writes={'gi.hdr'},recorder=reconstruct))
    return builder,states


def summaries(states):
    import numpy as np
    rows=[]
    for plan,mask,direct,reconstruct in states.values():
        counters=np.frombuffer(plan.counters.read(),np.uint32)
        flags=np.frombuffer(direct.read(),np.float32).reshape(-1,4)[:,3]
        eligible=flags>0
        anchors=np.frombuffer(plan.target.anchors.read(),np.uint32)
        valid=anchors!=0xffffffff
        counts=np.bincount(anchors[valid],minlength=plan.target.face_capacity)
        fixed_mask=getattr(reconstruct,"diagnostic_fixed_mask",None)
        rejected=np.zeros_like(counts) if fixed_mask is not None else np.frombuffer(reconstruct.invalid.read(),np.uint32)
        if np.any((rejected!=0)&(rejected!=counts)):
            raise AssertionError('Experimental reconstruction requires homogeneous eligibility within each face')
        if profile_timer:=getattr(reconstruct,'diagnostic_timer',None):
            rows.append(dict(operation_ms=profile_timer.read()))
        rows.append(dict(fixed_mask=fixed_mask,reconstruction_enabled=fixed_mask is None,visibility_format=plan.target.visibility_format,quad_links=plan.target.quad_links,pixels=plan.pixels,eligible_pixels=int(eligible.sum()),eligible_continuations=int((flags>1).sum()),planned_samples=int(counters[0])))
    return rows
