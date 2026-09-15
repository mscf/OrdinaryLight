"""Join full-frame capture to vxl8r grouping and sample selection on the GPU."""
from ordinarylight.pipeline.graph import VulkanOperation
from face_lighting_plan_experiment import FaceLightingPlan
from vxl8r_render.backends.visible_faces import VisibleFaceGroups


def install(stack, enabled, slots, *, extent, budget=8):
    from primary_visibility_experiment import install as capture_install
    plans = {}
    def after_capture(executor, allocation, sample):
        # Extent is supplied by the diagnostic renderer; source allocation stays
        # alive via the surrounding capture install's ExitStack.
        width,height = extent
        key = executor, allocation
        if key not in plans:
            groups=stack.enter_context(VisibleFaceGroups(executor.core.runtime,allocation,
                extent=(width,height),slots=slots()))
            plans[key]=stack.enter_context(FaceLightingPlan(groups))
        plan=plans[key]
        grouping=plan.target.operation(sample=sample)
        selection=plan.operation(budget=budget)
        return VulkanOperation((*grouping.passes,*selection.passes),validate=plan.require_open)
    capture_install(stack,enabled,full_frame=True,after_capture=after_capture)
    return plans


def summaries(plans):
    """Explicit end-of-run readback, after the caller has waited for rendering."""
    import numpy as np
    results=[]
    for plan in plans.values():
        counts=np.frombuffer(plan.counters.read(),np.uint32)
        assert counts[3]==0,counts
        assert counts[0]<=counts[2]<=plan.pixels
        assert counts[1]<=counts[0]<=counts[1]*32
        results.append(dict(source_pixels=int(counts[2]),visible_faces=int(counts[1]),
            planned_samples=int(counts[0]),sample_fraction=float(counts[0]/max(int(counts[2]),1)),
            before_lighting=True,lighting_samples_reduced=False))
    return results
