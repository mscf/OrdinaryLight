"""Diagnostic timing for the vxl8r-owned GPU face sample planner."""
import numpy as np
from vxl8r_render.backends.face_lighting import FaceLightingPlan, PlanConstants, select_samples

def profile_plan(renderer, *, budget=8):
    """Measure a plan over the final reference frame, not a faster GI renderer."""
    from ordinarylight.pipeline.graph import VulkanGraph
    from vxl8r_render.backends.gpu_timing import GpuGraphTimer
    target = renderer._last_target[0]  # Diagnostic access to vxl8r-owned grouping.
    with FaceLightingPlan(target) as plan, GpuGraphTimer(renderer.runtime) as timer:
        rows = []
        for index in range(10):
            VulkanGraph().add('native_gi',plan.operation(budget=budget,seed=index)).compile().execute(renderer.runtime).wait()
            if index >= 2:
                rows.append(timer.last['total'])
        counters, samples, ranges = plan.read()
        anchors = np.frombuffer(target.anchors.read(),np.uint32)
        valid = anchors != 0xffffffff
        faces, populations = np.unique(anchors[valid],return_counts=True)
        expected = np.minimum(populations,budget)
        assert int(counters[0]) == int(expected.sum())
        assert int(counters[1]) == len(faces)
        assert int(counters[2]) == int(valid.sum())
        np.testing.assert_array_equal(ranges[faces,1],expected)
        assert np.count_nonzero(ranges[:,1]) == len(faces)
        assert np.all(samples[:,0] < len(anchors))
        assert np.all(anchors[samples[:,0]] == samples[:,1])
        assert len(np.unique(samples[:,0])) == len(samples)
        if len(faces):
            ordered = np.argsort(ranges[faces,0])
            offsets = ranges[faces[ordered],0]
            counts = ranges[faces[ordered],1]
            np.testing.assert_array_equal(offsets,np.cumsum(np.r_[0,counts[:-1]],dtype=np.uint64))
            weights = samples[:,2].copy().view(np.float32)
            np.testing.assert_allclose(np.add.reduceat(weights,offsets),1.0,atol=1e-6)
        return dict(budget=budget, source_pixels=int(counters[2]), visible_faces=int(counters[1]),
            planned_samples=int(counters[0]), sample_fraction=float(counters[0]/max(int(counters[2]),1)),
            planner_gpu_median_ms=float(np.median(rows)),
            pixels_per_face_percentiles=np.percentile(populations,[50,90,99]).tolist() if len(populations) else [],
            snapshot_only=True, lighting_rays_reduced=False, plan_validated=True)
