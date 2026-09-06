"""Ray diagnostics through the same traversal used by multi-bounce transport."""

import struct

import numpy as np

HIT_DTYPE = np.dtype(
    [
        ("position_distance", "<f4", (4,)),
        ("geometric_normal", "<f4", (4,)),
        ("shading_normal", "<f4", (4,)),
        ("identity", "<u4", (4,)),
        ("boundary", "<u4", (4,)),
    ]
)


class VulkanRayQuery:
    """Persistent nearest-hit queries without transport, accumulation or readback.

    Inputs are two vec4s per ray (origin and unit direction); hits use HIT_DTYPE.
    GPU producers can bind inputs, declare writes and supply dependencies. The
    caller owns input validity. Recreate this client after scene bindings change.
    """

    def __init__(self, scene, origins, directions):
        from ..runtime import VulkanKernel, compile_compute
        from ..pipeline.vulkan import VulkanResource
        from ._shaders import scene_source, SCENE_BINDINGS

        origins = np.asarray(origins, np.float32)
        directions = np.asarray(directions, np.float32)
        if (
            origins.ndim != 2
            or origins.shape[1] != 3
            or origins.shape != directions.shape
            or not len(origins)
            or not np.isfinite(origins).all()
            or not np.isfinite(directions).all()
            or not np.allclose(np.linalg.norm(directions, axis=1), 1, rtol=1e-5)
        ):
            raise ValueError("Supply finite (N,3) origins and normalized directions")
        packed = np.zeros((len(origins), 2, 4), np.float32)
        packed[:, 0, :3] = origins
        packed[:, 1, :3] = directions
        source = (
            scene_source(scene)
            + """
    struct RayInput { vec4 origin; vec4 direction; };
    layout(set=0,binding=8,std430) readonly buffer Rays { RayInput rays[]; };
    layout(set=0,binding=9,std430) writeonly buffer Hits { OrdinaryLightHit hits[]; };
    layout(push_constant) uniform Constants { uint count; float t_min; float t_max; float tolerance; uint max_steps; } pc;
    void main() {
        uint i=gl_GlobalInvocationID.x; if(i>=pc.count) return;
        hits[i]=ordinarylightIntersect(rays[i].origin.xyz,rays[i].direction.xyz,pc.t_min,pc.t_max,pc.tolerance,pc.max_steps);
    }
    """
        )
        self.scene, self.runtime = scene, scene.runtime
        self.count = len(origins)
        self.inputs = self.hits = self.kernel = None
        self.closed = False
        self.last_completion = None
        with self.runtime.lock:
            scene.require_open()
            import vulkan as vk

            limit = vk.vkGetPhysicalDeviceProperties(
                self.runtime.physical_device
            ).limits.maxComputeWorkGroupCount[0]
            if (self.count + 63) // 64 > limit:
                raise ValueError("Ray count exceeds the device dispatch limit")
            self.revision = scene.binding_revision
            self.runtime.retain(self)
            scene._borrowers.add(self)
            try:
                self.inputs = self.runtime.buffer(packed.nbytes, data=packed)
                self.hits = self.runtime.buffer(self.count * HIT_DTYPE.itemsize)
                bindings = {
                    i: scene.resource(name) for i, name in enumerate(SCENE_BINDINGS)
                }
                bindings.update(scene.custom_bindings)
                bindings.update(
                    {
                        8: VulkanResource.buffer(self.inputs),
                        9: VulkanResource.buffer(self.hits),
                    }
                )
                self.bindings = bindings
                self.kernel = VulkanKernel(
                    self.runtime,
                    compile_compute(source),
                    bindings,
                    push_constant_size=20,
                )
            except Exception:
                self.close()
                raise

    def require_open(self):
        self.scene.require_open()
        if self.closed:
            raise RuntimeError("Ray query is closed")
        if self.revision != self.scene.binding_revision:
            raise RuntimeError("Scene bindings changed; recreate ray query")

    def operation(
        self, *, t_min=1e-4, t_max=1e6, tolerance=1e-5, max_steps=256, after=()
    ):
        from operator import index
        from ..pipeline.graph import VulkanOperation
        from ..pipeline.vulkan import VulkanPass
        from ._custom_resources import resource_uses

        with self.runtime.lock:
            self.require_open()
            max_steps = index(max_steps)
            if (
                not np.isfinite([t_min, t_max, tolerance]).all()
                or not 0 <= t_min < t_max
                or tolerance <= 0
                or not 1 <= max_steps <= 65536
            ):
                raise ValueError("Invalid intersection limits")
            push = struct.pack("IfffI", self.count, t_min, t_max, tolerance, max_steps)

            def submitted(completion):
                self.last_completion = completion

            return VulkanOperation(
                [
                    VulkanPass(
                        "intersections",
                        resource_uses(self.bindings, writable=(9,)),
                        lambda command: self.kernel.bind(command, push),
                        ((self.count + 63) // 64, 1, 1),
                    )
                ],
                validate=self.require_open,
                submitted=submitted,
                dependencies=lambda: tuple(after),
            )

    def read(self):
        with self.runtime.lock:
            self.require_open()
            if self.last_completion is None:
                raise RuntimeError("Submit a ray query before reading hits")
            self.last_completion.wait()
            return np.frombuffer(self.hits.read(), HIT_DTYPE).copy()

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self.last_completion:
                self.last_completion.wait()
            if self.kernel:
                self.kernel.close()
            for buffer in (self.inputs, self.hits):
                if buffer is not None:
                    buffer.close()
            self.scene._borrowers.discard(self)
            self.runtime.release(self)
            self.closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()


def intersect_rays(
    scene,
    origins,
    directions,
    *,
    t_min=1e-4,
    t_max=1e6,
    tolerance=1e-5,
    max_steps=256,
    after=(),
):
    """One-shot convenience query; use VulkanRayQuery for repeated GPU work."""
    with VulkanRayQuery(scene, origins, directions) as query:
        query.operation(
            t_min=t_min,
            t_max=t_max,
            tolerance=tolerance,
            max_steps=max_steps,
            after=after,
        ).execute(scene.runtime).wait()
        return query.read()
