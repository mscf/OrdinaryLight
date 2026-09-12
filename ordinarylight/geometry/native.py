"""Typed application programs for native camera surface evaluation.

This module compiles geometry declarations without creating a Vulkan runtime.
VulkanNativeGeometryResources binds these declarations for native camera GI.
Optional smooth lossless optical boundaries share native primary/secondary
transport and visibility. Optional emitter callbacks own area-light sampling
and emissive-hit PDFs in the same native transport.
"""
from dataclasses import dataclass
import hashlib
import inspect
import keyword
import linecache
import re

import ordinaryshade as osh

from ..shaders.dynamic import _lock
from ..shaders.native_intersection_programs import (
    NativeIntersection, nativeIntersectionMiss, nativeIntersectCandidate,
    NativeOpticalBoundary, nativeEvaluateBoundary,
)
from ..shaders.native_surface_programs import nativeEvaluateMaterial
from ..shaders.native_emitter_programs import (NativeEmitterSample, nativeEmitterCount,
    nativeSelectEmitter, nativeEvaluateEmitter, nativeEmitterPdf)
from ..shaders.transport_programs import MaterialData


@dataclass(frozen=True)
class NativeGeometryBuffer:
    """An application-owned read-only std430 array at native descriptor set 2."""
    name: str
    element_type: object

    def __post_init__(self):
        if (not self.name.isascii() or not self.name.isidentifier()
                or keyword.iskeyword(self.name) or self.name.startswith(("gl_", "native", "__"))
                or self.name in {"scene_tlas", "materials", "vertices", "attributes", "push"}):
            raise ValueError("Use a non-reserved application buffer identifier")
        # OrdinaryShade validates the element type and storage layout.
        osh.storage_buffer(self.element_type, access="read", binding=0, set=2)


@dataclass(frozen=True)
class NativeEmitterProgram:
    """Typed callbacks owning the complete area-emitter sampling distribution.

    IDs are contiguous in [0, count). Evaluation maps a pair of uniform random
    coordinates to a surface point and returns the joint selection/area PDF.
    The hit PDF must describe the same distribution, including emitter selection.
    """
    count: object
    select: object
    evaluate: object
    pdf: object

    @property
    def functions(self):
        return (self.count, self.select, self.evaluate, self.pdf)


@dataclass(frozen=True, init=False)
class NativeGeometryProgram:
    """Compile typed intersection/material callbacks and resident buffer layouts.

    Declare callbacks with ``@osh.function(name='nativeIntersectCandidate')``
    and ``@osh.function(name='nativeEvaluateMaterial')``. Their signatures match
    the exported NativeIntersection and MaterialData contracts.
    An optional ``boundary`` callback exports ``nativeEvaluateBoundary`` and
    returns NativeOpticalBoundary. Its IOR pair is (outside, inside); the
    intersection geometric normal points outside. Additional
    helpers must also be typed OrdinaryShade functions. No shader source strings
    or opaque application externals are accepted.
    """

    buffers: tuple
    helpers: tuple
    intersection: object
    material: object
    boundary: object
    emitters: object
    compiled: object

    def __init__(self, intersection, material, *, buffers=(), helpers=(), boundary=None, emitters=None):
        object.__setattr__(self, "buffers", tuple(buffers))
        object.__setattr__(self, "helpers", tuple(helpers))
        if any(not isinstance(b, NativeGeometryBuffer) for b in self.buffers):
            raise TypeError("buffers must contain NativeGeometryBuffer declarations")
        if len({b.name for b in self.buffers}) != len(self.buffers):
            raise ValueError("Geometry buffer names must be unique")
        if any(not isinstance(h, osh.ShaderFunction) for h in self.helpers):
            raise TypeError("Geometry helpers must be typed OrdinaryShade functions")
        contracts = [(intersection, nativeIntersectCandidate), (material, nativeEvaluateMaterial)]
        if boundary is not None:
            contracts.append((boundary, nativeEvaluateBoundary))
        if emitters is not None:
            if not isinstance(emitters, NativeEmitterProgram):
                raise TypeError("emitters must be NativeEmitterProgram")
            contracts.extend(zip(emitters.functions, (nativeEmitterCount,
                nativeSelectEmitter, nativeEvaluateEmitter, nativeEmitterPdf)))
        for callback, contract in contracts:
            if not isinstance(callback, osh.ShaderFunction):
                raise TypeError("Geometry callbacks must be typed OrdinaryShade functions")
            if callback.__name__ != contract.__name__:
                raise ValueError(f"Geometry callback must export {contract.__name__}")
            actual = inspect.signature(callback.function, eval_str=True)
            required = inspect.signature(contract.function, eval_str=True)
            if (tuple(p.annotation for p in actual.parameters.values()) !=
                    tuple(p.annotation for p in required.parameters.values())
                    or actual.return_annotation != required.return_annotation):
                raise TypeError(f"Geometry callback signature must match {contract.__name__}")
        object.__setattr__(self, "intersection", intersection)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "emitters", emitters)
        object.__setattr__(self, "compiled", self._compile())

    def _compile(self):
        scope = {"osh": osh}
        parameters = []
        for binding, declaration in enumerate(self.buffers):
            type_name = f"_element_{binding}"
            scope[type_name] = declaration.element_type
            parameters.append(f"{declaration.name}: osh.storage_buffer({type_name}, access='read', binding={binding}, set=2)")
        source = "@osh.compute(workgroup_size=(1, 1, 1))\ndef geometry_declarations(" + ", ".join(parameters) + "):\n    pass\n"
        filename = '<ordinarylight-native-geometry-' + hashlib.sha256(source.encode()).hexdigest() + '>'
        with _lock:
            linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
            try:
                exec(compile(source, filename, 'exec'), scope)
                return osh.compile(scope['geometry_declarations'],
                                   helpers=(*self.helpers, self.intersection, self.material,
                                            *((self.boundary,) if self.boundary is not None else ()),
                                            *(self.emitters.functions if self.emitters is not None else ())),
                                   externals=(osh.external(nativeIntersectionMiss.function),))
            finally:
                linecache.cache.pop(filename, None)

    @property
    def source(self):
        """Generated declarations/helpers for inclusion in a native stage.

        Remove only mechanical entry-point/ABI declarations already supplied by
        the native stage. All executable callback bodies come from OrdinaryShade.
        """
        source = self.compiled.source
        source = source[:source.index('void main()')]
        source = re.sub(r'^#(?:version|extension)\b[^\n]*\n', '', source, flags=re.MULTILINE)
        source = re.sub(r'layout\(local_size_x\s*=.*?\) in;\n', '', source)
        for name in ('NativeIntersection', 'NativeOpticalBoundary', 'NativeEmitterSample', 'MaterialData'):
            source = re.sub(r'\bstruct ' + name + r'\s*\{[^}]*\};\s*', '', source)
        return source


__all__ = ['NativeGeometryBuffer', 'NativeGeometryProgram', 'VulkanNativeGeometryResources',
           'NativeIntersection', 'NativeOpticalBoundary', 'NativeEmitterProgram',
           'NativeEmitterSample', 'MaterialData', 'nativeIntersectionMiss']


def _native_buffer_resource(runtime, value):
    from ..runtime.resources import VulkanBuffer
    from ..pipeline.vulkan import VulkanResource

    resource = VulkanResource.buffer(value) if isinstance(value, VulkanBuffer) else value
    if (not isinstance(resource, VulkanResource) or resource.kind != 'buffer'
            or resource.descriptor not in (None, 'buffer')
            or getattr(resource.owner, 'runtime', None) is not runtime):
        raise ValueError('Geometry buffers require same-runtime VulkanBuffer allocations or storage-buffer resources')
    owner = resource.owner
    if not all(callable(getattr(owner, method, None))
               for method in ('require_open', 'retain', 'release')):
        raise ValueError('Geometry buffer owners must support require_open, retain and release')
    owner.require_open()
    return resource


class VulkanNativeGeometryResources:
    """Persistent native set-2 buffers; borrowed allocations remain app-owned.

    Values may be VulkanBuffer allocations or storage-buffer VulkanResource
    views from a same-runtime owner supporting require_open/retain/release.
    Descriptor byte ranges are preserved, and their owners are leased until
    replacement or close. A scene resource therefore needs no private allocation
    access or staging copy.
    """

    def __init__(self, runtime, program, resources):
        from ..runtime.kernel import VulkanDescriptorSet
        if not isinstance(program, NativeGeometryProgram):
            raise TypeError('program must be NativeGeometryProgram')
        supplied = dict(resources)
        if supplied.keys() != {b.name for b in program.buffers}:
            raise ValueError('Geometry buffers must exactly match program declarations')
        bindings = {}
        for binding, declaration in enumerate(program.buffers):
            bindings[binding] = _native_buffer_resource(runtime, supplied[declaration.name])
        self.runtime, self._program = runtime, program
        self._borrowers = set()
        self._resources = supplied
        self.content_revision = self.binding_revision = 0
        self._descriptors = self._empty = None
        with runtime.lock:
            try:
                self._descriptors = VulkanDescriptorSet(runtime, bindings)
                self._empty = VulkanDescriptorSet(runtime, {})
            except BaseException:
                self.close()
                raise

    @property
    def program(self):
        return self._program

    @property
    def layout(self):
        self.require_open()
        return self._descriptors.layout

    @property
    def empty_material_layout(self):
        self.require_open()
        return self._empty.layout

    @property
    def uses(self):
        from ..transport._custom_resources import resource_uses
        self.require_open()
        return resource_uses(self._descriptors.bindings)

    def bind(self, command, pipeline_layout):
        self.require_open()
        self._descriptors.bind(command, pipeline_layout, set_index=2)

    def require_open(self):
        if self._descriptors is None:
            raise RuntimeError('Native geometry resources are closed')
        self._descriptors.require_open()

    def retain(self, consumer):
        with self.runtime.lock:
            self.require_open()
            self._borrowers.add(consumer)

    def release(self, consumer):
        with self.runtime.lock:
            self._borrowers.discard(consumer)

    def _between_frames(self):
        self.require_open()
        if any(getattr(owner, "gi_frame_prepared", False) for owner in self._borrowers):
            raise RuntimeError("Submit or cancel prepared GI frames before publishing geometry updates")

    def notify_content_changed(self, *, after=(), invalidate_history=True):
        """Publish already-submitted same-queue changes without a CPU wait.

        Update acceleration structures as necessary before the next frame. This
        changes neither buffer handles nor shader declarations. Setting
        invalidate_history=False preserves reusable rendering commands.
        """
        from ..runtime.resources import VulkanCompletion
        with self.runtime.lock:
            self._between_frames()
            if any(not isinstance(c, VulkanCompletion) or c.runtime is not self.runtime for c in after):
                raise ValueError("Geometry producers must use this runtime queue")
            self.content_revision += 1
            if invalidate_history:
                for owner in self._borrowers:
                    reset = getattr(owner, "_invalidate_scene_history", None)
                    if callable(reset):
                        reset()
            return self.content_revision

    def replace_buffers(self, replacements):
        """Replace allocations at an explicit idle boundary; retain the program.

        Omitted names preserve their current allocation. The descriptor layout
        stays compatible; native command caches/history are invalidated. Rebuild
        application graphs whose passes reference the old allocations directly.
        """
        from ..runtime.kernel import VulkanDescriptorSet
        import vulkan as vk
        with self.runtime.lock:
            self._between_frames()
            replacements = dict(replacements)
            if replacements.keys() - self._resources.keys():
                raise ValueError("Unknown geometry buffer name")
            if not replacements:
                return self.binding_revision
            if any(not callable(getattr(owner, "_invalidate_scene_history", None)) for owner in self._borrowers):
                raise RuntimeError("Close external geometry consumers before replacing buffers")
            supplied = {**self._resources, **replacements}
            bindings = {}
            for binding, declaration in enumerate(self.program.buffers):
                bindings[binding] = _native_buffer_resource(self.runtime, supplied[declaration.name])
            new = VulkanDescriptorSet(self.runtime, bindings)
            try:
                vk.vkQueueWaitIdle(self.runtime.queue)
            except BaseException:
                new.close()
                raise
            old, self._descriptors = self._descriptors, new
            self._resources = supplied
            self.binding_revision += 1
            self.content_revision += 1
            for owner in self._borrowers:
                owner._invalidate_scene_history()
            old.close()
            return self.binding_revision

    def close(self):
        with self.runtime.lock:
            if self._borrowers:
                raise RuntimeError('Close native geometry consumers before their resources')
            if self._empty is not None:
                self._empty.close()
                self._empty = None
            if self._descriptors is not None:
                self._descriptors.close()
                self._descriptors = None

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
