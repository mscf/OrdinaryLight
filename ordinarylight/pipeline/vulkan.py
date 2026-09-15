"""Explicit ordered Vulkan passes with typed resource barriers.

Callbacks record commands only. They must not submit, close or mutate resources.
All passes run on the runtime's one queue; resources persist until their owner
closes them. No implicit transient allocation or cross-queue ownership transfer.
"""

from dataclasses import dataclass
from functools import lru_cache
from operator import index
from typing import Callable

import vulkan as vk


@dataclass(frozen=True)
class VulkanResource:
    owner: object
    kind: str
    handle: object
    size: int = 0
    descriptor: str | None = None
    offset: int = 0

    def __post_init__(self):
        object.__setattr__(self, "size", index(self.size))
        object.__setattr__(self, "offset", index(self.offset))
        if self.kind not in {"buffer", "image", "acceleration_structure", "sampler"}:
            raise ValueError("Unknown Vulkan resource kind")
        if self.offset < 0 or (self.kind != "buffer" and self.offset):
            raise ValueError("Only buffer resources support nonnegative offsets")
        if self.kind == "buffer" and self.size <= 0:
            raise ValueError("Buffer views need a positive byte size")

    def byte_range(self, offset, size):
        """Borrow a bounded descriptor/hazard view relative to this buffer view."""
        from dataclasses import replace

        offset, size = index(offset), index(size)
        if (
            self.kind != "buffer"
            or offset < 0
            or size <= 0
            or offset + size > self.size
        ):
            raise ValueError("Buffer range is outside its parent view")
        return replace(self, offset=self.offset + offset, size=size)

    def version(self, number=0):
        from .graph import ResourceVersion

        return ResourceVersion(self, number)

    @classmethod
    def buffer(cls, allocation):
        return cls(allocation, "buffer", allocation.buffer, allocation.byte_size)

    @classmethod
    def image(cls, allocation):
        return cls(allocation, "image", allocation.image)

    @classmethod
    def uniform_buffer(cls, allocation):
        return cls(
            allocation,
            "buffer",
            allocation.buffer,
            allocation.byte_size,
            "uniform_buffer",
        )

    @classmethod
    def sampled_image(cls, allocation):
        return cls(
            allocation, "image", allocation.image, descriptor="sampled_texture_2d"
        )

    @classmethod
    def sampler(cls, allocation):
        return cls(allocation, "sampler", allocation.handle, descriptor="sampler")


@dataclass(frozen=True)
class VulkanResourceUse:
    resource: VulkanResource
    stage: int
    access: int
    layout: int | None = None

    def __post_init__(self):
        if not isinstance(self.resource, VulkanResource):
            raise TypeError("Expected a VulkanResource")
        if not self.stage:
            raise ValueError("A pipeline stage mask is required")
        if self.resource.kind == "image":
            if self.layout is None or self.layout in (
                vk.VK_IMAGE_LAYOUT_UNDEFINED,
                vk.VK_IMAGE_LAYOUT_PREINITIALIZED,
            ):
                raise ValueError("Image use requires a usable destination layout")
        elif self.layout is not None:
            raise ValueError("Only images have layouts")


@dataclass(frozen=True)
class VulkanPass:
    name: str
    uses: tuple[VulkanResourceUse, ...]
    record: Callable
    workgroups: tuple[int, int, int] | None = None

    def __post_init__(self):
        if not self.name or not callable(self.record):
            raise ValueError("A pass needs a name and command recorder")
        object.__setattr__(self, "uses", tuple(self.uses))
        if not all(isinstance(use, VulkanResourceUse) for use in self.uses):
            raise TypeError("Pass uses must be VulkanResourceUse values")
        if self.workgroups is not None:
            groups = tuple(index(n) for n in self.workgroups)
            if len(groups) != 3 or min(groups) <= 0:
                raise ValueError("workgroups must contain three positive integers")
            object.__setattr__(self, "workgroups", groups)
        # Distinct byte ranges may share an allocation. Overlapping declarations
        # in a single pass must still be combined by the caller.
        ranges = {}
        for use in self.uses:
            resource = use.resource
            key = (resource.kind, resource.handle)
            previous = ranges.get(key)
            if previous is None:
                ranges[key] = [(resource.offset, resource.offset + resource.size)]
            elif resource.kind != "buffer":
                raise ValueError("Combine read/write access for duplicate resources in a pass")
            else:
                previous.append((resource.offset, resource.offset + resource.size))
        for intervals in ranges.values():
            if len(intervals) < 2:
                continue
            intervals.sort()
            end = intervals[0][1]
            for start, next_end in intervals[1:]:
                if start < end:
                    raise ValueError("Combine read/write access for duplicate resources in a pass")
                end = next_end


@lru_cache(maxsize=128)
def _barrier_plan(signature, entry_layouts):
    """Immutable Vulkan structures, keyed by exact uses and current layouts.

    Contains raw handles only; resource ownership and frame callbacks are never
    cached. Vulkan consumes these structures as const input while recording.
    """
    states = {}
    layouts = dict(entry_layouts)
    barriers = []
    for stage_uses in signature:
        buffers, images, memory = [], [], []
        src_stages = dst_stages = 0
        for kind, handle, offset, size, dst_stage, dst_access, layout in stage_uses:
            key = (kind, handle)
            previous = states.get(key)
            # Conservative entry dependency includes prior submissions,
            # scene AS builds and host uploads on the runtime queue.
            src_stage, src_access = previous or (
                vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
                | vk.VK_PIPELINE_STAGE_HOST_BIT,
                vk.VK_ACCESS_MEMORY_READ_BIT
                | vk.VK_ACCESS_MEMORY_WRITE_BIT
                | vk.VK_ACCESS_HOST_WRITE_BIT,
            )
            src_stages |= src_stage
            dst_stages |= dst_stage
            if kind == "image":
                old = layouts[key]
                images.append(
                    vk.VkImageMemoryBarrier(
                        srcAccessMask=0
                        if old == vk.VK_IMAGE_LAYOUT_UNDEFINED
                        else src_access,
                        dstAccessMask=dst_access,
                        oldLayout=old,
                        newLayout=layout,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=handle,
                        subresourceRange=vk.VkImageSubresourceRange(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            levelCount=1,
                            layerCount=1,
                        ),
                    )
                )
                layouts[key] = layout
            elif kind == "buffer":
                buffers.append(
                    vk.VkBufferMemoryBarrier(
                        srcAccessMask=src_access,
                        dstAccessMask=dst_access,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        buffer=handle,
                        offset=offset,
                        size=size,
                    )
                )
            else:
                memory.append(
                    vk.VkMemoryBarrier(
                        srcAccessMask=src_access, dstAccessMask=dst_access
                    )
                )
            # Keep all prior stage/access types: an intervening use of a
            # disjoint byte range must not hide an earlier overlapping use.
            prior = states.get(key, (0, 0))
            states[key] = (prior[0] | dst_stage, prior[1] | dst_access)
        barriers.append((src_stages, dst_stages, tuple(memory), tuple(buffers), tuple(images)))
    return tuple(barriers), tuple(layouts.items())


class VulkanPassPipeline:
    def __init__(self, passes):
        self.passes = tuple(passes)
        if not all(isinstance(stage, VulkanPass) for stage in self.passes):
            raise TypeError("Expected VulkanPass values")
        if len({stage.name for stage in self.passes}) != len(self.passes):
            raise ValueError("Pass names must be unique")

    def execute(self, runtime, *, after=(), wait_semaphores=(), signal_semaphores=()):
        with runtime.lock:
            return self._execute(
                runtime,
                after=after,
                wait_semaphores=wait_semaphores,
                signal_semaphores=signal_semaphores,
            )

    def _prepare_recording(self, runtime):
        """Prepare command recording and deferred layout publication."""
        owners = tuple(
            dict.fromkeys(
                use.resource.owner for stage in self.passes for use in stage.uses
            )
        )
        for owner in owners:
            if owner.runtime is not runtime:
                raise ValueError(
                    "All pass resources must belong to the supplied runtime"
                )
            owner.require_open()
        layouts = {}
        image_owners = {}
        for stage in self.passes:
            for use in stage.uses:
                resource = use.resource
                if resource.kind == "image":
                    key = (resource.kind, resource.handle)
                    aliases = image_owners.setdefault(key, set())
                    if aliases and any(
                        owner.layout != resource.owner.layout for owner in aliases
                    ):
                        raise ValueError(
                            "Aliased image owners must agree on the imported layout"
                        )
                    aliases.add(resource.owner)

        signature = tuple(tuple((use.resource.kind, use.resource.handle,
                                 use.resource.offset, use.resource.size,
                                 use.stage, use.access, use.layout)
                                for use in stage.uses) for stage in self.passes)

        def record(command):
            # Read layouts at recording time, not preparation time: an external
            # caller may have submitted other work after preparing this graph.
            entry_layouts = tuple((key, next(iter(aliases)).layout)
                                  for key, aliases in image_owners.items())
            barriers, final_layouts = _barrier_plan(signature, entry_layouts)
            layouts.update(final_layouts)
            for stage, (src, dst, memory, buffers, images) in zip(self.passes, barriers):
                if stage.uses:
                    vk.vkCmdPipelineBarrier(command, src, dst, 0,
                        len(memory), memory or None, len(buffers), buffers or None,
                        len(images), images or None)
                stage.record(command)
                if stage.workgroups is not None:
                    vk.vkCmdDispatch(command, *stage.workgroups)
            # Make shader writes visible to subsequent queue consumers and host
            # readback after fence completion, including existing GI paths.
            if self.passes:
                vk.vkCmdPipelineBarrier(
                    command,
                    vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
                    | vk.VK_PIPELINE_STAGE_HOST_BIT,
                    0,
                    1,
                    [
                        vk.VkMemoryBarrier(
                            srcAccessMask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                            dstAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT
                            | vk.VK_ACCESS_HOST_READ_BIT,
                        )
                    ],
                    0,
                    None,
                    0,
                    None,
                )

        def commit():
            for key, layout in layouts.items():
                for owner in image_owners[key]:
                    owner.layout = layout

        return record, owners, commit

    def _execute(self, runtime, *, after, wait_semaphores, signal_semaphores):
        record, owners, commit = self._prepare_recording(runtime)
        completion = runtime.submit(
            record,
            resources=owners,
            after=after,
            wait_semaphores=wait_semaphores,
            signal_semaphores=signal_semaphores,
        )
        commit()
        return completion


def __getattr__(name):
    if name in {
        "VulkanGraph",
        "VulkanOperation",
        "ResourceVersion",
        "CompiledVulkanGraph",
        "reflected_operation",
    }:
        from . import graph

        return getattr(graph, name)
    raise AttributeError(name)
