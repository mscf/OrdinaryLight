"""Explicit ordered Vulkan passes with typed resource barriers.

Callbacks record commands only. They must not submit, close or mutate resources.
All passes run on the runtime's one queue; resources persist until their owner
closes them. No implicit transient allocation or cross-queue ownership transfer.
"""

from dataclasses import dataclass
from operator import index
from typing import Callable

import vulkan as vk


@dataclass(frozen=True)
class VulkanResource:
    owner: object
    kind: str
    handle: object
    size: int = 0

    def __post_init__(self):
        if self.kind not in {"buffer", "image", "acceleration_structure"}:
            raise ValueError("Unknown Vulkan resource kind")
        if self.kind == "buffer" and self.size <= 0:
            raise ValueError("Buffer views need a positive byte size")

    @classmethod
    def buffer(cls, allocation):
        return cls(allocation, "buffer", allocation.buffer, allocation.byte_size)

    @classmethod
    def image(cls, allocation):
        return cls(allocation, "image", allocation.image)


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
        # Each pass declares one combined use per native resource.
        keys = [(use.resource.kind, use.resource.handle) for use in self.uses]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Combine read/write access for duplicate resources in a pass"
            )


class VulkanPassPipeline:
    def __init__(self, passes):
        self.passes = tuple(passes)
        if not all(isinstance(stage, VulkanPass) for stage in self.passes):
            raise TypeError("Expected VulkanPass values")
        if len({stage.name for stage in self.passes}) != len(self.passes):
            raise ValueError("Pass names must be unique")

    def execute(self, runtime, *, after=()):
        with runtime.lock:
            return self._execute(runtime, after=after)

    def _execute(self, runtime, *, after):
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
        states = {}
        layouts = {}

        def record(command):
            for stage in self.passes:
                buffers, images, memory = [], [], []
                src_stages = dst_stages = 0
                for use in stage.uses:
                    resource = use.resource
                    key = (resource.kind, resource.handle)
                    previous = states.get(key)
                    # Conservative entry dependency includes prior submissions,
                    # scene AS builds and host uploads on the runtime queue.
                    src_stage, src_access = previous or (
                        vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
                        | vk.VK_PIPELINE_STAGE_HOST_BIT,
                        vk.VK_ACCESS_MEMORY_WRITE_BIT | vk.VK_ACCESS_HOST_WRITE_BIT,
                    )
                    src_stages |= src_stage
                    dst_stages |= use.stage
                    if resource.kind == "image":
                        old = layouts.get(resource.owner, resource.owner.layout)
                        images.append(
                            vk.VkImageMemoryBarrier(
                                srcAccessMask=0
                                if old == vk.VK_IMAGE_LAYOUT_UNDEFINED
                                else src_access,
                                dstAccessMask=use.access,
                                oldLayout=old,
                                newLayout=use.layout,
                                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                                image=resource.handle,
                                subresourceRange=vk.VkImageSubresourceRange(
                                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                                    levelCount=1,
                                    layerCount=1,
                                ),
                            )
                        )
                        layouts[resource.owner] = use.layout
                    elif resource.kind == "buffer":
                        buffers.append(
                            vk.VkBufferMemoryBarrier(
                                srcAccessMask=src_access,
                                dstAccessMask=use.access,
                                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                                buffer=resource.handle,
                                offset=0,
                                size=resource.size,
                            )
                        )
                    else:
                        memory.append(
                            vk.VkMemoryBarrier(
                                srcAccessMask=src_access, dstAccessMask=use.access
                            )
                        )
                    states[key] = (use.stage, use.access)
                if stage.uses:
                    vk.vkCmdPipelineBarrier(
                        command,
                        src_stages,
                        dst_stages,
                        0,
                        len(memory),
                        memory or None,
                        len(buffers),
                        buffers or None,
                        len(images),
                        images or None,
                    )
                stage.record(command)
                if stage.workgroups is not None:
                    # Recorder binds descriptors/pipeline; extent belongs to app.
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

        completion = runtime.submit(record, resources=owners, after=after)
        # Commit layout state only after successful recording and submission.
        for owner, layout in layouts.items():
            owner.layout = layout
        return completion
