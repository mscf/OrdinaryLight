"""Pre-migration direct primary recorder for controlled parity/timing comparisons."""

import vulkan as vk


def direct_primary(executor, command, pipeline, slot, constants, groups):
    vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vk.vkCmdBindDescriptorSets(
        command,
        vk.VK_PIPELINE_BIND_POINT_COMPUTE,
        executor.primary_pipeline_layout,
        0,
        1,
        [executor.primary_sets[slot]],
        0,
        None,
    )
    executor.core._bind_material_resources(command, executor.primary_pipeline_layout)
    vk.vkCmdPushConstants(
        command,
        executor.primary_pipeline_layout,
        vk.VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        len(constants),
        vk.ffi.from_buffer(constants),
    )
    vk.vkCmdDispatch(command, *groups)
