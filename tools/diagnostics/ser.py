"""Execute a minimal Vulkan ray-generation shader using NVIDIA SER."""

import argparse
from importlib.resources import files

import numpy as np
import vulkan as vk

from ordinarylight.vulkan import RendererConfig
from ordinarylight.vulkan_rt import (
    BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    PIPELINE_BIND_POINT_RAY_TRACING_KHR,
    VulkanRayQueryCore,
)


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reorder", action="store_true")
    args = parser.parse_args()
    core = VulkanRayQueryCore(config=RendererConfig(wavefront_ser=True))
    resources = []
    try:
        if not core.ray_pipeline_supported:
            raise RuntimeError("VK_KHR_ray_tracing_pipeline is unavailable")
        if not core.ser_reordering_supported:
            raise RuntimeError("hardware SER reordering is unavailable")

        count = 4096
        output = core._create_buffer(
            count * 4,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            device_address=True,
        )
        output_address = core._buffer_address(output)
        pipeline_layout = vk.vkCreatePipelineLayout(
            core.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pushConstantRangeCount=1,
                pPushConstantRanges=[vk.VkPushConstantRange(
                    stageFlags=vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                    offset=0,
                    size=8,
                )],
            ),
            None,
        )
        resources.append((vk.vkDestroyPipelineLayout, pipeline_layout))

        shader_name = (
            "ser_probe.rgen.spv" if args.no_reorder
            else "ser_probe_ser.rgen.spv"
        )
        shader_bytes = files("ordinarylight").joinpath(
            f"shaders/{shader_name}"
        ).read_bytes()
        module = vk.vkCreateShaderModule(
            core.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(shader_bytes), pCode=shader_bytes,
            ),
            None,
        )
        resources.append((vk.vkDestroyShaderModule, module))
        miss_bytes = files("ordinarylight").joinpath(
            "shaders/ser_probe.rmiss.spv"
        ).read_bytes()
        miss_module = vk.vkCreateShaderModule(
            core.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(miss_bytes), pCode=miss_bytes,
            ),
            None,
        )
        resources.append((vk.vkDestroyShaderModule, miss_module))
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR,
            module=module,
            pName="main",
        )
        miss_stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_MISS_BIT_KHR,
            module=miss_module,
            pName="main",
        )
        group = vk.VkRayTracingShaderGroupCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            type=vk.VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            generalShader=0,
            closestHitShader=vk.VK_SHADER_UNUSED_KHR,
            anyHitShader=vk.VK_SHADER_UNUSED_KHR,
            intersectionShader=vk.VK_SHADER_UNUSED_KHR,
        )
        miss_group = vk.VkRayTracingShaderGroupCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            type=vk.VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            generalShader=1,
            closestHitShader=vk.VK_SHADER_UNUSED_KHR,
            anyHitShader=vk.VK_SHADER_UNUSED_KHR,
            intersectionShader=vk.VK_SHADER_UNUSED_KHR,
        )
        create_info = vk.VkRayTracingPipelineCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            stageCount=2,
            pStages=[stage, miss_stage],
            groupCount=2,
            pGroups=[group, miss_group],
            maxPipelineRayRecursionDepth=1,
            layout=pipeline_layout,
        )
        pipelines = core.create_ray_tracing_pipelines(
            core.device, vk.ffi.NULL, vk.ffi.NULL, 1,
            [create_info], None,
        )
        pipeline = pipelines[0]
        resources.append((vk.vkDestroyPipeline, pipeline))

        handle_size = core.ray_tracing_shader_group_handle_size
        stride = _align(
            handle_size, core.ray_tracing_shader_group_handle_alignment
        )
        handle_data = vk.ffi.new("uint8_t[]", handle_size * 2)
        core.get_ray_tracing_shader_group_handles(
            core.device, pipeline, 0, 2, handle_size * 2,
            handle_data,
        )
        handle_bytes = bytes(vk.ffi.buffer(handle_data, handle_size * 2))
        region_offset = _align(
            stride, core.ray_tracing_shader_group_base_alignment
        )
        sbt_payload = np.zeros(region_offset + stride, dtype=np.uint8)
        sbt_payload[:handle_size] = np.frombuffer(
            handle_bytes[:handle_size], dtype=np.uint8
        )
        sbt_payload[region_offset:region_offset + handle_size] = np.frombuffer(
            handle_bytes[handle_size:], dtype=np.uint8
        )
        sbt = core._create_uploaded_device_buffer(
            sbt_payload,
            vk.VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
            | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            device_address=True,
        )
        sbt_address = core._buffer_address(sbt)
        if sbt_address % core.ray_tracing_shader_group_base_alignment:
            raise RuntimeError(
                f"SBT address {sbt_address:#x} is not base-aligned to "
                f"{core.ray_tracing_shader_group_base_alignment}"
            )
        if not any(handle_bytes):
            raise RuntimeError("ray-generation shader-group handle is all zero")
        raygen = vk.VkStridedDeviceAddressRegionKHR(
            deviceAddress=sbt_address, stride=stride, size=stride
        )
        miss = vk.VkStridedDeviceAddressRegionKHR(
            deviceAddress=sbt_address + region_offset,
            stride=stride,
            size=stride,
        )
        empty = vk.VkStridedDeviceAddressRegionKHR()
        output_address_data = np.array([output_address], dtype=np.uint64)

        def record(command):
            vk.vkCmdBindPipeline(
                command, PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline
            )
            vk.vkCmdPushConstants(
                command, pipeline_layout, vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                0, 8, vk.ffi.from_buffer(output_address_data),
            )
            core.cmd_trace_rays(
                command,
                vk.ffi.addressof(raygen),
                vk.ffi.addressof(miss),
                vk.ffi.addressof(empty),
                vk.ffi.addressof(empty),
                count, 1, 1,
            )
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                vk.VK_PIPELINE_STAGE_HOST_BIT,
                0,
                0,
                None,
                1,
                [vk.VkBufferMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    buffer=output.buffer,
                    offset=0,
                    size=output.size,
                )],
                0,
                None,
            )

        core._single_use(record)
        mapped = vk.vkMapMemory(core.device, output.memory, 0, output.size, 0)
        values = np.frombuffer(bytes(mapped[:output.size]), dtype=np.uint32)
        vk.vkUnmapMemory(core.device, output.memory)
        expected = np.arange(count, dtype=np.uint32) ^ np.uint32(0x51E4A9BD)
        if not np.array_equal(values, expected):
            mismatch = np.flatnonzero(values != expected)
            first = int(mismatch[0]) if mismatch.size else -1
            raise RuntimeError(
                "SER probe output validation failed: "
                f"{mismatch.size}/{count} mismatches; first index={first}, "
                f"actual={int(values[first]) if first >= 0 else 'n/a'}, "
                f"expected={int(expected[first]) if first >= 0 else 'n/a'}"
            )
        print(
            f"PASS: {core.device_name} executed {count} raygen invocations "
            f"({'baseline' if args.no_reorder else 'SER'})"
        )
    finally:
        if core.device:
            vk.vkDeviceWaitIdle(core.device)
            for destroy, resource in reversed(resources):
                destroy(core.device, resource, None)
        core.close()


if __name__ == "__main__":
    main()
