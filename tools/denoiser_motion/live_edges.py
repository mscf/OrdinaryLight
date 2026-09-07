"""Capture live denoiser guides and independent references for edge diagnosis."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import vulkan as vk

import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.showcases.rooms import build_object_motion_room
from tests.gates.relax_motion_quality import _config, _trajectory


def read_guides(core):
    if not core.config.denoiser_signal_capture:
        raise ValueError("live guide capture requires denoiser_signal_capture")
    frame = core.window_frames[(core.window_frame_index - 1) % 2]
    width, height = frame["wavefront_render_extent"]
    specs = {
        name: (np.float16, 4)
        for name in (
            "diffuse",
            "specular",
            "normal_roughness",
            "motion",
            "temporal_diffuse",
            "temporal_specular",
        )
    }
    specs.update(
        {
            name: (np.float32, 1)
            for name in ("view_z", "diffuse_history", "specular_history")
        }
    )
    specs["identity"] = (np.uint32, 1)
    images = {name: frame[f"wavefront_relax_{name}_image"] for name in specs}
    specs["material"] = (np.uint32, 1)
    images["material"] = frame["wavefront_material_image"]
    vk.vkWaitForFences(core.device, 1, [frame["fence"]], vk.VK_TRUE, (1 << 64) - 1)
    result = {}
    for name, (dtype, channels) in specs.items():
        image = images[name]
        size = width * height * channels * np.dtype(dtype).itemsize
        buf = core._create_buffer(
            size,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

        def record(command):
            barrier = core._image_barrier(
                image,
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_ACCESS_TRANSFER_READ_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )
            region = vk.VkBufferImageCopy(
                bufferOffset=0,
                bufferRowLength=0,
                bufferImageHeight=0,
                imageSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
                imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                imageExtent=vk.VkExtent3D(width=width, height=height, depth=1),
            )
            vk.vkCmdCopyImageToBuffer(
                command,
                image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                buf.buffer,
                1,
                [region],
            )
            barrier = core._image_barrier(
                image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_ACCESS_TRANSFER_READ_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )

        try:
            core._single_use(record)
            mapped = vk.vkMapMemory(core.device, buf.memory, 0, size, 0)
            result[name] = (
                np.frombuffer(mapped, dtype=dtype)
                .copy()
                .reshape(height, width, channels)
            )
            vk.vkUnmapMemory(core.device, buf.memory)
        finally:
            vk.vkDestroyBuffer(core.device, buf.buffer, None)
            vk.vkFreeMemory(core.device, buf.memory, None)
            core._buffers.remove(buf)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-batches", type=int, default=8)
    parser.add_argument("--evaluated-lobes", action="store_true")
    parser.add_argument("--scene", choices=("object-motion", "optics"), default="object-motion")
    args = parser.parse_args()
    if args.reference_batches < 2 or args.reference_batches % 2:
        parser.error("reference batches must be positive, even and at least two")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = SimpleNamespace(
        width=320,
        height=180,
        frames=7,
        reference_samples=16,
        bounces=8,
        atrous_iterations=3,
        camera_arc=0.2,
        history_floor=3,
    )
    trajectory = _trajectory(settings, "camera")
    scene_factory = build_object_motion_room
    if args.scene == "optics":
        from tools.denoiser_motion.run import fixture, camera as optics_camera
        def scene_factory():
            return fixture()[0]
        trajectory = [(0.0, optics_camera(x)) for x in np.linspace(-0.4, 0.4, 7)]
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(
        settings.width, settings.height, "Live edge capture", None, None
    )
    try:
        for floor in (1, 3):
            settings.history_floor = floor
            config = _config(settings, reference=False)
            from dataclasses import replace

            config = replace(
                config, denoiser_signal_capture=True,
                denoiser_sampled_indirect=args.evaluated_lobes,
            )
            scene = scene_factory()
            frames = []
            with ol.VulkanGlfwPresenter(window, config=config) as presenter:
                for index, (_, camera) in enumerate(trajectory):
                    presenter.present_wavefront(
                        scene, camera, settings.width, settings.height
                    )
                    frames.append(presenter.capture_wavefront_hdr())
                    np.savez_compressed(
                        args.output / f"floor{floor}-guides-{index:03d}.npz",
                        **read_guides(presenter._core),
                    )
            np.save(args.output / f"floor{floor}.npy", frames)
            print(f"Captured live floor {floor}", flush=True)
        scene = scene_factory()
        refs, split = [], []
        settings.reference_samples = 64
        with ol.VulkanGlfwPresenter(
            window, config=_config(settings, reference=True)
        ) as presenter:
            # Different seeds from the candidate; same live tracing/resolve path.
            for index, (_, camera) in enumerate(trajectory):
                batches = []
                for j in range(args.reference_batches):
                    presenter._core.wavefront_frame_sequence = (
                        1000 + index * args.reference_batches + j
                    )
                    presenter.present_wavefront(
                        scene, camera, settings.width, settings.height
                    )
                    batches.append(presenter.capture_wavefront_hdr()[..., :3])
                refs.append(np.mean(batches, axis=0))
                split.append(
                    [np.mean(batches[::2], axis=0), np.mean(batches[1::2], axis=0)]
                )
                print(f"Independent live reference {index + 1}/7", flush=True)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    np.save(args.output / "reference.npy", refs)
    np.save(args.output / "reference-split.npy", split)
    (args.output / "configuration.json").write_text(
        json.dumps(
            {
                "width": 320,
                "height": 180,
                "frames": 7,
                "reference_samples": 64 * args.reference_batches,
                "seed_offset": 1000,
                "scene": args.scene,
                "evaluated_lobes": args.evaluated_lobes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
