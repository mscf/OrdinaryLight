"""Serialized GPU-only timings for adversarial unique-face averaging tables."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile

import numpy as np
import ordinaryshade as osh


@osh.compute(workgroup_size=(8, 8, 1))
def fill_image(image: osh.storage_image('rgba32f', access='write', binding=0)):
    pixel = osh.ivec2(osh.global_invocation_id.xy)
    extent = image.size()
    if pixel.x < extent.x and pixel.y < extent.y:
        image.store(pixel, osh.vec4(osh.f32(pixel.x) / osh.f32(extent.x),
                                   osh.f32(pixel.y) / osh.f32(extent.y), 0.5, 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('smaller', 'direct'), default='smaller')
    parser.add_argument('--slots', type=int)
    parser.add_argument('--packed-faces', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--frames', type=int, default=24)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('frames must be positive')
    import vulkan as vk
    from ordinarylight.runtime import VulkanRuntime, VulkanKernel, compile_compute
    from ordinarylight.pipeline.graph import VulkanGraph, reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    from ordinarylight.pipeline.gi import GiBuffer
    from ordinarylight.wavefront import PRIMARY_HIT_IDENTITY_DTYPE
    from vxl8r_render.backends import sparse_average, gpu_timing
    from vxl8r_render.backends.native_readback import read_hdr
    if args.variant == 'direct':
        from face_direct_anchor_experiment import install
    else:
        from face_table_experiment import install
    from primary_hit_layout_experiment import load_module
    width, height = 512, 256
    count = width * height
    hits = np.zeros(count, PRIMARY_HIT_IDENTITY_DTYPE)
    slots = count
    hits['identity'][:, 1] = np.arange(count, dtype=np.uint32)
    if args.packed_faces:
        slots = (count + 5) // 6
        hits['identity'][:, 1] = np.arange(count, dtype=np.uint32) // 6
        hits['identity'][:, 2] = np.arange(count, dtype=np.uint32) % 6
    if args.slots is not None:
        if args.slots < slots:
            parser.error('--slots must cover every fixture identity')
        if args.packed_faces:
            hits['identity'][:, 1] += args.slots - slots
        slots = args.slots
    selected = False
    with ExitStack() as stack:
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        install(stack, directory, lambda: selected)
        timer_type = gpu_timing.GpuGraphTimer
        if args.split:
            source = Path(gpu_timing.__file__).read_text()
            source = source.replace('                name=node.name', "                name=str(i) + ':' + stage.name")
            timer_type = load_module('diagnostic_occupancy_timer', source, directory).GpuGraphTimer
        runtime = stack.enter_context(VulkanRuntime())
        primary = stack.enter_context(runtime.buffer(hits.nbytes, data=hits))
        hdr = stack.enter_context(runtime.image(width, height, format=vk.VK_FORMAT_R32G32B32A32_SFLOAT))
        compiled = osh.compile(fill_image)
        kernel = stack.enter_context(VulkanKernel(runtime, compile_compute(compiled.source), {0: VulkanResource.image(hdr)}))
        reflected_operation(kernel, compiled.reflection, workgroups=(width // 8, height // 8, 1)).execute(runtime).wait()
        wrapped_hits = GiBuffer(runtime, primary.buffer, primary.byte_size, primary.usage,
                               primary.require_open, record_dtype=PRIMARY_HIT_IDENTITY_DTYPE)
        targets = {}
        for name in ('production', args.variant):
            selected = name == args.variant
            targets[name] = stack.enter_context(sparse_average.SparseFaceAverageTarget(
                runtime, hdr, wrapped_hits, render_extent=(width, height),
                output_extent=(width, height), slots=slots))
        timer = stack.enter_context(timer_type(runtime))
        expected = np.zeros((height, width, 3), dtype=np.float32)
        expected[..., 0] = np.arange(width, dtype=np.float32)[None, :] / width
        expected[..., 1] = np.arange(height, dtype=np.float32)[:, None] / height
        expected[..., 2] = 0.5
        for fraction in (0.25, 0.5, 0.75, 1.0):
            valid = int(count * fraction)
            hits['valid'] = 0
            hits['valid'][:valid] = 1
            primary.upload(hits)
            rows = {name: [] for name in targets}
            for index in range(6 + args.frames):
                names = tuple(targets) if index % 2 == 0 else tuple(reversed(targets))
                for name in names:
                    target = targets[name]
                    operation = target._uncached_operation() if args.split else target.operation()
                    VulkanGraph().add('native_gi', operation).compile().execute(runtime).wait()
                    if index >= 6:
                        rows[name].append(dict(timer.last))
            for name, target in targets.items():
                actual = read_hdr(runtime, target.hdr, (width, height), target.completion)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
                work = np.frombuffer(target.worklist.read(), np.uint32)
                used = int(work[0])
                assert used == 2 * valid - np.count_nonzero((np.arange(valid) // 4) % target.stripes == 0)
                assert len(np.unique(work[4:4 + used])) == used
                print(json.dumps(dict(variant=name, split=args.split, extent=[width, height],
                    valid_fraction=fraction, entries=used, capacity=target.capacity,
                    occupancy=(used - valid if getattr(target, "direct_anchor_count", 0) else used)/target.capacity,
                    slots=slots, direct_anchor_count=getattr(target, "direct_anchor_count", 0), packed_faces=args.packed_faces, samples=args.frames, output_matches=True,
                    median_ms={k:float(np.median([r[k] for r in rows[name]])) for k in rows[name][0]},
                    total_p95_ms=float(np.percentile([r['total'] for r in rows[name]],95)))), flush=True)


if __name__ == '__main__':
    main()
