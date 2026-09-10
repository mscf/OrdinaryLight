"""Compare fused-primary cost at a fixed pose; save timings and HDR references.

The depth-one case is diagnostic, not an equivalent-quality rendering mode:
it skips primary lighting, BSDF continuation and later bounces together.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _catalog, _gi_config

CASES = (
    ('baseline', {}),
    ('repeat', {}),
    ('hit_material_floor', {'max_bounces': 1}),
    ('one_candidate', {'wavefront_restir_candidates': 1}),
    ('one_reservoir', {'wavefront_restir_reservoirs': 1}),
    ('native_textures', {'wavefront_native_textures': True}),
    ('primary_surface_only', {'wavefront_primary_scene_specialization': True}),
    ('primary_surface_repeat', {'wavefront_primary_scene_specialization': True}),
    ('baseline_end', {}),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width', type=int, default=2052)
    p.add_argument('--height', type=int, default=1764)
    p.add_argument('--warmup', type=int, default=32)
    p.add_argument('--frames', type=int, default=64)
    p.add_argument('--cases', nargs='+', choices=[name for name, _ in CASES])
    p.add_argument('--primary-specialization', action=argparse.BooleanOptionalAction,
                   default=True, help='Use the current production baseline (default: enabled)')
    p.add_argument('--pipeline-statistics', action='store_true',
                   help='Capture driver executable statistics; use a separate run from timing')
    p.add_argument('--reference', type=Path, help='HDR .npy from a previous identical experiment')
    p.add_argument('--showcase', default='optical-screen-rough-reflection')
    p.add_argument('--gi-mode', choices=('fast', 'full'), default='fast')
    p.add_argument('--primary-variant', choices=('current', 'surface-only', 'opaque'),
                   default='current', help='Ablate the newly connected primary specializations')
    p.add_argument('--primary-workgroup-rows', type=int, choices=(4, 8), default=8,
                   help='Diagnostic primary workgroup shape: 8x4 or production 8x8')
    p.add_argument('--primary-brdf-cache', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--output', type=Path, default=Path('/tmp/ordinarylight-primary-costs'))
    args = p.parse_args()
    if min(args.width, args.height, args.frames) < 1 or args.warmup < 2:
        p.error('positive extent/frame count and at least two warmup frames required')
    args.output.mkdir(parents=True, exist_ok=True)
    item = next(s for s in _catalog() if s.id == args.showcase)
    scene = item.build()
    camera = item.camera.camera(scene, angle=-0.45)
    config = replace(_gi_config(item, low_bounce=args.gi_mode == 'fast', capture=True),
                     wavefront_primary_scene_specialization=args.primary_specialization,
                     wavefront_pipeline_statistics=args.pipeline_statistics)
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError('GLFW initialization failed')
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    results = {'showcase': item.id, 'extent': [args.width, args.height],
               'spp': config.samples_per_pixel, 'bounces': config.max_bounces,
               'textures': len(scene.textures), 'warmup': args.warmup,
               'frames': args.frames,
               'primary_specialization': args.primary_specialization,
               'pipeline_statistics_enabled': args.pipeline_statistics,
               'primary_variant': args.primary_variant,
               'primary_workgroup_rows': args.primary_workgroup_rows,
               'primary_brdf_cache': args.primary_brdf_cache,
               'reference': str(args.reference) if args.reference else None,
               'cases': {}}
    reference = np.load(args.reference) if args.reference else None
    from ordinarylight.shaders import compiler
    from ordinarylight.targets.vulkan import primary_graph
    compile_shader = compiler.compile_wavefront_material_shader
    compile_source = compiler._compile_source
    def compile_variant(shader_name, *positional, **kwargs):
        if shader_name == 'wavefront_primary.comp':
            if args.primary_variant == 'surface-only':
                kwargs['opaque_primary'] = False
            if args.primary_variant != 'current':
                kwargs['production_restir'] = False
            if args.primary_workgroup_rows == 4 or not args.primary_brdf_cache:
                def compile_diagnostic(source, compiler_path):
                    defines = ''
                    if args.primary_workgroup_rows == 4:
                        defines += '#define WAVE_LOCAL_SIZE_Y 4\n'
                    if not args.primary_brdf_cache:
                        defines += '#define WAVE_PREPARED_PRIMARY_PBR 0\n'
                    return compile_source(source.replace(
                        '#version 460\n', '#version 460\n' + defines, 1),
                        compiler_path)
                with patch.object(compiler, '_compile_source', compile_diagnostic):
                    return compile_shader(shader_name, *positional, **kwargs)
        return compile_shader(shader_name, *positional, **kwargs)
    record_primary = primary_graph.record_primary
    def record_variant(executor, command, pipeline, slot, constants, groups):
        if args.primary_workgroup_rows == 4:
            # Twice as many half-height groups cover the same tile; the shader
            # retains its tile bounds check for padding at the lower edge.
            groups = (groups[0], groups[1] * 2, groups[2])
        return record_primary(executor, command, pipeline, slot, constants, groups)
    compiler_patch = patch.object(compiler, 'compile_wavefront_material_shader', compile_variant)
    recorder_patch = patch.object(primary_graph, 'record_primary', record_variant)
    compiler_patch.start()
    recorder_patch.start()
    try:
        for name, changes in CASES:
            if args.cases and name not in args.cases:
                continue
            window = glfw.create_window(args.width, args.height, name, None, None)
            if not window:
                raise RuntimeError('GLFW window creation failed')
            print('START', name, flush=True)
            try:
                with ol.VulkanGlfwPresenter(window, config=replace(config, **changes)) as renderer:
                    rows = []
                    for i in range(args.warmup + args.frames):
                        glfw.poll_events()
                        renderer.present_wavefront(scene, camera, args.width, args.height)
                        if i >= args.warmup:
                            rows.append(dict(renderer.last_timings))
                    hdr = renderer.capture_wavefront_hdr().copy()
                    native = renderer._core.native_textures_enabled
                    statistics = (dict(renderer._core.wavefront_executor.pipeline_statistics)
                                  if args.pipeline_statistics else {})
                if not np.isfinite(hdr).all():
                    raise RuntimeError(f'{name}: non-finite HDR')
                np.save(args.output / (name + '.npy'), hdr)
                if reference is None:
                    reference = hdr
                labels = set().union(*(r['wavefront_stage_ms'] for r in rows))
                delta = hdr[..., :3].astype(np.float64) - reference[..., :3]
                result = {
                    'changes': changes, 'native_textures_enabled': native,
                    'pipeline_statistics': statistics,
                    'gpu_ms': float(np.median([r['gpu_frame_ms'] for r in rows])),
                    'gpu_p90_ms': float(np.percentile([r['gpu_frame_ms'] for r in rows], 90)),
                    'stages_ms': {k: float(np.median([r['wavefront_stage_ms'].get(k, 0)
                                                    for r in rows])) for k in sorted(labels)},
                    'hdr_mean': float(hdr[..., :3].mean()),
                    'hdr_rmse_vs_baseline': float(np.sqrt(np.mean(delta * delta))),
                    'hdr_max_difference': float(np.max(np.abs(delta))),
                }
                results['cases'][name] = result
                (args.output / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
                print(name, json.dumps(result), flush=True)
            finally:
                glfw.destroy_window(window)
    finally:
        compiler_patch.stop()
        recorder_patch.stop()
        glfw.terminate()


if __name__ == '__main__':
    main()
