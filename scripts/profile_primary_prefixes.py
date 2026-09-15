"""Headless vxl8r primary-prefix diagnostics; never a production render mode.

Requires the vxl8r checkout on PYTHONPATH. Timings are cumulative workload
measurements, not additive timers inside the fused shader. Prefix output HDR is
not a valid lighting result. No packaged shader or renderer file is modified.
"""
import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from ordinarylight.shaders import compiler, fused_primary_programs


def prefix_module(stage, directory):
    source = Path(fused_primary_programs.__file__).read_text()
    start = source.index('@osh.function\ndef processPrimaryPixel(')
    end = source.index('\n@osh.function\ndef main()', start)
    body = source[start:end]
    anchors = {
        'trace': '    if osh.specialization("!WAVE_SURFACE_ONLY"):',
        'synthetic': '    if osh.specialization("!WAVE_SURFACE_ONLY"):',
        'trace-no-clear': '    if osh.specialization("!WAVE_SURFACE_ONLY"):',
        'trace-no-export': '    if osh.specialization("!WAVE_SURFACE_ONLY"):',
        'material': '    path.radiance.rgb = path.radiance.rgb + ordinarylight_primary_emission',
        'lighting': '        transportPbrPrepared = False',
    }
    if not stage.startswith('full-'):
        body = body[:body.index(anchors[stage])]
    if stage == 'synthetic':
        # Deliberately fake a custom hit, retaining camera-dependent export writes.
        # This measures a no-traversal workload, not valid geometry or lighting.
        body = body.replace(
            'nativeTraceSurface(ray_origin, 0.001, incoming, 1e+30, False, nativeSurfaceMask())',
            'NativeIntersection(osh.vec4(ray_origin + incoming, 1.0), '
            'osh.vec4(0.0, 1.0, 0.0, 0.0), osh.vec4(0.0, 1.0, 0.0, 0.0), '
            'osh.uvec4(0), osh.uvec4(0, 0, 0, 2), osh.vec4(0.0), '
            'osh.vec4(ray_origin + incoming, 1.0))')
    if stage in ('trace-no-clear', 'full-no-clear'):
        body = body.replace('    if indirect_capture_pixel:\n        ordinarylight_clear_secondary_path(path_index)\n', '')
    if stage == 'trace-no-export':
        begin = body.index("    if osh.specialization('WAVE_PRIMARY_HITS'):")
        stop = body.index("    if osh.specialization('WAVE_CUSTOM_GEOMETRY'):", begin)
        body = body[:begin] + body[stop:]
        body += '    path.radiance.rgb = intersection.geometric_normal.xyz + osh.vec3(distance)\n'
    if stage == 'full-single-export':
        begin = body.index('        primary_hits[hit_output_index] = PrimaryHitOutput(')
        stop = body.index('        if surface_hit:', begin)
        initializer = body[begin:stop]
        body = body[:begin] + body[stop:]
        begin = body.index('            primary_hits[hit_output_index].position_distance =')
        stop = body.index("    if osh.specialization('WAVE_CUSTOM_GEOMETRY'):", begin)
        body = body[:begin] + '''            primary_hits[hit_output_index] = PrimaryHitOutput(
                osh.vec4(ray_origin + distance * incoming, distance),
                osh.vec4(export_normal, 0), osh.vec4(export_normal, 0),
                intersection.identity, osh.vec4(ray_origin, jitter.x),
                osh.vec4(incoming, jitter.y))
        else:
''' + ''.join('    ' + line + '\n' for line in initializer.splitlines()) + body[stop:]
    if stage in ('trace', 'synthetic', 'trace-no-clear', 'trace-no-export'):
        body += '''    if push.gbuffer_enabled != osh.u32(0):
        position_image.store(osh.ivec2(pixel), ordinarylight_primary_invalid_position())
        normal_image.store(osh.ivec2(pixel), ordinarylight_primary_packed_payload(osh.u32(0)))
        material_image.store(osh.ivec2(pixel), ordinarylight_primary_invalid_material())
'''
    if stage == 'material':
        # Consume evaluated material data so it cannot be optimized away.
        body += '    path.radiance.rgb = material.base_roughness.rgb + material.emission_metallic.rgb\n'
    if not stage.startswith('full-'):
        body += '''    path.metadata.w = ordinarylight_primary_deactivate(path.metadata.w)
    ordinarylight_store_path(path_index, path)
    return
'''
    path = directory / ('primary_' + stage + '.py')
    path.write_text(source[:start] + body + source[end:])
    spec = importlib.util.spec_from_file_location('ordinarylight.shaders.diagnostic_' + stage, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emitter-object',action='store_true')
    parser.add_argument('--uneven-floor',action='store_true')
    parser.add_argument('--reject-dark-environment',action='store_true')
    parser.add_argument('--fused-resolve',action='store_true')
    parser.add_argument('--candidate-direct',action='store_true')
    parser.add_argument('--candidate-history',choices=('reset_on_content','local'))
    parser.add_argument('--candidate-iterations',type=int,choices=range(1,6))
    parser.add_argument('--candidate-tile-capacity',type=int,choices=(131072,262144,524288))
    parser.add_argument('--transport-counters',action='store_true')
    parser.add_argument('--capture-dir',type=Path,help='Save comparison frames; diagnostic readbacks invalidate throughput conclusions')
    parser.add_argument('--compact-continuations',action='store_true')
    parser.add_argument('--stage', choices=('trace', 'material', 'lighting', 'synthetic', 'trace-no-clear', 'trace-no-export', 'full-no-clear', 'full-single-export', 'full-hit-planes', 'full-compact-hits', 'full-control', 'full-public-identity', 'full-secondary-clear', 'full-secondary-fields', 'full-prepare-fields', 'full-prepare-single', 'full-reset-fusion', 'full-paired-spatial', 'full-as-2048', 'full-primary-capture', 'full-primary-clear-store', 'full-primary-queue', 'full-primary-shared-pbr', 'full-face-anchor', 'full-face-stripes', 'full-face-combined', 'full-face-table', 'full-face-direct', 'full-face-worklist', 'full-face-locality', 'full-face-reduce-pair', 'full-face-group', 'full-primary-visibility', 'full-visibility-face-plan', 'full-selected-diffuse', 'full-environment-cost'), required=True)
    parser.add_argument('--face-group-size', type=int, choices=(8,16), default=8)
    parser.add_argument('--face-lighting-budget', type=int, choices=range(1,33))
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--baseline', choices=('full', 'trace'), default='full')
    parser.add_argument('--warmup', type=int, default=12)
    parser.add_argument('--width', type=int, default=3840)
    parser.add_argument('--height', type=int, default=2160)
    parser.add_argument('--overview', action='store_true')
    parser.add_argument('--material-fixture', choices=('default', 'emissive', 'reflective', 'refractive'), default='default')
    parser.add_argument('--stable-emitter-fixture', action='store_true', help='Diagnostic deterministic face enumeration; not a performance option')
    parser.add_argument('--projection', choices=('perspective','orthographic'), default='perspective')
    parser.add_argument('--face-baseline-stripes', type=int, choices=(32,128), default=128)
    parser.add_argument('--face-stripes', type=int, choices=(16,32,64), default=32)
    parser.add_argument('--split-face', action='store_true', help='Diagnostic uncached per-pass face averaging GPU timings')
    parser.add_argument('--split-denoiser', action='store_true')
    parser.add_argument('--reset-every', type=int, default=0, help='Diagnostic history reset period; zero disables')
    parser.add_argument('--history-policy', choices=('reset_on_content','local'), default='reset_on_content')
    parser.add_argument('--split-spatial', action='store_true')
    parser.add_argument('--spatial-iterations', type=int, choices=range(1,6), default=3)
    parser.add_argument('--check-each-frame', action='store_true', help='Diagnostic HDR comparisons; excludes throughput conclusions')
    parser.add_argument('--primary-hit-format', choices=('full','identity'))
    parser.add_argument("--environment-samples",type=int,choices=range(5),default=1)
    parser.add_argument('--profile-selected-diffuse',action='store_true')
    parser.add_argument('--quad-visible-links',action='store_true')
    parser.add_argument('--selected-baseline',action='store_true',help='Compare selected-diffuse linking variants directly')
    parser.add_argument('--replay-width',type=int,choices=(8,16,32,64),default=8)
    parser.add_argument('--omit-primary-analytic',action='store_true',help='Cost ablation only: omit primary analytic diffuse AND specular light')
    parser.add_argument('--single-visibility-dispatch',action='store_true',help='Capture the full visibility plane with one dispatch')
    parser.add_argument('--distance-visibility',action='store_true',help='Use the 100-byte distance visibility cache')
    parser.add_argument('--planar-visibility',action='store_true',help='Use seven vector field planes for visibility')
    parser.add_argument("--distance-planar-visibility",action="store_true",help="Use six vector planes plus packed distances")
    parser.add_argument("--baseline-visibility-format",choices=("full","planes"),default="full",help="Cache layout for --selected-baseline")
    parser.add_argument("--compare-capture-dependencies",action="store_true",help="Compare conservative versus precise capture uses with matched selected-diffuse options")
    parser.add_argument("--selected-deferred-capture",action="store_true",help="Test deferred complete secondary-record initialization in selected replay")
    parser.add_argument("--selected-fixed-mask",choices=("none","all"),help="Deterministic transport check; bypasses face reconstruction, requires per-frame comparisons")
    parser.add_argument("--selected-local-state",action="store_true",help="Test private local-only records for rejected diffuse continuations against deferred full records")
    parser.add_argument("--selected-local-packed",action="store_true",help="Store local-only state in five contiguous vectors; requires local-state experiment")
    args = parser.parse_args()
    if (args.reject_dark_environment or args.fused_resolve or args.candidate_direct or args.candidate_history or args.candidate_iterations or args.candidate_tile_capacity) and args.stage != 'full-control':
        parser.error('Maintained-path levers require --stage full-control')
    if args.selected_local_packed and not args.selected_local_state:
        parser.error("Packed local state requires --selected-local-state")
    if args.selected_local_state and (args.stage != "full-selected-diffuse" or not args.selected_baseline or args.selected_deferred_capture or args.compare_capture_dependencies or args.omit_primary_analytic):
        parser.error("Local-state comparison requires matched selected baseline without other ablations")
    if args.compact_continuations and (args.stage != "full-selected-diffuse" or not args.selected_baseline or args.selected_local_state or args.selected_deferred_capture or args.compare_capture_dependencies or args.omit_primary_analytic):
        parser.error("Compact continuations requires selected baseline without other transport ablations")
    matched_selected_options=args.compact_continuations or args.compare_capture_dependencies or args.selected_deferred_capture or bool(args.selected_fixed_mask) or args.selected_local_state
    if args.selected_deferred_capture and (args.stage != "full-selected-diffuse" or not args.selected_baseline or args.compare_capture_dependencies or args.omit_primary_analytic):
        parser.error("Deferred selected initialization requires a matched selected baseline without other ablations")
    if args.selected_fixed_mask and (not args.selected_baseline or args.stage != "full-selected-diffuse" or not args.check_each_frame):
        parser.error("Fixed selection masks require selected baseline and per-frame checks")
    if args.compare_capture_dependencies and (args.stage != "full-selected-diffuse" or not args.selected_baseline):
        parser.error("Capture dependency comparison requires selected-diffuse on both sides")
    if args.baseline_visibility_format != "full" and not args.selected_baseline:
        parser.error("Baseline cache format requires --selected-baseline")
    if args.compare_capture_dependencies and args.omit_primary_analytic:
        parser.error("Capture dependency comparison cannot also ablate lighting")
    if args.distance_planar_visibility and (args.planar_visibility or args.distance_visibility or args.stage != "full-selected-diffuse"):
        parser.error("Compact planar visibility requires selected-diffuse and excludes other cache formats")
    if args.planar_visibility and (args.distance_visibility or args.stage != 'full-selected-diffuse'):
        parser.error('Planar visibility requires selected-diffuse and excludes distance visibility')
    if args.distance_visibility and args.stage != 'full-selected-diffuse':
        parser.error('Distance visibility requires selected-diffuse')
    if args.single_visibility_dispatch and args.stage != 'full-selected-diffuse':
        parser.error('Single visibility dispatch requires selected-diffuse')
    if args.stage == 'full-environment-cost' and (args.environment_samples == 0 or args.check_each_frame):
        parser.error('Environment cost requires environment samples and disables HDR parity')
    if args.omit_primary_analytic and (args.stage != 'full-selected-diffuse' or args.check_each_frame):
        parser.error('Analytic ablation requires selected-diffuse and cannot check HDR parity')
    if args.selected_baseline and args.stage != 'full-selected-diffuse':
        parser.error('--selected-baseline requires full-selected-diffuse')
    if args.face_lighting_budget and not args.stage.startswith('full-'):
        parser.error('Lighting sample planning requires a full reference render')
    if args.split_spatial: args.split_denoiser = True
    if min(args.frames, args.warmup, args.width, args.height) < 1:
        parser.error('frames, warmup and extent must be positive')
    if args.baseline == 'trace' and args.stage.startswith('full-'):
        parser.error('full-render candidates require the full baseline for HDR parity')
    import generate_fused_primary as generator
    from ordinarylight.targets.vulkan.api import RendererConfig
    from ordinarylight.targets.vulkan import core, primary_graph
    from vxl8r_render.backends import DynamicResidentVulkanRenderer
    from vxl8r_render.backends.gpu_timing import GpuGraphTimer
    from vxl8r_render.viewer.scenes import scene_sources
    from vxl8r_render.viewer.state import ViewerState
    state = ViewerState(native_gi=True, playing=True, projection=args.projection)
    if args.overview:
        state.zoom = 138.24
    spec, _, sources = scene_sources('voxel', 512, sparse=True,emitter_object=args.emitter_object,uneven_floor=args.uneven_floor)
    if args.material_fixture == 'emissive':
        from vxl8r_render import Cell, GridSnapshot, VoxelSource
        sources[1] = VoxelSource(GridSnapshot(spec, {
            key: Cell(cell.albedo, (3.0, 1.5, 0.5)) for key, cell in sources[1].voxels.cells.items()
        }))
    config = RendererConfig(wavefront_profiling=args.transport_counters,max_bounces=4, samples_per_pixel=1, wavefront_environment_samples=args.environment_samples, denoiser_enabled=True,
        temporal_history=True, progressive_accumulation=True, denoiser_sampled_indirect=True,
        denoiser_iterations=args.spatial_iterations,
        wavefront_raw_hdr_output=True, wavefront_timestamps=True, wavefront_tile_capacity=524288,
        wavefront_primary_hit_format=args.primary_hit_format or ("identity" if args.stage in ("full-secondary-clear", "full-secondary-fields", "full-prepare-fields", "full-prepare-single", "full-reset-fusion", "full-paired-spatial", "full-as-2048", "full-primary-capture", "full-primary-clear-store", "full-primary-queue", "full-primary-shared-pbr", "full-face-anchor", "full-face-stripes", "full-face-combined", "full-face-table", "full-face-direct", "full-face-worklist", "full-face-locality", "full-face-reduce-pair", "full-face-group") else "full"))
    def comparable(source):
        if args.emitter_object and not args.stable_emitter_fixture:
            return False
        if source!='raw' and (args.candidate_history or args.candidate_iterations):
            return False
        return not (source=='output' and args.candidate_direct)
    original_expand, original_program = compiler._expanded_shader_source, generator.p
    selected = 'full'
    try:
        with tempfile.TemporaryDirectory(prefix='ordinarylight-primary-prefix-') as temp:
            if args.stage in ('full-hit-planes', 'full-compact-hits'):
                from primary_hit_layout_experiment import variants, unpack_hits
                generated, plane_consumer = variants(Path(temp), compact=args.stage == 'full-compact-hits')
            else:
                if args.selected_local_state:
                    from selected_local_state_experiment import variant
                    generator.p = variant(Path(temp),packed=args.selected_local_packed)
                elif args.selected_deferred_capture:
                    from primary_capture_store_experiment import variant
                    generator.p = variant(Path(temp))
                elif args.stage == 'full-environment-cost':
                    from primary_environment_experiment import variant
                    generator.p = variant(Path(temp))
                elif args.stage == 'full-primary-shared-pbr':
                    from primary_shared_pbr_experiment import variant
                    generator.p = variant(Path(temp))
                elif args.stage == 'full-primary-queue':
                    from primary_queue_store_experiment import variant
                    generator.p = variant(Path(temp))
                elif args.stage in ('full-primary-capture', 'full-primary-clear-store'):
                    from primary_capture_store_experiment import variant
                    generator.p = variant(Path(temp), clear_only=args.stage == 'full-primary-clear-store')
                else:
                    generator.p = prefix_module(args.stage, Path(temp))
                generated = None if args.stage in ('full-control', 'full-public-identity', 'full-secondary-clear', 'full-secondary-fields', 'full-prepare-fields', 'full-prepare-single', 'full-reset-fusion', 'full-paired-spatial', 'full-as-2048', 'full-primary-visibility', 'full-visibility-face-plan', 'full-selected-diffuse', 'full-face-anchor', 'full-face-stripes', 'full-face-combined', 'full-face-table', 'full-face-direct', 'full-face-worklist', 'full-face-locality', 'full-face-reduce-pair', 'full-face-group') else generator.generated_source()
            if args.selected_deferred_capture or args.selected_local_state:
                generated = generator.generated_source()
            secondary_generated = None
            if args.stage == 'full-secondary-fields':
                import generate_core_shaders as secondary_generator
                from primary_hit_layout_experiment import load_module
                source = Path(secondary_generator.__file__).read_text()
                original = """        secondary = secondary_paths[path_index]
        secondary.position_valid = osh.vec4(loaded.hit.position_t.xyz, 1.0)
        secondary.normal_pdf = osh.vec4(
            surface.normal, secondary.normal_pdf.w
        )
        secondary_paths[path_index] = secondary"""
                replacement = """        secondary_paths[path_index].position_valid = osh.vec4(loaded.hit.position_t.xyz, 1.0)
        secondary_paths[path_index].normal_pdf = osh.vec4(
            surface.normal, secondary_paths[path_index].normal_pdf.w
        )"""
                assert source.count(replacement) == 1
                module = load_module('secondary_field_diagnostic', source.replace(replacement, original), Path(temp))
                module.ROOT = secondary_generator.ROOT
                secondary_generated = module.generated_source(
                    module.wavefront_shade_candidate, module.WAVEFRONT_SHADE_CANDIDATE_HELPERS)
            baseline_generated = None
            if args.selected_local_state:
                from primary_capture_store_experiment import variant as deferred_variant
                generator.p=deferred_variant(Path(temp))
                baseline_generated=generator.generated_source()
            if args.baseline == 'trace':
                generator.p = prefix_module('trace', Path(temp))
                baseline_generated = generator.generated_source()
            generator.p = original_program
            def expanded(name, seen=()):
                # Reconstruct the old whole-record store only for the baseline.
                if name == 'wavefront_shade_candidate.glsl' and args.stage == 'full-secondary-fields' and selected == 'full':
                    return secondary_generated
                chosen_source = baseline_generated if selected == 'full' else generated
                if chosen_source is None or name != 'wavefront_primary_impl.glsl':
                    return original_expand(name, seen)
                lines = []
                for line in chosen_source.splitlines():
                    stripped = line.strip()
                    if stripped.startswith('#include "') and stripped.endswith('"'):
                        include = stripped[len('#include "'):-1]
                        lines.append(original_expand(include, (*seen, name)))
                    else:
                        lines.append(line)
                return '\n'.join(lines) + '\n'
            compiler._expanded_shader_source = expanded
            with ExitStack() as stack:
                if args.selected_local_state:
                    from ordinarylight.runtime import relax_prepare
                    from selected_local_state_experiment import compile_prepare
                    local_prepare=compile_prepare(Path(temp),packed=args.selected_local_packed)
                    original_prepare=relax_prepare._custom_history_shader
                    stack.enter_context(patch.object(relax_prepare,'_custom_history_shader',
                        lambda:local_prepare if selected==args.stage else original_prepare()))
                    if args.selected_local_packed:
                        from ordinarylight.targets.vulkan import path_resolve_graph
                        from selected_local_state_experiment import compile_resolve
                        local_resolve=compile_resolve(Path(temp))
                        original_resolve_kernel=path_resolve_graph.VulkanKernel
                        def resolve_kernel(runtime,spirv,*a,**kw):
                            bindings=a[0]
                            aliased=bindings[3].handle==bindings[2].handle and bindings[5].handle==bindings[2].handle
                            if args.check_each_frame:
                                print(json.dumps(dict(native_resolve_placeholders_aliased=aliased,workload=selected)),flush=True)
                            # Native resolve now disables unused seed writes. Keep its
                            # dormant placeholders to exercise that production policy.
                            return original_resolve_kernel(runtime,local_resolve if selected==args.stage else spirv,*a,**kw)
                        stack.enter_context(patch.object(path_resolve_graph,'VulkanKernel',resolve_kernel))
                if args.stage == 'full-selected-diffuse':
                    from selected_diffuse_render_experiment import install
                    diffuse_builder,diffuse_states=install(stack,lambda:selected==args.stage or args.selected_baseline,
                        lambda:renderers[selected].capacity,extent=(args.width,args.height),
                        budget=args.face_lighting_budget or 8,profile=args.profile_selected_diffuse,quad_links=lambda:args.quad_visible_links and (selected==args.stage or matched_selected_options),
                        replay_workgroup=lambda:(args.replay_width,64//args.replay_width) if selected==args.stage or matched_selected_options else (8,8),
                        omit_primary_analytic=lambda:args.omit_primary_analytic and selected==args.stage,
                        single_capture=lambda:args.single_visibility_dispatch and (selected==args.stage or matched_selected_options),
                        fixed_mask=args.selected_fixed_mask,
                        compact_continuations=lambda:args.compact_continuations and selected==args.stage,
                        conservative_capture=lambda:args.compare_capture_dependencies and selected!=args.stage,
                        visibility_format=lambda:("distance_planes" if args.distance_planar_visibility else "planes" if args.planar_visibility else "distance" if args.distance_visibility else "full") if selected==args.stage or matched_selected_options else args.baseline_visibility_format)
                if args.stage == 'full-visibility-face-plan':
                    from visibility_face_plan_experiment import install
                    visibility_plans = install(stack, lambda:selected == args.stage,
                        lambda:renderers[args.stage].capacity, extent=(args.width,args.height))
                if args.stage == 'full-environment-cost':
                    from primary_visibility_experiment import install
                    install(stack,lambda:True)
                if args.stage == 'full-primary-visibility':
                    from primary_visibility_experiment import install
                    install(stack, lambda: selected == args.stage)
                if args.stage == 'full-face-group':
                    from face_group_size_experiment import install
                    install(stack, Path(temp), lambda: selected == args.stage, args.face_group_size)
                if args.stage == 'full-face-reduce-pair':
                    from face_reduce_pair_experiment import install
                    install(stack, Path(temp), lambda: selected == args.stage)
                if args.stage == 'full-face-locality':
                    from face_stripe_locality_experiment import install
                    install(stack, Path(temp), lambda: selected == args.stage)
                if args.stage == 'full-face-worklist':
                    from face_worklist_experiment import install
                    install(stack, Path(temp), lambda: selected == args.stage)
                if args.stage == 'full-face-direct':
                    from face_direct_anchor_experiment import install
                    install(stack, Path(temp), lambda: selected == args.stage)
                if args.stage == 'full-face-table':
                    from face_table_experiment import install
                    install(stack, Path(temp), lambda: selected == args.stage)
                if args.stage in ('full-face-stripes', 'full-face-combined'):
                    from vxl8r_render.backends import sparse_average
                    original_average_init = sparse_average.SparseFaceAverageTarget.__init__
                    def average_init(target, *a, **kw):
                        original_average_init(target, *a, **kw)
                        limit = 1 << ((4294967295 // (target.slots * 6)).bit_length() - 1)
                        target.stripes = min(limit, args.face_stripes if selected == args.stage else args.face_baseline_stripes)
                    stack.enter_context(patch.object(sparse_average.SparseFaceAverageTarget,
                        '__init__', average_init))
                if args.stage in ('full-face-anchor', 'full-face-combined'):
                    from face_anchor_experiment import install
                    install(stack, Path(temp), lambda: selected == args.stage)
                if args.split_face:
                    from vxl8r_render.backends import sparse_average, gpu_timing
                    from primary_hit_layout_experiment import load_module
                    timer_source = Path(gpu_timing.__file__).read_text()
                    marker = '                name=node.name\n'
                    assert timer_source.count(marker) == 1
                    timer_source = timer_source.replace(marker, marker + '''                if name == 'face_output':
                    labels = ('clear', 'link', 'prepare_dispatch', 'reduce', 'finish', 'resolve')
                    if len(node.operation.passes) != len(labels):
                        raise RuntimeError('Unexpected face averaging pass layout')
                    name = 'face.' + labels[i]
''')
                    GpuGraphTimer = load_module('diagnostic_face_timer', timer_source, Path(temp)).GpuGraphTimer
                    stack.enter_context(patch.object(sparse_average.SparseFaceAverageTarget,
                        'operation', sparse_average.SparseFaceAverageTarget._uncached_operation))
                if args.stage == 'full-as-2048':
                    from vxl8r_render.backends import native_acceleration
                    OriginalAcceleration = native_acceleration.NativeAcceleration
                    class SelectedAcceleration(OriginalAcceleration):
                        # Keep class-level brick ordering unchanged; alter only BLAS grouping.
                        def __init__(self, *a, **kw):
                            self.chunk_size = 2048 if selected == args.stage else 4096
                            super().__init__(*a, **kw)
                    stack.enter_context(patch.object(native_acceleration, 'NativeAcceleration', SelectedAcceleration))
                if args.stage == 'full-paired-spatial':
                    from paired_spatial_experiment import install
                    install(stack, Path(temp), lambda: selected == args.stage)
                if args.stage == 'full-reset-fusion':
                    from reset_prepare_fusion_experiment import install
                    install(stack, Path(temp), lambda: selected == args.stage)
                if args.stage in ('full-prepare-fields', 'full-prepare-single'):
                    from ordinarylight.runtime import relax_prepare
                    from prepare_read_experiment import compile_variant
                    prepare_candidate = compile_variant(Path(temp), single_sample=args.stage == 'full-prepare-single')
                    prepare_baseline = relax_prepare._custom_history_shader
                    stack.enter_context(patch.object(relax_prepare, '_custom_history_shader',
                        lambda: prepare_candidate if selected == args.stage else prepare_baseline()))
                if args.stage == 'full-secondary-clear':
                    original_clear = core.VulkanWavefrontExecutor._needs_secondary_transfer_clear
                    def needs_clear(executor, fused):
                        return executor._denoiser_signals_active() if selected == 'full' else original_clear(executor, fused)
                    stack.enter_context(patch.object(core.VulkanWavefrontExecutor, '_needs_secondary_transfer_clear', needs_clear))
                if args.material_fixture in ('reflective', 'refractive') or args.stable_emitter_fixture:
                    from vxl8r_render.backends import native_geometry
                    from primary_hit_layout_experiment import load_module
                    material_source = Path(native_geometry.__file__).read_text()
                    if args.material_fixture in ('reflective', 'refractive'):
                        material_source = material_source.replace('osh.vec4(value.albedo_kind.xyz, 1.0)',
                            'osh.vec4(value.albedo_kind.xyz, 0.0)')
                    if args.material_fixture == 'reflective':
                        material_source = material_source.replace('osh.vec4(value.emission.xyz, 0.0)', 'osh.vec4(value.emission.xyz, 1.0)')
                    elif args.material_fixture == 'refractive':
                        material_source = material_source.replace('osh.vec4(1.0, 1.0, 1.0, 0.0)', 'osh.vec4(1.0, 1.0, 1.0, 1.0)')
                    if args.stable_emitter_fixture:
                        material_source = material_source.replace('    return voxel_emitter_faces[lower * voxel_emitter_prefix[1] * osh.u32(6) + offset]', '''    face = lower * voxel_emitter_prefix[1] * osh.u32(6)
    end = osh.minimum(face + voxel_emitter_prefix[1] * osh.u32(6), osh.array_length(voxel_exposed))
    while face < end:
        slot = face / osh.u32(6)
        if voxel_records[slot].metadata.x != osh.u32(4294967295) and voxel_exposed[face] != osh.u32(0) and osh.any_value(voxel_palette[slot].emission.xyz > osh.vec3(0.0)):
            if offset == osh.u32(0):
                return face
            offset = offset - osh.u32(1)
        face = face + osh.u32(1)
    return osh.u32(4294967295)''')
                    material_module = load_module('vxl8r_render.backends.diagnostic_material', material_source, Path(temp))
                    native_geometry.native_voxel_program.cache_clear()
                    stack.callback(native_geometry.native_voxel_program.cache_clear)
                    stack.enter_context(patch.object(native_geometry, 'evaluate_voxel', material_module.evaluate_voxel))
                    if args.stable_emitter_fixture:
                        stack.enter_context(patch.object(native_geometry, 'select_compact_emitter', material_module.select_compact_emitter))
                if args.stage in ('full-hit-planes', 'full-compact-hits'):
                    from vxl8r_render.backends import sparse_average
                    original_consumer = sparse_average.compiled_programs
                    def consumer(image_format):
                        return (original_consumer if selected == 'full' else plane_consumer)(image_format)
                    stack.enter_context(patch.object(sparse_average, 'compiled_programs', consumer))
                original_dispatch = core.VulkanWavefrontExecutor.dispatch
                original_record = primary_graph.record_primary
                active_timestamp = None
                last_timestamp = {}
                def dispatch(executor, *positional, **kwargs):
                    nonlocal active_timestamp
                    active_timestamp = kwargs.get('timestamp')
                    last_timestamp[executor.core.runtime] = active_timestamp
                    try:
                        return original_dispatch(executor, *positional, **kwargs)
                    finally:
                        active_timestamp = None
                def record_primary(executor, command, *positional):
                    if active_timestamp is not None:
                        active_timestamp(command, 'primary_setup')
                    return original_record(executor, command, *positional)
                stack.enter_context(patch.object(core.VulkanWavefrontExecutor, 'dispatch', dispatch))
                stack.enter_context(patch.object(primary_graph, 'record_primary', record_primary))
                if args.split_denoiser:
                    from ordinarylight.runtime.relax import VulkanRelaxSpatial
                    from ordinarylight.pipeline.vulkan import VulkanPass
                    spatial_operation = VulkanRelaxSpatial.operation
                    def measured_spatial(stage, *positional, **kwargs):
                        operation = spatial_operation(stage, *positional, **kwargs)
                        timestamp = last_timestamp.get(stage.runtime)
                        if timestamp is not None:
                            if args.split_spatial:
                                from dataclasses import replace
                                import vulkan as vk
                                def measured_pass(p):
                                    def record(command):
                                        p.record(command)
                                        if p.workgroups is not None:
                                            vk.vkCmdDispatch(command, *p.workgroups)
                                        timestamp(command, 'spatial.' + p.name)
                                    return replace(p, record=record, workgroups=None)
                                operation.passes = tuple(measured_pass(p) for p in operation.passes)
                            operation.passes = (VulkanPass('temporal_end', (),
                                lambda command: timestamp(command, 'temporal_filter')), *operation.passes)
                        return operation
                    stack.enter_context(patch.object(VulkanRelaxSpatial, 'operation', measured_spatial))
                history_counts = {}
                update_policy = core.VulkanWavefrontExecutor.update_relax_temporal_constants
                def observed_policy(executor, slot, width, height, valid, **policy):
                    counts = history_counts.setdefault(selected, {'valid': 0, 'reset': 0})
                    counts['valid' if valid else 'reset'] += 1
                    return update_policy(executor, slot, width, height, valid, **policy)
                stack.enter_context(patch.object(core.VulkanWavefrontExecutor, 'update_relax_temporal_constants', observed_policy))
                renderers = {}
                for name in ('full', args.stage):
                    from dataclasses import replace
                    selected = name
                    renderer_config = replace(config, wavefront_primary_hit_format="identity") if name == "full-public-identity" else config
                    if name == args.stage:
                        renderer_config=replace(renderer_config,denoiser_fused_resolve=args.fused_resolve,
                            wavefront_environment_early_reject=args.reject_dark_environment,
                            denoiser_iterations=args.candidate_iterations or args.spatial_iterations,
                            wavefront_tile_capacity=args.candidate_tile_capacity or config.wavefront_tile_capacity)
                    from ordinarylight import EnvironmentLight
                    renderers[name] = stack.enter_context(DynamicResidentVulkanRenderer(
                        spec, sources, bounds=((-256,-5,-256),(255,7,255)),
                        static_coordinates=sources[0].voxels.cells, static_sources=(0,),
                        chunk_capacity=128, gpu_animation=True, gpu_layout=True,
                        tight_dynamic_bounds=True, config=renderer_config,
                        environment=EnvironmentLight(color=(0,0,0)) if args.emitter_object else None,
                        history_policy=(args.candidate_history or args.history_policy) if name==args.stage else args.history_policy))
                if args.stage == 'full-selected-diffuse':
                    renderers[args.stage].pipeline.set_gi_pipeline_builder(diffuse_builder,reuse_commands=True)
                    if args.selected_baseline:
                        renderers['full'].pipeline.set_gi_pipeline_builder(diffuse_builder,reuse_commands=True)
                def frame(name, index):
                    nonlocal selected
                    selected = name
                    r = renderers[name]
                    if args.reset_every and index % args.reset_every == 0:
                        r.pipeline.invalidate_gi_history()
                    state.time = index / 60.0
                    r.set_transform(r.objects[1], state.transform())
                    return r.render(camera=state.camera(), width=args.width, height=args.height,
                                    render_extent=(args.width,args.height), averaged=args.stage.startswith("full-") and not (args.candidate_direct and name==args.stage))
                for name in renderers:
                    frame(name, 0).wait()
                timers = {name: stack.enter_context(GpuGraphTimer(r.runtime)) for name,r in renderers.items()}
                rows = {name: [] for name in renderers}
                setup_rows = {name: [] for name in renderers}
                total_rows = {name: [] for name in renderers}
                face_rows = {name: [] for name in renderers}
                face_stage_rows = {name: [] for name in renderers}
                secondary_rows = {name: [] for name in renderers}
                prepare_rows = {name: [] for name in renderers}
                temporal_rows = {name: [] for name in renderers}
                filter_rows = {name: [] for name in renderers}
                spatial_rows = {name: [] for name in renderers}
                for index in range(args.warmup + args.frames):
                    order = tuple(renderers) if index % 2 == 0 else tuple(reversed(renderers))
                    for name in order:
                        frame(name, index+1).wait()
                        if index >= args.warmup:
                            spatial_rows[name].append({k:v for k,v in renderers[name].pipeline.last_timings['wavefront_stage_ms'].items() if k.startswith('spatial.')})
                            filter_rows[name].append(renderers[name].pipeline.last_timings['wavefront_stage_ms'].get('temporal_filter', 0.0))
                            temporal_rows[name].append(renderers[name].pipeline.last_timings['wavefront_stage_ms'].get('relax_temporal', 0.0))
                            prepare_rows[name].append(renderers[name].pipeline.last_timings['wavefront_stage_ms'].get('relax_prepare', 0.0))
                            secondary_rows[name].append(renderers[name].pipeline.last_timings['wavefront_stage_ms'].get('intersect_shade.1', 0.0))
                            rows[name].append(renderers[name].pipeline.last_timings['wavefront_stage_ms']['primary'])
                            setup_rows[name].append(renderers[name].pipeline.last_timings['wavefront_stage_ms']['primary_setup'])
                            total_rows[name].append(timers[name].last['total'])
                            face_stage_rows[name].append({k:v for k,v in timers[name].last.items() if k.startswith('face.')})
                            face_rows[name].append(timers[name].last.get('face_output', sum(face_stage_rows[name][-1].values())))
                    if args.capture_dir and index in (args.warmup,args.warmup+args.frames//2,args.warmup+args.frames-1):
                        args.capture_dir.mkdir(parents=True,exist_ok=True)
                        for capture_name,capture_renderer in renderers.items():
                            capture_renderer.read_image().save(args.capture_dir/f'{capture_name}-{index:04d}.png')
                    if (args.selected_local_state or args.compact_continuations or args.fused_resolve or args.reject_dark_environment) and args.check_each_frame and comparable("upstream"):
                        from selected_local_state_experiment import read_guide
                        for guide in ('normal_roughness','motion','view_z','diffuse','specular'):
                            ga,gb=(read_guide(renderers[n],guide) for n in ('full',args.stage))
                            print(json.dumps(dict(frame=index,guide=guide,max_error=float(np.max(abs(ga-gb))))),flush=True)
                            np.testing.assert_array_equal(ga,gb,err_msg=f'{index}: {guide}')
                    if args.check_each_frame and (args.stage != 'full-selected-diffuse' or (args.environment_samples > 0 or bool(args.selected_fixed_mask))):
                        for source in ('raw', 'upstream', 'output'):
                            a=renderers['full'].read_hdr(source=source)
                            b=renderers[args.stage].read_hdr(source=source)
                            assert np.isfinite(a).all() and np.isfinite(b).all()
                            if comparable(source):np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6,err_msg=f'{index}: {source}')
                for name, values in rows.items():
                    workload = args.baseline if name == 'full' else name
                    print(json.dumps(dict(workload=workload, extent=[args.width,args.height], samples=len(values),
                        primary_hit_format=renderers[name].pipeline.config.wavefront_primary_hit_format,
                        environment_samples=args.environment_samples,
                        analytic_light_count=renderers[name].resident.scene.analytic_light_count,
                        distance_planar_visibility=args.distance_planar_visibility and (name==args.stage or matched_selected_options),
                        planar_visibility=args.planar_visibility and (name==args.stage or matched_selected_options),
                        distance_visibility=args.distance_visibility and (name==args.stage or matched_selected_options),
                        single_visibility_dispatch=args.single_visibility_dispatch and (name==args.stage or matched_selected_options),
                        omit_primary_analytic=args.omit_primary_analytic and name==args.stage,
                        omit_primary_environment=args.stage=='full-environment-cost' and name==args.stage,
                        profile_selected_diffuse=args.profile_selected_diffuse,replay_width=args.replay_width if name==args.stage or matched_selected_options else 8,
                        baseline_visibility_format=args.baseline_visibility_format,
                        compare_capture_dependencies=args.compare_capture_dependencies,
                        selected_local_packed=args.selected_local_packed and name==args.stage,
                        compact_continuations=args.compact_continuations and name==args.stage,
                        selected_local_state=args.selected_local_state and name==args.stage,
                        baseline_deferred_capture=args.selected_local_state and name!=args.stage,
                        selected_deferred_capture=args.selected_deferred_capture and name==args.stage,
                        selected_fixed_mask=args.selected_fixed_mask,
                        conservative_capture=args.compare_capture_dependencies and name!=args.stage,
                        selected_baseline=args.selected_baseline,quad_visible_links=args.quad_visible_links and (name==args.stage or matched_selected_options),
                        candidate_group_size=args.face_group_size if args.stage == "full-face-group" else None,
                        face_averaged=args.stage.startswith("full-") and not (args.candidate_direct and name==args.stage),
                        emitter_object=args.emitter_object,uneven_floor=args.uneven_floor,
                        diagnostic_capture=bool(args.capture_dir),
                        work_counters=renderers[name].pipeline.last_timings.get("wavefront_work_counters",{}),
                        fused_resolve=renderers[name].config.denoiser_fused_resolve,
                        reject_dark_environment=renderers[name].config.wavefront_environment_early_reject,
                        tile_capacity=renderers[name].config.wavefront_tile_capacity,
                        baseline_face_stripes=args.face_baseline_stripes if args.stage in ("full-face-stripes", "full-face-combined") else None,
                        candidate_face_stripes=args.face_stripes if args.stage in ("full-face-stripes", "full-face-combined") else None,
                        split_face=args.split_face,
                        face_stage_ms={k:float(np.median([r[k] for r in face_stage_rows[name]])) for k in face_stage_rows[name][0]},
                        spatial_iterations=renderers[name].config.denoiser_iterations, checked_each_frame=args.check_each_frame and (args.stage != 'full-selected-diffuse' or (args.environment_samples > 0 or bool(args.selected_fixed_mask))),
                        spatial_ms={k:float(np.median([r[k] for r in spatial_rows[name]])) for k in spatial_rows[name][0]},
                        projection=args.projection,
                        history_policy=renderers[name].history_policy,
                        history_policy_frames=history_counts.get(name),
                        split_denoiser=args.split_denoiser, reset_every=args.reset_every,
                        temporal_filter_median_ms=float(np.median(filter_rows[name])),
                        temporal_median_ms=float(np.median(temporal_rows[name])),
                        prepare_median_ms=float(np.median(prepare_rows[name])),
                        secondary_first_median_ms=float(np.median(secondary_rows[name])),
                        primary_median_ms=float(np.median(values)), primary_p95_ms=float(np.percentile(values,95)),
                        setup_median_ms=float(np.median(setup_rows[name])),
                        total_median_ms=float(np.median(total_rows[name])),
                        face_output_median_ms=float(np.median(face_rows[name])), overview=args.overview,
                        material_fixture=args.material_fixture,
                        stable_emitter_fixture=args.stable_emitter_fixture,
                        diagnostic_only=True, prefix_hdr_valid=(not args.omit_primary_analytic and args.stage != 'full-environment-cost' and (workload == 'full' or workload.startswith('full-'))))), flush=True)
                if args.stage == 'full-visibility-face-plan':
                    from visibility_face_plan_experiment import summaries
                    print(json.dumps(dict(pre_lighting_plans=summaries(visibility_plans))),flush=True)
                if args.face_lighting_budget:
                    from face_lighting_plan_experiment import profile_plan
                    print(json.dumps(dict(face_lighting_plan=profile_plan(renderers['full'], budget=args.face_lighting_budget))), flush=True)
                if args.stage == 'full-selected-diffuse':
                    from selected_diffuse_render_experiment import summaries
                    print(json.dumps(dict(selected_diffuse=summaries(diffuse_states))),flush=True)
                if args.stage.startswith('full-'):
                    for source in ('raw', 'upstream', 'output'):
                        a = renderers['full'].read_hdr(source=source)
                        b = renderers[args.stage].read_hdr(source=source)
                        assert np.isfinite(a).all() and np.isfinite(b).all(), source
                        print(json.dumps(dict(source=source, max_error=float(np.max(abs(a-b))), mean_error=float(np.mean(abs(a-b))), baseline_max=float(np.max(a)))), flush=True)
                        if not args.omit_primary_analytic and args.stage != 'full-environment-cost' and (args.stage != 'full-selected-diffuse' or (args.environment_samples > 0 or bool(args.selected_fixed_mask))):
                            if comparable(source):np.testing.assert_allclose(a, b, atol=0.000002, rtol=0.000002)
                if args.stage == 'trace-no-export':
                    print(json.dumps(dict(sampled_primary_hits_equal=None, reason='Primary hit exports disabled; no hit parity assertion.')), flush=True)
                    return
                reference = renderers['full'].read_primary_hits()
                candidate = renderers[args.stage].read_primary_hits()
                if 'valid' in reference.dtype.names and 'valid' in candidate.dtype.names:
                    if args.stage == 'synthetic':
                        np.testing.assert_array_equal(candidate['identity'], 0)
                        np.testing.assert_array_equal(candidate['valid'], 1)
                        print(json.dumps(dict(synthetic_identity_valid=True, camera_ray_parity_checked=False)), flush=True)
                    else:
                        np.testing.assert_array_equal(reference, candidate)
                        print(json.dumps(dict(compact_identity_validity_equal=True, camera_ray_parity_checked=False)), flush=True)
                    return
                if args.stage in ('full-secondary-clear', 'full-secondary-fields', 'full-prepare-fields', 'full-prepare-single', 'full-reset-fusion', 'full-paired-spatial'):
                    np.testing.assert_array_equal(reference, candidate)
                    print(json.dumps(dict(compact_identity_validity_equal=True, optimization=args.stage)), flush=True)
                    return
                if args.stage == 'full-public-identity':
                    np.testing.assert_array_equal(reference['identity'], candidate['identity'])
                    np.testing.assert_array_equal(reference['position_distance'][..., 3] >= 0, candidate['valid'] != 0)
                    assert renderers[args.stage].frame.buffers['primary_hits'].byte_size == candidate.size * 20
                    print(json.dumps(dict(compact_identity_validity_equal=True, public_api=True, bytes_per_hit=20)), flush=True)
                    return
                if args.stage == 'full-compact-hits':
                    count = reference.size
                    words = candidate.view(np.uint32).reshape(-1)[:count * 5].reshape(*reference.shape, 5)
                    np.testing.assert_array_equal(reference['identity'], words[..., :4])
                    np.testing.assert_array_equal(reference['position_distance'][..., 3] >= 0, words[..., 4] != 0)
                    print(json.dumps(dict(compact_identity_validity_equal=True, bytes_written_per_hit=20)), flush=True)
                    return
                if args.stage == 'full-hit-planes':
                    candidate = unpack_hits(candidate)
                if args.stage == 'synthetic':
                    np.testing.assert_array_equal(candidate['identity'], 0)
                    np.testing.assert_array_equal(candidate['position_distance'][..., 3], 1.0)
                fields = ('ray_origin', 'ray_direction') if args.stage == 'synthetic' else ('identity','ray_origin','ray_direction','position_distance','geometric_normal')
                if args.stage == 'full-hit-planes':
                    fields = (*fields, 'shading_normal')
                for field in fields:
                    np.testing.assert_array_equal(reference[field], candidate[field], err_msg=field)
                print(json.dumps(dict(sampled_primary_hits_equal=args.stage != 'synthetic', sampled_camera_rays_equal=True, fields=fields)), flush=True)
    finally:
        compiler._expanded_shader_source = original_expand
        generator.p = original_program


if __name__ == '__main__':
    main()
