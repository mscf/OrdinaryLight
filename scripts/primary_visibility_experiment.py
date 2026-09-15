"""Headless integration harness for the public primary capture/replay contract.

Only this diagnostic replaces the native recorder. Production shader variants,
VulkanKernel geometry descriptors and primary_operation are used unchanged.
Full-image storage is persistent. Capture/replay is adjacent *per tile* here;
this validates transport parity. full_frame=True captures every tile before
replaying the first tile, with an optional GPU operation between the phases.
The native production scheduler is not changed by this diagnostic harness.
"""
import struct
from unittest.mock import patch


def install(stack, enabled, *, full_frame=False, after_capture=None, replay_options=None, replay_bindings=None, measure=None, replay_workgroup=lambda:(8,8), omit_primary_analytic=lambda:False,single_capture=lambda:False,visibility_format=lambda:"full",conservative_capture=lambda:False,compact_continuations=lambda:False):
    from ordinarylight.shaders import compiler
    from ordinarylight.targets.vulkan import primary_graph
    from ordinarylight.runtime import VulkanKernel
    from ordinarylight.wavefront import primary_visibility_byte_size
    from ordinarylight.runtime.primary import primary_operation
    from ordinarylight.pipeline.graph import VulkanGraph
    from ordinarylight.pipeline.vulkan import VulkanResource

    if after_capture is not None and not full_frame:
        raise ValueError("A pre-lighting consumer requires full-frame capture")
    original_compile = compiler.compile_wavefront_material_shader
    original_record = primary_graph.record_primary
    variants = {}
    resources = {}
    graphs = {}
    pipeline_variants = {}
    continuations = {}

    def compile_shader(name, programs, **kwargs):
        result = original_compile(name, programs, **kwargs)
        if enabled() and name == 'wavefront_primary.comp':
            variants.clear()
            variants['capture'] = original_compile(name, programs, **kwargs, primary_visibility='capture',primary_visibility_format=visibility_format())
            variants['replay'] = original_compile(name, programs, **kwargs, primary_visibility='replay',primary_visibility_format=visibility_format(), primary_workgroup=replay_workgroup(), primary_continuation='classify' if compact_continuations() else 'fused', **(replay_options or {}))
            if compact_continuations():
                variants['resume'] = original_compile(name,programs,**kwargs,primary_visibility='replay',primary_visibility_format=visibility_format(),primary_workgroup=(64,1),primary_continuation='resume',**(replay_options or {}))
        return result

    def record(executor, command, pipeline, slot, constants, groups):
        if not enabled():
            return original_record(executor, command, pipeline, slot, constants, groups)
        width, height, _, _, _, _, samples, sample = struct.unpack_from('8I', constants)
        generation = executor.core.window_frames[slot]['wavefront_primary_hit_buffer'].buffer
        key = executor, width, height, max(samples, sample + 1), pipeline, generation, executor.core.swapchain_generation
        pipeline_key = executor, pipeline
        if pipeline_key not in pipeline_variants:
            pipeline_variants[pipeline_key] = dict(variants), replay_workgroup(), visibility_format()
        if key not in resources:
            allocation = stack.enter_context(executor.core.runtime.buffer(primary_visibility_byte_size(width,height,samples=key[3],format=pipeline_variants[pipeline_key][2]), memory='device'))
            resources[key] = allocation, {}, *pipeline_variants[pipeline_key]
        allocation, kernels, binaries, replay_shape, cache_format = resources[key]
        if slot not in kernels:
            native = primary_graph.NativePrimaryKernel(executor, pipeline, slot)
            bindings = {**native.bindings, 33: VulkanResource.buffer(allocation)}
            extra = replay_bindings(executor,allocation,slot) if replay_bindings else {}
            if 'resume' in binaries:
                from compact_continuation_experiment import CompactContinuation
                tw,th=struct.unpack_from('<2I',constants,16)
                compact=CompactContinuation(stack,native.runtime,tw*th)
                continuations[key,slot]=compact
                extra={**extra,**compact.bindings}
            kernels[slot] = tuple(stack.enter_context(VulkanKernel(
                native.runtime, binaries[mode], {**bindings, **(extra if mode != "capture" else {})}, push_constant_size=176,
                sampled_image_arrays=native.sampled_image_arrays,
                sampled_image_layouts=native.sampled_image_layouts,
                material_resources=native.material_resources,
                geometry_resources=native.geometry_resources,
            )) for mode in (('capture','replay','resume') if 'resume' in binaries else ('capture','replay')))
        tw,th=struct.unpack_from('<2I',constants,16)
        if tuple(groups)!=((tw+7)//8,(th+7)//8,1):
            raise ValueError("Capture diagnostic requires 8x8 primary workgroups")
        cache_key = key, slot, bytes(constants), tuple(groups)
        if cache_key not in graphs:
            graph = VulkanGraph()
            previous = ()
            prefix = struct.unpack_from('<8I', constants)
            if not full_frame or prefix[2:4] == (0, 0):
                tiles = [(prefix[2], prefix[3], prefix[4], prefix[5])]
                if full_frame:
                    tiles = [(x,y,min(prefix[4],width-x),min(prefix[5],height-y))
                             for y in range(0,height,prefix[5]) for x in range(0,width,prefix[4])]
                if full_frame and single_capture():
                    tiles=[(0,0,width,height)]
                for index, (x,y,tw,th) in enumerate(tiles):
                    tile = struct.pack('<8I',width,height,x,y,tw,th,samples,sample) + constants[32:]
                    name = f'capture_{index}'
                    operation=primary_operation(kernels[slot][0],tile,
                        workgroups=((tw+7)//8,(th+7)//8,1),visibility="capture",visibility_format=cache_format)
                    if conservative_capture():
                        # Reproduce the old capture resource declarations for A/B
                        # timing; bind and dispatch the SAME capture kernel.
                        from dataclasses import replace
                        import vulkan as vk
                        old=primary_operation(kernels[slot][0],tile,
                            workgroups=((tw+7)//8,(th+7)//8,1),visibility="replay",visibility_format=cache_format)
                        cache_handle=allocation.buffer
                        operation.passes=tuple(replace(p,uses=tuple(
                            replace(u,access=vk.VK_ACCESS_SHADER_WRITE_BIT)
                            if u.resource.kind=="buffer" and u.resource.handle==cache_handle
                            else u for u in p.uses)) for p in old.passes)
                    if measure:operation=measure(executor,allocation,name,operation)
                    graph.add(name,operation,after=previous)
                    previous = (name,)
                if after_capture is not None:
                    graph.add('visible_face_plan', after_capture(executor, allocation, sample), after=previous)
                    previous = ('visible_face_plan',)
            replay_constants=constants
            if omit_primary_analytic():
                replay_constants=bytearray(constants)
                # PrimarySettings: max_bounces at byte 32, analytic count at 36.
                struct.pack_into('<I',replay_constants,36,0)
            if 'resume' in binaries:
                compact=continuations[key,slot]
                graph.add('reset_continuations',compact.reset(),after=previous)
                previous=('reset_continuations',)
            rx,ry=replay_shape
            replay_groups=((prefix[4]+rx-1)//rx,(prefix[5]+ry-1)//ry,1)
            operation=primary_operation(kernels[slot][1],replay_constants,workgroups=replay_groups,visibility="replay",visibility_format=cache_format)
            if measure:operation=measure(executor,allocation,f'replay_{prefix[2]}_{prefix[3]}',operation)
            graph.add('replay',operation,after=previous)
            if 'resume' in binaries:
                graph.add('prepare_continuations',compact.prepare(),after=('replay',))
                operation=primary_operation(kernels[slot][2],replay_constants,workgroups=(1,1,1),visibility='replay',visibility_format=cache_format)
                operation=compact.indirect(operation,kernels[slot][2],replay_constants)
                if measure:operation=measure(executor,allocation,'resume_continuations',operation)
                graph.add('resume_continuations',operation,after=('prepare_continuations',))
            graphs[cache_key] = graph.compile()
        graphs[cache_key].prepare_recording(executor.core.runtime).record(command)

    stack.enter_context(patch.object(compiler, 'compile_wavefront_material_shader', compile_shader))
    stack.enter_context(patch.object(primary_graph, 'record_primary', record))
