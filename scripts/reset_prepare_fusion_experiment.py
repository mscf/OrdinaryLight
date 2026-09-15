"""Diagnostic only: prepare temporal reset outputs directly; retain raw signals.

All shader changes are AST transformations of the typed OrdinaryShade source.
The dynamic history policy remains GPU-owned and valid-history filtering stays
unchanged. Native/private interception is confined to this comparison harness.
"""
import ast
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


def install(stack, directory, selected):
    import ordinaryshade as osh
    import vulkan as vk
    from ordinarylight.denoising import kernels
    from ordinarylight.runtime import compile_compute, relax_prepare, relax_temporal
    from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse
    from ordinarylight.targets.vulkan import relax_prepare_graph
    from ordinarylight.targets.vulkan.denoiser_graph import _Binding
    from primary_hit_layout_experiment import load_module

    source = Path(kernels.__file__).read_text()
    tree = ast.parse(source)
    prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_relax_signals')
    policy = ast.parse('''@osh.structure
class FusionPolicy:
    extent_history: osh.vec4
    rejection: osh.vec4
''').body[0]
    tree.body.insert(tree.body.index(prepare)-1, policy)
    declarations = ast.parse('''def extra(
    final_diffuse: osh.storage_image("rgba16f", access="write", binding=14),
    final_specular: osh.storage_image("rgba16f", access="write", binding=15),
    diffuse_length: osh.storage_image("r32f", access="write", binding=16),
    specular_length: osh.storage_image("r32f", access="write", binding=17),
    fusion_policy: osh.uniform_buffer(FusionPolicy, binding=18),
): pass
''').body[0]
    prepare.args.args.extend(declarations.args.args)
    reset = 'not (fusion_policy.extent_history.w > 0.5) and constants.samples.x + osh.u32(1) == osh.maximum(constants.samples.y, osh.u32(1))'
    class Stores(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'store' and isinstance(call.func.value, ast.Name):
                target = {'diffuse_output':'final_diffuse','specular_output':'final_specular'}.get(call.func.value.id)
                if target:
                    extra = ast.parse(f'if {reset}:\n    {target}.store({ast.unparse(call.args[0])}, {ast.unparse(call.args[1])})').body[0]
                    return [node, extra]
            return node
    Stores().visit(prepare)
    index = next(i for i,n in enumerate(prepare.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='secondary' for t in n.targets))
    prepare.body[index:index] = ast.parse(f'if {reset}:\n    diffuse_length.store(pixel, osh.vec4(1.0))\n    specular_length.store(pixel, osh.vec4(1.0))').body
    temporal = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='relax_temporal')
    reset_branch = next(n for n in temporal.body if isinstance(n,ast.If) and ast.unparse(n.test)=='not constants.extent_history.w > 0.5')
    reset_branch.body = [ast.Return()]
    module = load_module('reset_prepare_fusion', ast.unparse(ast.fix_missing_locations(tree)), directory)
    prepared = compile_compute(osh.compile(module.prepare_relax_signals,helpers=(module.prepare_decode_normal,module.prepare_unpack_normal,module.prepare_previous_pixel,module.prepare_custom_surface_history)).source)
    filtered = compile_compute(osh.compile(module.relax_temporal).source)
    old_prepare_kernel, old_temporal_kernel = relax_prepare.VulkanKernel, relax_temporal.VulkanKernel
    old_record = relax_prepare_graph.record_relax_prepare
    old_operation = relax_prepare.relax_prepare_operation
    context = []
    def record(executor, command, slot, *args, **kwargs):
        context.append((executor,slot))
        try: return old_record(executor,command,slot,*args,**kwargs)
        finally: context.pop()
    def prepare_kernel(runtime, binary, bindings, **kwargs):
        if not selected(): return old_prepare_kernel(runtime,binary,bindings,**kwargs)
        executor,slot = context[-1]
        frame = executor.core.window_frames[slot]
        width,height=frame['wavefront_allocation_extent']
        extras={}
        for binding,name,fmt in ((14,'temporal_diffuse',vk.VK_FORMAT_R16G16B16A16_SFLOAT),(15,'temporal_specular',vk.VK_FORMAT_R16G16B16A16_SFLOAT),(16,'diffuse_history',vk.VK_FORMAT_R32_SFLOAT),(17,'specular_history',vk.VK_FORMAT_R32_SFLOAT)):
            prefix='wavefront_relax_'+name
            extras[binding]=VulkanResource.image(_Binding(runtime=runtime,image=frame[prefix+'_image'],view=frame[prefix+'_view'],width=width,height=height,format=fmt,layout=vk.VK_IMAGE_LAYOUT_GENERAL,usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,require_open=runtime.require_open))
        buffer=executor.relax_temporal_constant_buffers[slot]
        extras[18]=VulkanResource.uniform_buffer(_Binding(runtime=runtime,buffer=buffer.buffer,byte_size=32,usage=vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,require_open=runtime.require_open))
        return old_prepare_kernel(runtime,prepared,{**bindings,**extras},**kwargs)
    def temporal_kernel(runtime,binary,bindings,**kwargs):
        return old_temporal_kernel(runtime,filtered if selected() else binary,bindings,**kwargs)
    def operation(kernel,**kwargs):
        op=old_operation(kernel,**kwargs)
        if 18 in kernel.bindings:
            def usage(u):
                for b,r in kernel.bindings.items():
                    if b>=14 and r is u.resource:
                        return replace(u,access=vk.VK_ACCESS_UNIFORM_READ_BIT if b==18 else vk.VK_ACCESS_SHADER_WRITE_BIT)
                return u
            op.passes=tuple(replace(p,uses=tuple(usage(u) for u in p.uses)) for p in op.passes)
        return op
    stack.enter_context(patch.object(relax_prepare_graph,'record_relax_prepare',record))
    stack.enter_context(patch.object(relax_prepare,'VulkanKernel',prepare_kernel))
    stack.enter_context(patch.object(relax_temporal,'VulkanKernel',temporal_kernel))
    stack.enter_context(patch.object(relax_prepare,'relax_prepare_operation',operation))
