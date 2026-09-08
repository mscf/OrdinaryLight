import inline_probe
import ordinarylight.shaders.compiler as compiler
original=compiler.wavefront_material_shader_source

def source(name,*args,**kwargs):
 s=original(name,*args,**kwargs)
 if name=='wavefront_primary.comp':
  anchor='''        path_index, medium_depth, rng, cone_width, cone_spread);
    WAVE_STORE_PATH(path_index, path);'''
  assert anchor in s
  s=s.replace(anchor,anchor+'''
    if (pathBounce(path) == 3u) {
        secondary_paths[path_index].normal_pdf = vec4(continuation_direction, uintBitsToFloat(rng));
        secondary_paths[path_index].primary_throughput.xyz = continuation_origin;
        secondary_paths[path_index].primary_radiance = vec4(path.throughput.rgb, -999.0);
    }
''',1)
 elif name=='wavefront_shade_candidate.glsl':
  anchor=next(line for line in s.splitlines() if 'ShadeContinuationResult continuation = shadeBuildContinuation(' in line)
  s=s.replace(anchor,anchor+'''
    if (next_bounce == 3u) {
        secondary_paths[path_index].normal_pdf = vec4(continuation.ray.direction_tmax.xyz, uintBitsToFloat(roulette.random_state));
        secondary_paths[path_index].primary_throughput.xyz = continuation.ray.origin_tmin.xyz;
        secondary_paths[path_index].primary_radiance = vec4(continuation.path.throughput.rgb, -999.0);
    }
''',1)
 if name=='wavefront_primary.comp' and os.environ.get('REPLAY_RAY')=='1':
  anchor='    WAVE_STORE_PATH(path_index, path);\n    if (pathBounce(path) == 3u) {'
  replacement='    WAVE_STORE_PATH(path_index, path);\n    if (pathBounce(path) == 3u && path.metadata.x == 56179u) {\n        continuation_origin = vec3(0.32361587882041931, 1.9422378540039062, 0.003489519702270627);\n        continuation_direction = vec3(-0.25843822956085205, -0.70494538545608521, -0.66050100326538086);\n    }\n    if (pathBounce(path) == 3u) {'
  assert anchor in s
  s=s.replace(anchor,replacement,1)
 return s
compiler.wavefront_material_shader_source=source
import os,inspect,textwrap
import ordinarylight.targets.vulkan.core as core
if 'SAMPLE' in os.environ:
 method=textwrap.dedent(inspect.getsource(core.VulkanRayQueryCore.present_wavefront_window))
 method=method.replace('for sample_index in range(sample_count):',f'for sample_index in [{int(os.environ["SAMPLE"])}]:',1)
 namespace=dict(vars(core));exec(compile(method,'<single stream diagnostic>','exec'),namespace)
 core.VulkanRayQueryCore.present_wavefront_window=namespace['present_wavefront_window']
 original_resolve=core.VulkanWavefrontExecutor.record_path_to_hdr
 def resolve(self,command,slot,path_count,image_width,image_height,sample_index=0,sample_count=1):
  return original_resolve(self,command,slot,path_count,image_width,image_height,0,1)
 core.VulkanWavefrontExecutor.record_path_to_hdr=resolve
