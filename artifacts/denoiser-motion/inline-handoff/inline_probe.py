"""Diagnostic-only custom hybrid execution; no production source changes."""
import inspect, textwrap, os
import ordinarylight.targets.vulkan.core as core
import ordinarylight.shaders.compiler as compiler
if os.environ.get('INLINE') == '1':
 original=compiler.wavefront_material_shader_source
 def source(name,*args,**kwargs):
  result=original(name,*args,**kwargs)
  if name=='wavefront_primary.comp':
   assert '#define WAVE_HYBRID 0' in result
   result=result.replace('#define WAVE_HYBRID 0','#define WAVE_HYBRID 1',1)
  if name=='wavefront_primary.comp':
   from pathlib import Path
   Path('/tmp/inline-primary.glsl').write_text(result)
  return result
 compiler.wavefront_material_shader_source=source
 method=textwrap.dedent(inspect.getsource(core.VulkanWavefrontExecutor.dispatch))
 method=method.replace('if self.custom_primary_pipeline is not None:', 'if self.custom_primary_pipeline is not None and strategy != "hybrid":',1)
 method=method.replace('"hybrid": self.hybrid_pipeline,','"hybrid": self.custom_primary_pipeline or self.hybrid_pipeline,',1)
 namespace=dict(vars(core));exec(compile(method,'<custom hybrid diagnostic>','exec'),namespace)
 core.VulkanWavefrontExecutor.dispatch=namespace['dispatch']
if os.environ.get('PRECISE') == '1':
 import re
 before_precision=compiler.wavefront_material_shader_source
 def precise_source(name,*args,**kwargs):
  result=before_precision(name,*args,**kwargs)
  return re.sub(r'(?m)^(\s+)(vec3)(\s+(?:hit|normal|shading_normal|geometric_normal|weights)\s*=)',r'\1precise \2\3',result)
 compiler.wavefront_material_shader_source=precise_source
