import cProfile, pstats, time, json, os
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture, pose
if os.environ.get('BASELINE') == '1':
 import ast, subprocess
 import ordinarylight.scene._core as sc
 import ordinarylight.targets.vulkan.core as vc
 import ordinarylight.targets.vulkan.scene as vs
 for module, cls, names in [(sc,sc.Scene,['emissive_light_weight','emissive_triangle_count']), (vc,vc.VulkanRayQueryCore,['_try_update_window_scene']), (vs,vs.VulkanSceneResources,['_update_content_signatures'])]:
  relative=Path(module.__file__).relative_to(Path.cwd())
  tree=ast.parse(subprocess.check_output(['git','show','23c4a7520d8294e18f4c93920c85297aee54e558:'+str(relative)],text=True))
  node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls.__name__)
  for fn in node.body:
   if isinstance(fn,ast.FunctionDef) and fn.name in names:
    namespace=dict(vars(module));exec(compile(ast.Module(body=[fn],type_ignores=[]),str(relative),'exec'),namespace);setattr(cls,fn.name,namespace[fn.name])
out=Path(os.environ.get('OUT','/tmp/glass-performance-before'));out.mkdir(exist_ok=True)
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),denoiser_signal_capture=True,denoiser_transmission_motion_cap=True,direct_swapchain_storage=False,wavefront_tile_capacity=640*480)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(640,480,'motion profile',None,None)
try:
 scene,glass,bars=fixture()
 with ol.VulkanGlfwPresenter(w,config=cfg) as p:
  for i in range(8):p.present_wavefront(scene,pose(scene,glass,bars,'glass',-.3),640,480)
  results={}; frames=[]
  for phase in ['moving','stationary']:
   profiler=cProfile.Profile();times=[];profiler.enable() if os.environ.get('PROFILE','1') == '1' else None
   for i in range(32):
    start=time.perf_counter();camera=pose(scene,glass,bars,'glass',-.3+.6*i/31 if phase=='moving' else .3)
    p.present_wavefront(scene,camera,640,480)
    times.append((time.perf_counter()-start)*1000)
   profiler.disable();profiler.dump_stats(str(out/(phase+'.prof')))
   if os.environ.get('PROFILE','1') == '1':
    with (out/(phase+'.txt')).open('w') as f:pstats.Stats(profiler,stream=f).sort_stats('cumulative').print_stats(45)
   frames.append(p.capture_wavefront_hdr())
   results[phase]={'median_ms':float(np.median(times[2:])),'timings':times}
  np.save(out/'frames.npy',frames);(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
finally:g.destroy_window(w);g.terminate()
