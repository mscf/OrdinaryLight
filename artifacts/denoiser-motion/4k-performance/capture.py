import cProfile, pstats, time, json, os
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture, pose
from collections import defaultdict
import ordinarylight.targets.vulkan.core as vc
costs=defaultdict(list)
if os.environ.get('FORCE_RECORD') == '1':
 original_update=vc.VulkanRayQueryCore._try_update_window_scene
 def force_record(self,scene):
  result=original_update(self,scene)
  if result:
   for frame in self.window_frames:frame['wavefront_command_key']=None
  return result
 vc.VulkanRayQueryCore._try_update_window_scene=force_record

for name in ['_update_device_buffers','_rebuild_scene_tlas','trace_wavefront_tile']:
 original=getattr(vc.VulkanRayQueryCore,name)
 def timed(self,*args,_name=name,_original=original,**kwargs):
  start=time.perf_counter()
  try:return _original(self,*args,**kwargs)
  finally:costs[_name].append((time.perf_counter()-start)*1000)
 setattr(vc.VulkanRayQueryCore,name,timed)
out=Path(os.environ.get('OUT','/tmp/glass-performance-before'));out.mkdir(exist_ok=True)
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),wavefront_timestamps=os.environ.get('TIMESTAMPS','1')=='1',wavefront_restir_reservoirs=int(os.environ.get('STREAMS','4')),denoiser_signal_capture=True,denoiser_transmission_motion_cap=True,direct_swapchain_storage=os.environ.get('DIRECT','0')=='1',wavefront_render_scale=float(os.environ.get('SCALE','1')),wavefront_tile_capacity=int(os.environ.get('CAPACITY','131072')))
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(3840,2160,'motion profile',None,None)
try:
 scene,glass,bars=fixture()
 with ol.VulkanGlfwPresenter(w,config=cfg) as p:
  for i in range(8):p.present_wavefront(scene,pose(scene,glass,bars,os.environ.get('SUBJECT','glass'),-.3),3840,2160)
  results={}; frames=[]
  for phase in ['moving','stationary']:
   costs.clear(); stage=[]; profiler=cProfile.Profile();times=[];profiler.enable() if os.environ.get('PROFILE','1') == '1' else None
   for i in range(int(os.environ.get('FRAMES','12'))):
    start=time.perf_counter();camera=pose(scene,glass,bars,os.environ.get('SUBJECT','glass'),-.3+.6*(i%32)/31 if phase=='moving' else .3)
    if os.environ.get('SCALE_LIGHT') == '1':
     scene.update_instance(bars[0], transform=np.diag([1.+(i%8)*.1,1.,1.,1.]))
    if os.environ.get('RECREATE') == '1' and phase == 'moving' and i in [10,60,120,240]:
     p._core.swapchain_extent = None
    p.present_wavefront(scene,camera,3840,2160)
    times.append((time.perf_counter()-start)*1000)
    stage.append({k:v for k,v in p._core.last_timings.items() if k.startswith('wavefront_') or k in ['gpu_frame_ms','fence_wait_ms']})
   profiler.disable();profiler.dump_stats(str(out/(phase+'.prof')))
   if os.environ.get('PROFILE','1') == '1':
    with (out/(phase+'.txt')).open('w') as f:pstats.Stats(profiler,stream=f).sort_stats('cumulative').print_stats(45)
   frames.append(p.capture_wavefront_hdr())
   results[phase]={'median_ms':float(np.median(times[2:])),'timings':times,'stages':stage,'cost_ms':{k:sum(v)/len(times) for k,v in costs.items()}}
  np.save(out/'frames.npy',frames);(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
finally:g.destroy_window(w);g.terminate()
