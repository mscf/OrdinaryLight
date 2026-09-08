"""Paired 4K reconstruction timestamps; concurrent GPU load is uncontrolled."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture,pose
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE)
w=g.create_window(3840,2160,'Upscale profile',None,None);result=[]
try:
 for mode in ('bilinear','clamped-cubic','clamped-cubic','bilinear'):
  scene,glass,bars=fixture();samples=[]
  config=_gi_config(SimpleNamespace(id='glass-detail',renderer={}),present=True,render_scale=.5,upscale_filter=mode,restir_reservoirs=1)
  with ol.VulkanGlfwPresenter(w,config=config) as p:
   for i in range(24):
    p.present_wavefront(scene,pose(scene,glass,bars,'camera',-.3+.6*(i%12)/11),3840,2160)
    if i>=8:
     t=p.last_timings;samples.append({'gpu_ms':t['gpu_frame_ms'],'reconstruct_ms':t['wavefront_stage_ms']['reconstruct']})
  result.append({'filter':mode,'samples':samples,'median_reconstruct_ms':float(np.median([s['reconstruct_ms'] for s in samples])),'median_gpu_ms':float(np.median([s['gpu_ms'] for s in samples]))})
  print(result[-1],flush=True)
 Path('/tmp/cubic-profile.json').write_text(json.dumps(result,indent=2))
finally:g.destroy_window(w);g.terminate()
