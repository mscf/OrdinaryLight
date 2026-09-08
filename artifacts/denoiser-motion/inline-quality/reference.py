from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture,pose
out=Path('/tmp/inline-quality-reference');out.mkdir(exist_ok=True)
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),wavefront_timestamps=False,denoiser_enabled=False,progressive_accumulation=False,temporal_history=False,wavefront_restir_di=False,samples_per_pixel=64,wavefront_tile_capacity=320*240)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(320,240,'quality reference',None,None)
try:
 for kind in ['glass','target','camera']:
  scene,glass,bars=fixture();refs=[]
  with ol.VulkanGlfwPresenter(w,config=cfg) as p:
   for index in [7,15]:
    camera=pose(scene,glass,bars,kind,float(np.linspace(-.3,.3,16)[index]));batches=[]
    for batch in range(8):
     p._core.wavefront_frame_sequence=1000+index*8+batch
     p.present_wavefront(scene,camera,320,240);batches.append(p.capture_wavefront_hdr()[...,:3])
    refs.append(batches)
  np.save(out/(kind+'.npy'),refs);print(kind,flush=True)
finally:g.destroy_window(w);g.terminate()
