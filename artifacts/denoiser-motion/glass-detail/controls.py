from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import ordinarylight as ol
from tools.denoiser_motion.glass_detail import fixture,pose,display
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from PIL import Image
out=Path('/tmp/glass-detail')
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),direct_swapchain_storage=False,wavefront_tile_capacity=320*240)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(320,240,'controls',None,None)
try:
 s,glass,bars=fixture();cam=pose(s,glass,bars,'glass',.3)
 with ol.VulkanGlfwPresenter(w,config=cfg) as p:
  for i in range(48):p.present_wavefront(s,cam,320,240)
  a=p.capture_wavefront_hdr()[...,:3];np.save(out/'glass-fresh-denoised.npy',a);Image.fromarray(display(a)).save(out/'glass-fresh-denoised.png')
 raw=replace(cfg,denoiser_enabled=False,progressive_accumulation=False,temporal_history=False,wavefront_restir_di=False,samples_per_pixel=64)
 s,glass,bars=fixture()
 with ol.VulkanGlfwPresenter(w,config=raw) as p:
  for x in np.linspace(-.3,.3,16):p.present_wavefront(s,pose(s,glass,bars,'glass',float(x)),320,240)
  batches=[]
  for i in range(8):
   p._core.wavefront_frame_sequence=1120+i;p.present_wavefront(s,cam,320,240);batches.append(p.capture_wavefront_hdr()[...,:3])
  np.save(out/'glass-moved-raw.npy',batches);Image.fromarray(display(np.mean(batches,axis=0))).save(out/'glass-moved-raw.png')
finally:g.destroy_window(w);g.terminate()
