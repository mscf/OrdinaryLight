import os
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture,pose
from tools.denoiser_motion.live_edges import read_guides
import ordinarylight.targets.vulkan.core as core
out=Path(os.environ['OUT']);out.mkdir(exist_ok=True)
import inline_probe, os
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),wavefront_execution_strategy='hybrid' if os.environ.get('INLINE')=='1' else 'wavefront',wavefront_timestamps=False,max_bounces=8,denoiser_enabled=False,denoiser_signal_capture=True,denoiser_transmission_motion_cap=True,wavefront_tile_capacity=320*240)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(320,240,'guide parity',None,None)
try:
 for depth in [2,3,4,8]:
  subject='target'
  for active in ['result']:
   scene,glass,bars=fixture();frames=[]
   with ol.VulkanGlfwPresenter(w,config=replace(cfg,max_bounces=depth)) as p:
    for x in list(np.linspace(-.3,.3,16))+[.3]*32:
     p.present_wavefront(scene,pose(scene,glass,bars,subject,float(x)),320,240)
     frames.append(p.capture_wavefront_hdr())
    np.save(out/(str(depth)+'.npy'),frames)

finally:g.destroy_window(w);g.terminate()
