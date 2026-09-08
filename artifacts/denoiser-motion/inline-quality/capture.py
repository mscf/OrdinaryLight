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
import os
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),wavefront_custom_inline=True,wavefront_execution_strategy='hybrid',wavefront_timestamps=False,denoiser_signal_capture=True,denoiser_transmission_motion_cap=True,wavefront_tile_capacity=320*240)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(320,240,'guide parity',None,None)
try:
 for subject in ['glass','target','camera']:
  for active in ['result']:
   scene,glass,bars=fixture();frames=[]
   with ol.VulkanGlfwPresenter(w,config=cfg) as p:
    for x in list(np.linspace(-.3,.3,16))+[.3]*32:
     p.present_wavefront(scene,pose(scene,glass,bars,subject,float(x)),320,240)
     frames.append(p.capture_wavefront_hdr())
    assert p._core.last_timings['wavefront_execution_strategy']=='hybrid'
    np.save(out/(subject+'-'+active+'.npy'),frames)
    np.savez_compressed(out/(subject+'-guides.npz'),**read_guides(p._core))
finally:g.destroy_window(w);g.terminate()
