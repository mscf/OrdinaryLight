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
out=Path('/tmp/final-guides-parity');out.mkdir(exist_ok=True)
original=core.files
active='baseline'
class Resources:
 def __init__(self,package): self.base=original(package)
 def joinpath(self,name):
  if active=='baseline' and name=='shaders/denoiser_relax_prepare.comp.spv':return Path(__file__).with_name('baseline-prepare.spv')
  return self.base.joinpath(name)
core.files=Resources
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),wavefront_timestamps=False,denoiser_signal_capture=True,denoiser_transmission_motion_cap=True,wavefront_tile_capacity=320*240)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(320,240,'guide parity',None,None)
try:
 for subject in ['glass','target','camera']:
  for active in ['baseline','optimized']:
   scene,glass,bars=fixture();frames=[]
   with ol.VulkanGlfwPresenter(w,config=cfg) as p:
    for x in list(np.linspace(-.3,.3,16))+[.3]*32:
     p.present_wavefront(scene,pose(scene,glass,bars,subject,float(x)),320,240)
     frames.append(p.capture_wavefront_hdr())
    np.save(out/(subject+'-'+active+'.npy'),frames)
  a=np.load(out/(subject+'-baseline.npy'));b=np.load(out/(subject+'-optimized.npy'))
  print(subject,'max',float(np.max(np.abs(a-b))),'rmse',float(np.sqrt(np.mean((a-b)**2))),flush=True)
finally:g.destroy_window(w);g.terminate()
