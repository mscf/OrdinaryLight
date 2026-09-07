from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture,pose
from tools.denoiser_motion.live_edges import read_guides
out=Path('/tmp/glass-cap-native');out.mkdir(exist_ok=True)
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),denoiser_signal_capture=True,direct_swapchain_storage=False,wavefront_tile_capacity=320*240)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(320,240,'native cap',None,None)
try:
 for mode in ['off','on','reordered']:
  scene,glass,bars=fixture()
  if mode=='reordered':
   original=scene;scene=ol.Scene()
   scene.add_mesh(np.array([[100,0,0],[101,0,0],[100,1,0]],np.float32),np.array([[0,1,2]],np.uint32),ol.Material())
   copies=[scene.add_mesh(m.vertices,m.indices,m.material,name=m.name) for m in original.meshes]
   glass= copies[0];bars=copies[1:]
  frames=[]
  with ol.VulkanGlfwPresenter(w,config=replace(cfg,denoiser_transmission_motion_cap=mode!='off')) as p:
   for x in list(np.linspace(-.3,.3,16))+[.3]*32:
    p.present_wavefront(scene,pose(scene,glass,bars,'glass',float(x)),320,240);frames.append(p.capture_wavefront_hdr()[...,:3])
   np.save(out/(mode+'.npy'),frames);guides=read_guides(p._core);np.savez_compressed(out/(mode+'-guides.npz'),**guides)
   flagged=guides['motion'][...,3]>.5
   print(mode,'flagged',int(flagged.sum()),'identities',np.unique(guides['identity'][...,0][flagged]).tolist(),flush=True)
finally:g.destroy_window(w);g.terminate()
for mode,previous in [('off','baseline'),('on','cap4')]:
 a=np.load(out/(mode+'.npy'));b=np.load('/tmp/glass-rejection/glass-'+previous+'.npy');print(mode,'maxdiff',float(np.max(abs(a-b))),flush=True)
