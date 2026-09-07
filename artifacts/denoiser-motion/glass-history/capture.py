from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import json
import numpy as np
import ordinarylight as ol
from tools.denoiser_motion.glass_detail import fixture,pose,signal,display
from tools.denoiser_motion.live_edges import read_guides
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from PIL import Image
out=Path('/tmp/glass-history');out.mkdir(exist_ok=True)
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),denoiser_signal_capture=True,direct_swapchain_storage=False,wavefront_tile_capacity=320*240)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(320,240,'history',None,None)
try:
 for mode in ['moved','reset','fresh']:
  scene,glass,bars=fixture();xs=[.3]*48 if mode=='fresh' else list(np.linspace(-.3,.3,16))+[.3]*32
  with ol.VulkanGlfwPresenter(w,config=cfg) as p:
   for i,x in enumerate(xs):
    camera=pose(scene,glass,bars,'glass',float(x))
    if mode=='reset' and i==16:
     for frame in p._core.window_frames:
      frame['wavefront_relax_history_valid']=False
      frame['wavefront_command_key']=None
    p.present_wavefront(scene,camera,320,240)
    if i in [15,16,47]:
     a=p.capture_wavefront_hdr()[...,:3];np.save(out/f'{mode}-{i}.npy',a)
     np.savez_compressed(out/f'{mode}-{i}-guides.npz',**read_guides(p._core))
   Image.fromarray(display(a)).save(out/f'{mode}.png')
  print(mode,flush=True)
finally:g.destroy_window(w);g.terminate()
ref=np.load('/tmp/glass-detail/glass-reference-batches.npy')[1].mean(0);sl=np.s_[85:155,125:195]
stats={}
for mode in ['moved','reset','fresh']:
 a=np.load(out/f'{mode}-47.npy');d=np.load(out/f'{mode}-47-guides.npz')
 stats[mode]={'rmse':float(np.sqrt(np.mean((signal(a[sl])-signal(ref[sl]))**2))),'motion_abs_mean':float(np.mean(abs(d['motion'][sl][...,:2]))),'history_mean':float(np.mean(d['specular_history'][sl])),'zero_depth':float(np.mean(d['view_z'][sl]==0))}
print(json.dumps(stats,indent=2));(out/'metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
