from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from PIL import Image, ImageDraw
import ordinarylight as ol
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.showcases.rooms import build_planar_mirror_guides
from tools.denoiser_motion.live_edges import read_guides
out=Path('/tmp/glass-comparison');out.mkdir(exist_ok=True)
w,h=320,240
cfg=_gi_config(SimpleNamespace(renderer={},id='planar-mirror-guides'),capture=True)
cfg=replace(cfg,denoiser_signal_capture=True,direct_swapchain_storage=False,wavefront_tile_capacity=w*h)
glfw=load_glfw();assert glfw.init();glfw.window_hint(glfw.CLIENT_API,glfw.NO_API);glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
window=glfw.create_window(w,h,'Glass comparison',None,None)
xs=list(np.linspace(-.3,.3,16))+[.3]*32
frames={}
try:
 for enabled in (False,True):
  images=[]
  with ol.VulkanGlfwPresenter(window,config=replace(cfg,denoiser_planar_mirror_guides=enabled)) as p:
   scene=build_planar_mirror_guides()
   for i,x in enumerate(xs):
    cam=ol.PerspectiveCamera(position=(float(x),2,-7),target=(float(x),2,0))
    p.present_wavefront(scene,cam,w,h);images.append(p.capture_wavefront_hdr()[...,:3])
   np.savez_compressed(out/f'prepared-{enabled}.npz',**read_guides(p._core))
   print('Captured guides',enabled,flush=True)
  frames[str(enabled)]=np.asarray(images);np.save(out/f'guides-{enabled}.npy',images)
 refcfg=replace(cfg,denoiser_enabled=False,progressive_accumulation=False,temporal_history=False,wavefront_restir_di=False,samples_per_pixel=64)
 refs=[]
 with ol.VulkanGlfwPresenter(window,config=refcfg) as p:
  scene=build_planar_mirror_guides();cam=ol.PerspectiveCamera(position=(.3,2,-7),target=(.3,2,0))
  for i in range(8):
   p._core.wavefront_frame_sequence=1000+i
   p.present_wavefront(scene,cam,w,h);refs.append(p.capture_wavefront_hdr()[...,:3]);print('Reference batch',i+1,flush=True)
 np.save(out/'reference-batches.npy',refs)
finally:
 glfw.destroy_window(window);glfw.terminate()
def display(a):
 a=np.maximum(a,0);return (np.clip(a/(1+a),0,1)**(1/2.2)*255).astype('uint8')
canvas=Image.new('RGB',(w*3,(h+24)*2));draw=ImageDraw.Draw(canvas)
for row,key in enumerate(['False','True']):
 for col,(idx,label) in enumerate([(15,'last moving frame'),(16,'first stopped frame'),(47,'32 stopped frames')]):
  x,y=col*w,row*(h+24);canvas.paste(Image.fromarray(display(frames[key][idx])),(x,y+24));draw.text((x+4,y+4),f'Guides {key}: {label}',fill='white')
canvas.save(out/'comparison.png');Image.fromarray(display(np.mean(refs,axis=0))).save(out/'reference.png')
print(out,flush=True)
