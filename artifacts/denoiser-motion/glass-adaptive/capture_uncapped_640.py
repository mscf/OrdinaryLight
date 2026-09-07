from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import importlib.util
import sys
import json
import numpy as np
from PIL import Image,ImageDraw
import ordinaryshade as osh
import ordinarylight as ol
from ordinarylight.targets.vulkan import core
from ordinarylight.shaders.compiler import find_glsl_compiler
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.integrations.glfw_platform import load_glfw
from tools.denoiser_motion.glass_detail import fixture,pose,signal,display
out=Path('/tmp/glass-uncapped-640');out.mkdir(exist_ok=True)
source=Path('ordinarylight/denoising/kernels.py').read_text()
original_files=core.files
active='baseline'
class Resources:
 def __init__(self,package):self.base=original_files(package)
 def joinpath(self,name):
  if active!='baseline' and name=='shaders/denoiser_relax_temporal.comp.spv':return out/(active+'.spv')
  return self.base.joinpath(name)
core.files=Resources
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),denoiser_transmission_motion_cap=False,direct_swapchain_storage=False,wavefront_tile_capacity=640*480)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);window=g.create_window(640,480,'rejection',None,None)
xs=list(np.linspace(-.3,.3,16))+[.3]*32
metrics={};sheet=Image.new('RGB',(1280,792));draw=ImageDraw.Draw(sheet)
try:
 for row,kind in enumerate(['camera','glass','target']):
  refs=np.load('/tmp/glass-adaptive-640/'+kind+'-references.npy')
  metrics[kind]={}
  for col,mode in enumerate(['baseline']):
   active=mode;scene,glass,bars=fixture();frames=[]
   with ol.VulkanGlfwPresenter(window,config=cfg) as p:
    for x in xs:
     cam=pose(scene,glass,bars,kind,float(x));p.present_wavefront(scene,cam,640,480);frames.append(p.capture_wavefront_hdr()[...,:3])
   a=np.asarray(frames);np.save(out/f'{kind}-{mode}.npy',a)
   sl=np.s_[170:310,250:390];stats={}
   for label,index,refindex in [('moving',15,1),('settled',47,1)]:
    v=signal(a[index][sl]);r=signal(refs[refindex].mean(0)[sl]);stats[label+'_rmse']=float(np.sqrt(np.mean((v-r)**2)))
   stats['stopped_std']=float(np.std(signal(a[-16:,170:310,250:390]),axis=0).mean())
   metrics[kind][mode]=stats
   sheet.paste(Image.fromarray(display(a[-1])).resize((320,240)),(col*320,row*264+24));draw.text((col*320+3,row*264+3),kind+' / '+mode+' settled',fill='white')
   print(kind,mode,json.dumps(stats),flush=True)
  sheet.paste(Image.fromarray(display(refs[1].mean(0))).resize((320,240)),(960,row*264+24));draw.text((963,row*264+3),kind+' / reference',fill='white')
finally:g.destroy_window(window);g.terminate()
sheet.save(out/'comparison.png');(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
