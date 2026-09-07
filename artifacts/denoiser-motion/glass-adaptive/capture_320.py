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
out=Path('/tmp/glass-adaptive');out.mkdir(exist_ok=True)
source=Path('ordinarylight/denoising/kernels.py').read_text()
for mode in ['supported','consensus']:
 s=source.replace('constants.rejection.w','reactive_sigma')
 s=s.replace('    motion_vector = motion_sample.xy','    motion_vector = motion_sample.xy\n    reactive_sigma = constants.rejection.w\n    if motion_sample.w > 0.5:\n        reactive_sigma = 1.0')
 anchor='                accepted = (\n                    osh.absolute(history_luma - mean_luma) <= reactive_limit\n                )'
 if mode=='supported':
  addition='\n                if motion_sample.w > 0.5:\n                    current_luma = osh.dot(current.rgb, osh.vec3(0.2126, 0.7152, 0.0722))\n                    support_limit = osh.maximum(deviation_luma * 0.5, osh.absolute(mean_luma) * 0.05 + 0.005)\n                    accepted = accepted or osh.absolute(current_luma - mean_luma) > support_limit\n'
 else:
  s=s.replace('            neighborhood_count = 0.0','            neighborhood_count = 0.0\n            brighter = 0.0\n            darker = 0.0\n            old_luma = osh.dot(history.rgb, osh.vec3(0.2126, 0.7152, 0.0722))\n            evidence_limit = osh.absolute(old_luma) * 0.1 + 0.01')
  s=s.replace('                    neighborhood_sum = neighborhood_sum + neighbor','                    neighbor_luma = osh.dot(neighbor, osh.vec3(0.2126, 0.7152, 0.0722))\n                    if neighbor_luma > old_luma + evidence_limit:\n                        brighter = brighter + 1.0\n                    if neighbor_luma < old_luma - evidence_limit:\n                        darker = darker + 1.0\n                    neighborhood_sum = neighborhood_sum + neighbor')
  addition='\n                if motion_sample.w > 0.5:\n                    support = darker\n                    if mean_luma > history_luma:\n                        support = brighter\n                    accepted = accepted or support < osh.maximum(3.0, neighborhood_count * 0.75)\n'
 assert s.count(anchor)==1
 s=s.replace(anchor,anchor+addition)
 p=out/(mode+'.py');p.write_text(s)
 spec=importlib.util.spec_from_file_location('glass_'+mode,p);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 binary=osh.compile(module.relax_temporal,target='spirv',validate=True,spirv_compiler=find_glsl_compiler())
 (out/(mode+'.spv')).write_bytes(bytes(binary.binary))
original_files=core.files
active='baseline'
class Resources:
 def __init__(self,package):self.base=original_files(package)
 def joinpath(self,name):
  if active!='baseline' and name=='shaders/denoiser_relax_temporal.comp.spv':return out/(active+'.spv')
  return self.base.joinpath(name)
core.files=Resources
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),denoiser_transmission_motion_cap=True,direct_swapchain_storage=False,wavefront_tile_capacity=320*240)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);window=g.create_window(320,240,'rejection',None,None)
xs=list(np.linspace(-.3,.3,16))+[.3]*32
metrics={};sheet=Image.new('RGB',(1280,792));draw=ImageDraw.Draw(sheet)
try:
 for row,kind in enumerate(['camera','glass','target']):
  refs=np.load('/tmp/glass-detail-fixed/'+kind+'-reference-batches.npy')
  metrics[kind]={}
  for col,mode in enumerate(['baseline','supported','consensus']):
   active=mode;scene,glass,bars=fixture();frames=[]
   with ol.VulkanGlfwPresenter(window,config=cfg) as p:
    for x in xs:
     cam=pose(scene,glass,bars,kind,float(x));p.present_wavefront(scene,cam,320,240);frames.append(p.capture_wavefront_hdr()[...,:3])
   a=np.asarray(frames);np.save(out/f'{kind}-{mode}.npy',a)
   sl=np.s_[85:155,125:195];stats={}
   for label,index,refindex in [('moving',15,1),('settled',47,1)]:
    v=signal(a[index][sl]);r=signal(refs[refindex].mean(0)[sl]);stats[label+'_rmse']=float(np.sqrt(np.mean((v-r)**2)))
   stats['stopped_std']=float(np.std(signal(a[-16:,85:155,125:195]),axis=0).mean())
   metrics[kind][mode]=stats
   sheet.paste(Image.fromarray(display(a[-1])),(col*320,row*264+24));draw.text((col*320+3,row*264+3),kind+' / '+mode+' settled',fill='white')
   print(kind,mode,json.dumps(stats),flush=True)
  sheet.paste(Image.fromarray(display(refs[1].mean(0))),(960,row*264+24));draw.text((963,row*264+3),kind+' / reference',fill='white')
finally:g.destroy_window(window);g.terminate()
sheet.save(out/'comparison.png');(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
