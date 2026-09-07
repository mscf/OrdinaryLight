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
out=Path('/tmp/glass-cap-sweep');out.mkdir(exist_ok=True)
source=Path('ordinarylight/denoising/kernels.py').read_text()
for mode in ['cap4']:
 s=source
 if mode=='cap4':
  anchor='    if accepted:\n        alpha = 1.0 / osh.maximum(history_length, 1.0)'
  assert s.count(anchor)==1
  s=s.replace(anchor,'    if identity.load(pixel).r == osh.u32(0) and osh.length(motion_vector) > 1.0:\n        history_length = osh.minimum(history_length, 4.0)\n'+anchor)
 else:
  s=s.replace('constants.rejection.w','reactive_sigma')
  anchor='    motion_vector = motion_sample.xy'
  s=s.replace(anchor,anchor+'\n    reactive_sigma = constants.rejection.w\n    if identity.load(pixel).r == osh.u32(0):\n        reactive_sigma = 1.0')
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
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),direct_swapchain_storage=False,wavefront_tile_capacity=320*240)

g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);window=g.create_window(320,240,'cap sweep',None,None)
cases=[('low-ior',1.1,1.,16),('water-ior',1.33,1.,16),('glass-ior',1.52,1.,16),('high-ior',1.8,1.,16),('thin',1.52,.5,16),('thick',1.52,1.5,16),('fast',1.52,1.,8),('slow',1.52,1.,32)]
def build(ior):
 scene,glass,bars=fixture();scene.update_instance(glass,material=replace(glass.material,ior=ior));return scene,glass,bars
def set_pose(scene,glass,zscale,x):
 matrix=np.eye(4,dtype=np.float32);matrix[2,2]=zscale;matrix[0,3]=x;scene.update_instance(glass,transform=matrix)
 return ol.PerspectiveCamera(position=(0,1,-5),target=(0,1,0))
refcfg=replace(cfg,denoiser_enabled=False,progressive_accumulation=False,temporal_history=False,wavefront_restir_di=False,samples_per_pixel=64)
metrics={};references={}
try:
 for name,ior,zscale,count in cases:
  xs=list(np.linspace(-.3,.3,count))+[.3]*32
  if (ior,zscale) not in references:
   active='baseline';scene,glass,bars=build(ior);cam=set_pose(scene,glass,zscale,.3);batches=[]
   with ol.VulkanGlfwPresenter(window,config=refcfg) as p:
    for i in range(8):
     p._core.wavefront_frame_sequence=2000+i;p.present_wavefront(scene,cam,320,240);batches.append(p.capture_wavefront_hdr()[...,:3])
   references[ior,zscale]=np.asarray(batches)
  refs=references[ior,zscale];np.save(out/(name+'-reference.npy'),refs)
  metrics[name]={'ior':ior,'zscale':zscale,'moving_frames':count}
  sl=np.s_[85:155,125:195];r=signal(refs.mean(0)[sl])
  metrics[name]['reference_split_rmse']=float(np.sqrt(np.mean((signal(refs[::2].mean(0)[sl])-signal(refs[1::2].mean(0)[sl]))**2)))
  for mode in ['baseline','cap4']:
   active=mode;scene,glass,bars=build(ior);frames=[]
   with ol.VulkanGlfwPresenter(window,config=cfg) as p:
    for x in xs:
     cam=set_pose(scene,glass,zscale,float(x));p.present_wavefront(scene,cam,320,240);frames.append(p.capture_wavefront_hdr()[...,:3])
   a=np.asarray(frames);assert np.isfinite(a).all();np.save(out/(name+'-'+mode+'.npy'),a)
   metrics[name][mode]={'moving_rmse':float(np.sqrt(np.mean((signal(a[count-1][sl])-r)**2))),'settled_rmse':float(np.sqrt(np.mean((signal(a[-1][sl])-r)**2))),'stopped_std':float(np.std(signal(a[-16:,85:155,125:195]),axis=0).mean())}
  print(name,json.dumps(metrics[name]),flush=True)
  (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
finally:g.destroy_window(window);g.terminate()
