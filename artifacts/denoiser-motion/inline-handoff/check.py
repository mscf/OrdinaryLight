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
import inline_handoff, os
cfg=replace(_gi_config(SimpleNamespace(renderer={},id='glass-detail'),capture=True),wavefront_execution_strategy='hybrid' if os.environ.get('INLINE')=='1' else 'wavefront',wavefront_timestamps=False,max_bounces=int(os.environ['BOUNCES']),denoiser_enabled=False,denoiser_signal_capture=True,denoiser_transmission_motion_cap=True,wavefront_tile_capacity=320*240)
g=load_glfw();assert g.init();g.window_hint(g.CLIENT_API,g.NO_API);g.window_hint(g.VISIBLE,g.FALSE);w=g.create_window(320,240,'guide parity',None,None)
try:
 for subject in ['target']:
  for active in ['result']:
   scene,glass,bars=fixture();frames=[]
   with ol.VulkanGlfwPresenter(w,config=cfg) as p:
    for x in list(np.linspace(-.3,.3,16))+[.3]*29:
     p.present_wavefront(scene,pose(scene,glass,bars,subject,float(x)),320,240)
     frames.append(p.capture_wavefront_hdr())
    import vulkan as vk
    ex=p._core.wavefront_executor; size=320*240*128
    p._core._single_use(lambda cmd: vk.vkCmdCopyBuffer(cmd,ex.secondary_path_buffer.buffer,ex.secondary_path_readback.buffer,1,[vk.VkBufferCopy(srcOffset=0,dstOffset=0,size=size)]))
    memory=vk.vkMapMemory(p._core.device,ex.secondary_path_readback.memory,0,size,0)
    data=np.frombuffer(memory,dtype=np.float32,count=320*240*32).copy().reshape(240,320,8,4)
    vk.vkUnmapMemory(p._core.device,ex.secondary_path_readback.memory)
    np.save(out/'handoff.npy',data)
    np.save(out/(subject+'-'+active+'.npy'),frames)

finally:g.destroy_window(w);g.terminate()
