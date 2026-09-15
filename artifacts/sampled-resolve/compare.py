"""Compare packaged baseline and optimized HDR resolve with animated native 4K inputs."""
import json,struct,sys,importlib.util
from pathlib import Path
import ordinarylight.targets.vulkan.path_resolve_graph as resolve_module
original_files=resolve_module.files
variants={False:lambda package:Path(__file__).with_name('baseline'),True:original_files}
from contextlib import ExitStack
from dataclasses import replace
import numpy as np
from vxl8r_render.backends import DynamicResidentVulkanRenderer
from vxl8r_render.backends.gpu_timing import GpuGraphTimer
from vxl8r_render.backends.gpu_animation import GpuAnimation
from vxl8r_render.backends.native_acceleration import compiled_bounds
from vxl8r_render.viewer.scenes import scene_sources
from vxl8r_render.viewer.state import ViewerState
from vxl8r_render.viewer.overlay import HelpOverlay
from ordinarylight.targets.vulkan.api import RendererConfig
from ordinarylight.runtime import VulkanKernel
from ordinarylight.pipeline.graph import reflected_operation
from ordinarylight.pipeline.vulkan import VulkanResource
spec,_,sources=scene_sources('voxel',512,sparse=True)
state=ViewerState(native_gi=True,playing=True)
if "--overview" in sys.argv:state.zoom=512*.2*1.35
config=RendererConfig(max_bounces=4,samples_per_pixel=1,denoiser_enabled=True,temporal_history=True,progressive_accumulation=True,denoiser_sampled_indirect=True,wavefront_raw_hdr_output=True,wavefront_timestamps=True,wavefront_tile_capacity=524288)
with ExitStack() as stack:
 rs={n:stack.enter_context(DynamicResidentVulkanRenderer(spec,sources,bounds=((-256,-5,-256),(255,7,255)),static_coordinates=sources[0].voxels.cells,static_sources=(0,),chunk_capacity=128,gpu_animation=True,gpu_layout=True,tight_dynamic_bounds=True,config=config,frame_effect_factory=lambda runtime,hdr:HelpOverlay(runtime,hdr,state))) for n in (False,True)}
 for n,r in rs.items():
  resolve_module.files=variants[n]
  r.set_transform(r.objects[1],state.transform())
  r.render(camera=state.camera(),width=3840,height=2160,render_extent=(3840,2160)).wait()
 timers={n:stack.enter_context(GpuGraphTimer(r.runtime)) for n,r in rs.items()}
 rows={n:[] for n in rs}
 for i in range(64):
  state.time=(i+1)/60
  for n in ((False,True) if i%2==0 else (True,False)):
   resolve_module.files=variants[n]
   r=rs[n];r.set_transform(r.objects[1],state.transform())
   r.render(camera=state.camera(),width=3840,height=2160,render_extent=(3840,2160),averaged=True).wait()
   if i>=16:rows[n].append(dict(**timers[n].last,**r.pipeline.last_timings['wavefront_stage_ms']))
 for n,data in rows.items():
  print(json.dumps(dict(optimized_resolve=n,samples=len(data),gpu={k:dict(median=float(np.median([x[k] for x in data])),p95=float(np.percentile([x[k] for x in data],95))) for k in ('total','native_gi','primary','intersect_shade.1','gpu_population','face_output','resolve_hdr','relax_prepare')})),flush=True)
 for source in ('raw','upstream','output'):
  a=rs[False].read_hdr(source=source);b=rs[True].read_hdr(source=source)
  print(json.dumps(dict(source=source,max_error=float(np.max(abs(a-b))),mean_error=float(np.mean(abs(a-b))))),flush=True)
  np.testing.assert_allclose(a,b,atol=.002,rtol=.002)
