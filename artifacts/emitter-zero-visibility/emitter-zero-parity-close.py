import sys,json
import numpy as np
from ordinarylight import EnvironmentLight
from ordinarylight.targets.vulkan.api import RendererConfig
from vxl8r_render import GridSpec,GridSnapshot,Cell,VoxelSource,OrthographicCamera
from vxl8r_render.backends import DynamicResidentVulkanRenderer
rows={}; counts=[]
grid=GridSpec(cell_size=.2)
floor=VoxelSource(GridSnapshot(grid,{(x,-1,z):Cell((.3,.4,.5)) for x in range(-5,6) for z in range(-5,6)}))
emitter=VoxelSource(GridSnapshot(grid,{(0,1,0):Cell(emission=(4.,2.,1.))}))
for sampling in ('uniform','power'):
 config=RendererConfig(max_bounces=4,denoiser_enabled=True,temporal_history=True,progressive_accumulation=True,denoiser_sampled_indirect=True,wavefront_raw_hdr_output=True,wavefront_profiling=True,wavefront_timestamps=True)
 with DynamicResidentVulkanRenderer(grid,[floor,emitter],bounds=((-5,-2,-5),(5,4,5)),static_coordinates=floor.voxels.cells,static_sources=(0,),chunk_capacity=16,gpu_animation=True,gpu_layout=True,tight_dynamic_bounds=True,environment=EnvironmentLight(color=(0,0,0)),config=config,emitter_sampling=sampling) as r:
  for i,x in enumerate((0.,0.,0.,0.)):
   pose=np.eye(4);pose[0,3]=x
   r.render([np.eye(4),pose],camera=OrthographicCamera(position=(1.+i*.01,1.,-2.),target=(0.,0.,0.),vertical_size=2.4),width=97,height=73).wait()
   rows[f'{sampling}_{i}_identity']=r.read_primary_hits()['identity']
   for source in ('raw','upstream','output'):rows[f'{sampling}_{i}_{source}']=r.read_hdr(source=source)
   counts.append(dict(sampling=sampling,frame=i,counters=r.pipeline.last_timings.get('wavefront_work_counters',{})))
np.savez(sys.argv[1],**rows)
print(json.dumps(counts))
