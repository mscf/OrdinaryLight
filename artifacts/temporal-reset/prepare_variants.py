from dataclasses import replace
import struct
import ordinaryshade as osh
import ordinarylight.runtime.relax_prepare as prepare_module
from ordinarylight.denoising.kernels import prepare_relax_signals,prepare_decode_normal,prepare_unpack_normal,prepare_previous_pixel,prepare_custom_surface_history
from ordinarylight.runtime import compile_compute
import vulkan as vk
original_operation=prepare_module.relax_prepare_operation
helpers=(prepare_decode_normal,prepare_unpack_normal,prepare_previous_pixel,prepare_custom_surface_history)
binaries={n:compile_compute(osh.compile(replace(prepare_relax_signals,workgroup_size=(n,1,1)),helpers=helpers).source) for n in (64,128,256)}
current=64
prepare_module._custom_history_shader=lambda:binaries[current]
def operation(kernel,**kwargs):
 op=original_operation(kernel,**kwargs)
 size=current
 pc=struct.pack('8I',*kwargs['extent'],kwargs['path_count'],int(kwargs.get('transmission_motion_cap',False)),kwargs.get('sample_index',0),kwargs.get('sample_count',1),int(kwargs.get('sampled_indirect',False)),int(kwargs.get('planar_mirror_guides',False)))
 def record(command):
  kernel.bind(command,pc)
  vk.vkCmdDispatch(command,(kwargs['path_count']+size-1)//size,1,1)
 op.passes=tuple(replace(stage,record=record) for stage in op.passes)
 return op
prepare_module.relax_prepare_operation=operation
