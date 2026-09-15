"""GPU-only tile-local continuation list for the selected diffuse diagnostic."""
from dataclasses import replace
import ordinaryshade as osh

@osh.compute(workgroup_size=(1,1,1))
def prepare_dispatch(control: osh.storage_buffer(osh.u32,access='read_write',binding=0)):
    groups=(control[3]+osh.u32(63))/osh.u32(64)
    control[0]=osh.minimum(groups,osh.u32(4096))
    control[1]=(groups+osh.u32(4095))/osh.u32(4096)
    control[2]=osh.u32(1)

class CompactContinuation:
    def __init__(self,stack,runtime,capacity):
        import vulkan as vk
        from ordinarylight.runtime import VulkanKernel,compile_compute
        from ordinarylight.pipeline.vulkan import VulkanResource
        self.control=stack.enter_context(runtime.buffer(16,memory='device',usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT))
        self.indices=stack.enter_context(runtime.buffer(capacity*4,memory='device'))
        self.bindings={36:VulkanResource.buffer(self.control),37:VulkanResource.buffer(self.indices)}
        self.compiled=osh.compile(prepare_dispatch)
        self.kernel=stack.enter_context(VulkanKernel(runtime,compile_compute(self.compiled.source),{0:self.bindings[36]}))

    def reset(self):
        import vulkan as vk
        from ordinarylight.pipeline.graph import VulkanOperation
        from ordinarylight.pipeline.vulkan import VulkanPass,VulkanResourceUse
        return VulkanOperation([VulkanPass('reset_continuations',(VulkanResourceUse(self.bindings[36],vk.VK_PIPELINE_STAGE_TRANSFER_BIT,vk.VK_ACCESS_TRANSFER_WRITE_BIT),),lambda cmd:vk.vkCmdFillBuffer(cmd,self.control.buffer,0,16,0))])

    def prepare(self):
        from ordinarylight.pipeline.graph import reflected_operation
        return reflected_operation(self.kernel,self.compiled.reflection,workgroups=(1,1,1))

    def indirect(self,operation,kernel,constants):
        import vulkan as vk
        def record(command):
            kernel.bind(command,constants)
            vk.vkCmdDispatchIndirect(command,self.control.buffer,0)
        operation.passes=tuple(replace(p,record=record,uses=tuple(replace(u,stage=u.stage|vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,access=u.access|vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT) if u.resource == self.bindings[36] else u for u in p.uses)) for p in operation.passes)
        return operation
