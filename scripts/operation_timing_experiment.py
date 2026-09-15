"""Serialized per-operation timestamps for the headless diagnostic only."""
from dataclasses import replace
import vulkan as vk


class OperationTimer:
    def __init__(self,runtime):
        self.runtime=runtime
        bits=vk.vkGetPhysicalDeviceQueueFamilyProperties(runtime.physical_device)[runtime.queue_family].timestampValidBits
        if not bits:raise RuntimeError('Queue has no timestamps')
        self.mask=(1<<bits)-1
        self.period=vk.vkGetPhysicalDeviceProperties(runtime.physical_device).limits.timestampPeriod
        self.pool=vk.vkCreateQueryPool(runtime.device,vk.VkQueryPoolCreateInfo(queryType=vk.VK_QUERY_TYPE_TIMESTAMP,queryCount=2048),None)
        self.labels={}

    def wrap(self,label,operation):
        stages=[]
        for number,stage in enumerate(operation.passes):
            key=f'{label}.{number}:{stage.name}'
            if key not in self.labels:self.labels[key]=len(self.labels)*2
            first=self.labels[key]
            if first+1>=2048:raise RuntimeError('Diagnostic timestamp capacity exceeded')
            def record(command,stage=stage,first=first):
                vk.vkCmdResetQueryPool(command,self.pool,first,2)
                vk.vkCmdWriteTimestamp(command,vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,self.pool,first)
                stage.record(command)
                if stage.workgroups is not None:vk.vkCmdDispatch(command,*stage.workgroups)
                vk.vkCmdWriteTimestamp(command,vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,self.pool,first+1)
            stages.append(replace(stage,record=record,workgroups=None))
        operation.passes=tuple(stages)
        return operation

    def read(self):
        count=len(self.labels)*2
        values=vk.ffi.new('uint64_t[]',count)
        vk.vkGetQueryPoolResults(self.runtime.device,self.pool,0,count,vk.ffi.sizeof(values),values,8,vk.VK_QUERY_RESULT_64_BIT|vk.VK_QUERY_RESULT_WAIT_BIT)
        return {key:((int(values[i+1])-int(values[i]))&self.mask)*self.period/1e6 for key,i in self.labels.items()}

    def close(self):
        with self.runtime.lock:vk.vkQueueWaitIdle(self.runtime.queue)
        vk.vkDestroyQueryPool(self.runtime.device,self.pool,None)
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
