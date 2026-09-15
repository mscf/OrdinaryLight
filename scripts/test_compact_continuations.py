"""GPU argument generation and reset at empty, tail, and multi-row capacities."""
from contextlib import ExitStack
import struct
import pytest
from compact_continuation_experiment import CompactContinuation

@pytest.mark.parametrize('count', (0,1,63,64,65,262144,262145,524288))
def test_gpu_continuation_dispatch_and_reset(count):
    from ordinarylight.runtime import VulkanRuntime
    from ordinarylight.pipeline.graph import VulkanGraph
    with ExitStack() as stack:
        runtime=stack.enter_context(VulkanRuntime())
        compact=CompactContinuation(stack,runtime,524288)
        compact.control.upload(struct.pack('<4I',0,0,0,count))
        VulkanGraph().add('prepare',compact.prepare()).compile().execute(runtime).wait()
        x,y,z,actual=struct.unpack('<4I',compact.control.read())
        groups=(count+63)//64
        assert (x,y,z,actual)==(min(groups,4096),(groups+4095)//4096,1,count)
        assert x*y*64>=count
        graph=VulkanGraph().add('reset',compact.reset())
        graph.add('prepare',compact.prepare(),after=('reset',)).compile().execute(runtime).wait()
        assert struct.unpack('<4I',compact.control.read())==(0,0,1,0)
