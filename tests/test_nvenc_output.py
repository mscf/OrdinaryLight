import unittest

from ordinarylight.gpu import VulkanBufferMetadata
from ordinarylight.outputs.nvenc import _Nv12CudaFrame


class NvencOutputTests(unittest.TestCase):
    def test_nv12_cuda_planes_preserve_external_pitch_and_offsets(self):
        metadata = VulkanBufferMetadata(
            width=1920, height=1080, format="NV12", pitch=2048,
            y_offset=0, uv_offset=2048 * 1080,
            memory_size=2048 * 1620, memory_offset=0,
            dedicated_allocation=True, device_uuid="00" * 16,
            buffer_handle=1, memory_handle=2, device_handle=3,
            physical_device_handle=4, completion_fence_handle=5,
            queue_family_index=0,
        )
        frame = _Nv12CudaFrame(0x100000, metadata)
        luma, chroma = frame.cuda()
        y = luma.__cuda_array_interface__
        uv = chroma.__cuda_array_interface__
        self.assertEqual(y["shape"], (1080, 1920, 1))
        self.assertEqual(y["strides"], (2048, 1, 1))
        self.assertEqual(uv["shape"], (540, 960, 2))
        self.assertEqual(uv["strides"], (2048, 2, 1))
        self.assertEqual(uv["data"][0], 0x100000 + 2048 * 1080)
        self.assertEqual(frame.frameSize, 1920 * 1080 * 3 // 2)


if __name__ == "__main__":
    unittest.main()
