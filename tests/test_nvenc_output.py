import unittest
from types import SimpleNamespace

from ordinarylight.gpu import VulkanBufferMetadata
from ordinarylight.outputs.nvenc import (
    NvencVideoWriter, _Nv12CudaFrame, _Yuv420CudaFrame,
)


class NvencOutputTests(unittest.TestCase):
    @staticmethod
    def _keyframe_writer(encoder):
        writer = object.__new__(NvencVideoWriter)
        writer._encoder = encoder
        writer._nvc = SimpleNamespace(
            NV_ENC_PIC_FLAGS=SimpleNamespace(FORCEIDR=2, OUTPUT_SPSPPS=4)
        )
        writer._closed = False
        writer._pending_keyframe = False
        writer._pending_repeat_headers = False
        return writer

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

    def test_p010_cuda_planes_use_msb_aligned_16_bit_storage(self):
        metadata = VulkanBufferMetadata(
            width=1920, height=1080, format="P010", pitch=3840,
            y_offset=0, uv_offset=3840 * 1080,
            memory_size=3840 * 1620, memory_offset=0,
            dedicated_allocation=True, device_uuid="00" * 16,
            buffer_handle=1, memory_handle=2, device_handle=3,
            physical_device_handle=4, completion_fence_handle=5,
            queue_family_index=0, bit_depth=10, storage_bits=16,
        )
        frame = _Yuv420CudaFrame(0x200000, metadata)
        luma, chroma = frame.cuda()
        y = luma.__cuda_array_interface__
        uv = chroma.__cuda_array_interface__
        self.assertEqual(y["shape"], (1080, 1920, 1))
        self.assertEqual(y["strides"], (3840, 2, 1))
        self.assertEqual(y["typestr"], "|u2")
        self.assertEqual(uv["shape"], (540, 960, 2))
        self.assertEqual(uv["strides"], (3840, 2, 1))
        self.assertEqual(uv["typestr"], "|u2")
        self.assertEqual(uv["data"][0], 0x200000 + 3840 * 1080)
        self.assertEqual(frame.frameSize, 1920 * 1080 * 3)

    def test_buffer_metadata_defaults_describe_eight_bit_storage(self):
        metadata = VulkanBufferMetadata(
            width=8, height=4, format="NV12", pitch=8,
            y_offset=0, uv_offset=32, memory_size=48, memory_offset=0,
            dedicated_allocation=True, device_uuid="00" * 16,
            buffer_handle=1, memory_handle=2, device_handle=3,
            physical_device_handle=4, completion_fence_handle=5,
            queue_family_index=0,
        )
        self.assertEqual(metadata.bit_depth, 8)
        self.assertEqual(metadata.storage_bits, 8)

    def test_writer_rejects_unsupported_pixel_format_before_loading_cuda(self):
        with self.assertRaisesRegex(ValueError, "pixel_format"):
            NvencVideoWriter(None, (1280, 720), pixel_format="rgba8")

    def test_writer_rejects_p010_with_h264(self):
        with self.assertRaisesRegex(ValueError, "HEVC/H.265 or AV1"):
            NvencVideoWriter(None, (1280, 720), pixel_format="p010")

    def test_encode_combines_idr_and_repeated_header_flags(self):
        class Encoder:
            def __init__(self):
                self.calls = []

            def Encode(self, frame, **kwargs):
                self.calls.append((frame, kwargs))
                return []

        encoder = Encoder()
        writer = self._keyframe_writer(encoder)
        marker = object()
        writer._encode(marker, force_idr=True, repeat_headers=True)
        self.assertEqual(encoder.calls, [(marker, {"pic_flags": 6})])

    def test_encode_uses_unflagged_path_for_normal_frames(self):
        class Encoder:
            def __init__(self):
                self.calls = []

            def Encode(self, *args):
                self.calls.append(args)
                return []

        encoder = Encoder()
        writer = self._keyframe_writer(encoder)
        marker = object()
        writer._encode(marker)
        self.assertEqual(encoder.calls, [(marker,)])

    def test_encode_falls_back_to_positional_picture_flags(self):
        class Encoder:
            def __init__(self):
                self.calls = []

            def Encode(self, frame, flags=None):
                self.calls.append((frame, flags))
                return []

        encoder = Encoder()
        writer = self._keyframe_writer(encoder)
        marker = object()
        writer._encode(marker, force_idr=True, repeat_headers=True)
        self.assertEqual(encoder.calls[-1], (marker, 6))

    def test_request_keyframe_is_coalesced_and_retained(self):
        writer = self._keyframe_writer(None)
        writer.request_keyframe(repeat_headers=False)
        writer.request_keyframe(repeat_headers=True)
        self.assertTrue(writer._pending_keyframe)
        self.assertTrue(writer._pending_repeat_headers)

    def test_request_keyframe_rejects_closed_writer(self):
        writer = self._keyframe_writer(None)
        writer._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            writer.request_keyframe()


if __name__ == "__main__":
    unittest.main()
