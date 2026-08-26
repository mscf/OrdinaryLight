import io
import unittest
from unittest.mock import patch

import numpy as np

import ordinarylight as ol


class OutputTests(unittest.TestCase):
    def test_hdr_to_sdr_is_rgba_uint8_and_finite(self):
        hdr = np.asarray([[[0.0, 0.18, 1.0, 0.5], [8.0, 2.0, 0.1, 1.0]]])
        result = ol.outputs.to_sdr(hdr, alpha=True)
        self.assertEqual(result.shape, (1, 2, 4))
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result[0, 0, 3]), 128)
        self.assertGreater(int(result[0, 1, 0]), int(result[0, 0, 0]))

    def test_sdr_conversion_validates_shape_mode_and_dtype(self):
        with self.assertRaises(ValueError):
            ol.outputs.to_sdr(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            ol.outputs.to_sdr(np.zeros((2, 2, 3)), tone_mapping="unknown")
        with self.assertRaises(TypeError):
            ol.outputs.to_sdr(np.zeros((2, 2, 3), np.uint16))

    @patch("ordinarylight.outputs.video.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_video_writer_builds_path_and_stream_commands(self, _which):
        path_writer = ol.outputs.FFmpegVideoWriter(
            "movie.mp4", (1920, 1080), fps=24, codec="libx265", quality=20
        )
        command = path_writer._command(4)
        self.assertIn("1920x1080", command)
        self.assertIn("libx265", command)
        self.assertEqual(command[-1], "movie.mp4")

        stream_writer = ol.outputs.FFmpegVideoWriter(io.BytesIO(), (8, 4))
        self.assertEqual(stream_writer._command(3)[-3:], ["-f", "matroska", "pipe:1"])
        with self.assertRaises(TypeError):
            stream_writer.write(np.zeros((4, 8, 3), np.float32))

    @patch("ordinarylight.outputs.video.shutil.which", return_value=None)
    def test_video_writer_reports_missing_optional_encoder(self, _which):
        with self.assertRaisesRegex(RuntimeError, "not found"):
            ol.outputs.FFmpegVideoWriter("movie.mp4", (4, 4))


if __name__ == "__main__":
    unittest.main()
