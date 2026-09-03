import unittest
from pathlib import Path

from ordinarylight.shaders.abi import (
    SECONDARY_PATH_STATE_ABI, VOLUME_HEADER_ABI, reflect_spirv_struct,
)
from ordinarylight.volume import VOLUME_HEADER_DTYPE
from ordinarylight.wavefront import SECONDARY_PATH_STATE_DTYPE


ROOT = Path(__file__).parents[1]
SHADERS = ROOT / "ordinarylight" / "shaders"


class ShaderAbiTests(unittest.TestCase):
    def test_canonical_contracts_define_cpu_dtypes(self):
        SECONDARY_PATH_STATE_ABI.validate_dtype(SECONDARY_PATH_STATE_DTYPE)
        VOLUME_HEADER_ABI.validate_dtype(VOLUME_HEADER_DTYPE)

    def test_every_secondary_path_source_has_identical_members(self):
        paths = (
            "wavefront_primary_impl.glsl",
            "wavefront_shade.comp",
            "wavefront_shade_candidate.glsl",
            "wavefront_path_to_hdr.comp",
            "denoiser_relax_prepare.comp.wgsl",
        )
        for path in paths:
            with self.subTest(path=path):
                SECONDARY_PATH_STATE_ABI.validate_shader_source(
                    (SHADERS / path).read_text()
                )

    def test_volume_header_sources_match_compact_and_generated_forms(self):
        VOLUME_HEADER_ABI.validate_shader_source(
            (SHADERS / "wavefront_volumes.glsl").read_text()
        )
        VOLUME_HEADER_ABI.validate_shader_source(
            (SHADERS / "wavefront_shade_candidate.glsl").read_text(),
            expand_arrays=True,
        )

    def test_spirv_secondary_path_offsets_and_array_stride(self):
        expected = tuple(field.offset for field in SECONDARY_PATH_STATE_ABI.fields)
        for path in (
            "wavefront_primary.comp.spv",
            "wavefront_shade.comp.spv",
            "wavefront_path_to_hdr.comp.spv",
            "wavefront_shade_ordinaryshade.comp.spv",
        ):
            with self.subTest(path=path):
                offsets, stride = reflect_spirv_struct(
                    SHADERS / path, SECONDARY_PATH_STATE_ABI.name
                )
                self.assertEqual(offsets, expected)
                self.assertEqual(stride, SECONDARY_PATH_STATE_ABI.size)

    def test_spirv_volume_header_offsets_and_array_stride(self):
        expanded_offsets = tuple(range(192, 320, 16))
        expected = tuple(
            field.offset for field in VOLUME_HEADER_ABI.fields[:-1]
        ) + expanded_offsets
        compact = tuple(field.offset for field in VOLUME_HEADER_ABI.fields)
        for path, expected_offsets in (
            ("wavefront_primary.comp.spv", compact),
            ("wavefront_shade.comp.spv", compact),
            ("wavefront_shade_ordinaryshade.comp.spv", expected),
        ):
            with self.subTest(path=path):
                offsets, stride = reflect_spirv_struct(
                    SHADERS / path, VOLUME_HEADER_ABI.name,
                )
                self.assertEqual(offsets, expected_offsets)
                self.assertEqual(stride, VOLUME_HEADER_ABI.size)

    def test_spirv_reflection_rejects_invalid_binary(self):
        with self.assertRaisesRegex(ValueError, "SPIR-V"):
            reflect_spirv_struct(__file__, "Missing")


if __name__ == "__main__":
    unittest.main()
