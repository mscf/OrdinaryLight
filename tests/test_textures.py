import unittest

import numpy as np

import ordinarylight as ol


class TextureTests(unittest.TestCase):
    def test_texture_1d_sampling_and_validation(self):
        texture = ol.Texture1D(((0, 0, 0), (1, 0.5, 0.25)))
        np.testing.assert_allclose(texture.sample((0, 0.5, 1)), (
            (0, 0, 0, 1), (0.5, 0.25, 0.125, 1), (1, 0.5, 0.25, 1),
        ))
        self.assertFalse(texture.values.flags.writeable)
        repeated = ol.Texture1D(
            ((0, 0, 0, 1), (1, 1, 1, 1)),
            address_mode="repeat", linear_filter=False,
        )
        np.testing.assert_array_equal(
            repeated.sample((-0.25, 1.25)), ((1, 1, 1, 1), (0, 0, 0, 1))
        )
        with self.assertRaises(ValueError):
            ol.Texture1D(np.zeros((2, 2), np.float32))
        with self.assertRaises(ValueError):
            ol.Texture1D(((0, 0, 0),), address_mode="border")

    def test_texture_validates_rgba_pixels(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            ol.Texture(np.zeros((2, 2, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "wrap_s"):
            ol.Texture(np.zeros((1, 1, 4), dtype=np.uint8), wrap_s="border")

    def test_scene_packs_texture_metadata_and_texels(self):
        pixels = np.asarray(
            [[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.uint8
        )
        texture = ol.Texture(
            pixels, wrap_s="clamp", wrap_t="mirror", linear_filter=False
        )
        scene = ol.Scene()
        scene.add_mesh(
            np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
            np.asarray(((0, 1, 2),), np.uint32),
            ol.Material(base_color_texture=texture),
        )
        packed = scene.texture_data()
        self.assertEqual(tuple(packed[:9]), (1, 9, 12, 2, 1, 9, 2, 0, 0))
        self.assertEqual(packed[9], 0x04030201)
        self.assertEqual(packed[10], 0x08070605)
        self.assertEqual(packed[12], 0x04030201)
        self.assertEqual(packed[13], 0x08070605)
        self.assertEqual(scene.triangle_material_data()[0, 4, 0], 0.0)

    def test_untextured_material_uses_negative_index(self):
        scene = ol.Scene()
        scene.add_mesh(
            np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
            np.asarray(((0, 1, 2),), np.uint32),
        )
        self.assertEqual(scene.texture_data().tolist(), [0])
        self.assertEqual(scene.triangle_material_data()[0, 4, 0], -1.0)

    def test_mips_separate_srgb_and_linear_averaging(self):
        pixels = np.asarray((
            ((0, 0, 0, 255), (255, 255, 255, 255)),
            ((255, 255, 255, 255), (0, 0, 0, 255)),
        ), dtype=np.uint8)
        texture = ol.Texture(pixels)
        scene = ol.Scene()
        scene.add_mesh(
            np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
            np.asarray(((0, 1, 2),), np.uint32),
            ol.Material(base_color_texture=texture),
        )
        packed = scene.texture_data()
        self.assertEqual(tuple(packed[1:7]), (9, 14, 2, 2, 16, 2))
        self.assertEqual(packed[13], 0xFFBCBCBC)
        self.assertEqual(packed[18], 0xFF808080)

    def test_material_texture_slots_share_stable_table(self):
        first = ol.Texture(np.zeros((1, 1, 4), dtype=np.uint8))
        second = ol.Texture(np.full((1, 1, 4), 255, dtype=np.uint8))
        scene = ol.Scene()
        scene.add_mesh(
            np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
            np.asarray(((0, 1, 2),), np.uint32),
            ol.Material(
                base_color_texture=first,
                metallic_roughness_texture=second,
                emissive_texture=first,
                normal_texture=second,
                normal_scale=0.75,
                occlusion_texture=first,
                occlusion_strength=0.6,
                transmission_texture=second,
            ),
        )
        self.assertEqual(scene.textures, (first, second))
        self.assertEqual(
            scene.triangle_material_data()[0, 4].tolist(),
            [0.0, 1.0, 0.0, 1.0],
        )
        self.assertEqual(scene.triangle_material_data()[0, 5, 0], 0.75)
        np.testing.assert_allclose(
            scene.triangle_material_data()[0, 5], (0.75, 0.0, 0.6, 1.0)
        )
        self.assertIs(scene.textures[1], second)

    def test_advanced_material_parameters_and_textures_pack_stably(self):
        texture = ol.Texture(np.full((2, 2, 4), 192, dtype=np.uint8))
        material = ol.Material(
            clearcoat=0.8, clearcoat_roughness=0.12,
            sheen_color=(0.2, 0.4, 0.8), sheen_roughness=0.35,
            anisotropy=-0.6, thin_walled=True,
            subsurface=0.55, subsurface_color=(1.0, 0.25, 0.1),
            subsurface_radius=0.7, clearcoat_texture=texture,
            sheen_texture=texture, anisotropy_texture=texture,
            subsurface_texture=texture,
        )
        scene = ol.Scene()
        scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),), material,
            texcoords=((0, 0), (1, 0), (0, 1)),
        )
        packed = scene.triangle_material_data()[0]
        self.assertEqual(packed.shape, (12, 4))
        np.testing.assert_allclose(packed[6], (0.8, 0.12, 0.35, -0.6))
        np.testing.assert_allclose(packed[7], (0.55, 0.7, 1.0, 0.0))
        np.testing.assert_allclose(packed[8, :3], (0.2, 0.4, 0.8))
        np.testing.assert_allclose(packed[9, :3], (1.0, 0.25, 0.1))
        self.assertTrue(np.all(packed[10] >= 0.0))
        with self.assertRaises(ValueError):
            ol.Material(clearcoat=1.1)
        with self.assertRaises(ValueError):
            ol.Material(anisotropy=-1.1)

    def test_texture_bindings_deduplicate_resource_and_transform(self):
        texture = ol.Texture(np.full((1, 1, 4), 255, dtype=np.uint8))
        transform = ol.TextureTransform(
            offset=(0.25, -0.5), scale=(2.0, 0.5), rotation=np.pi / 2.0,
            texcoord_set=1,
        )
        scene = ol.Scene()
        vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32)
        indices = np.asarray(((0, 1, 2),), np.uint32)
        scene.add_mesh(vertices, indices, ol.Material(
            base_color_texture=texture, base_color_transform=transform,
            emissive_texture=texture, emissive_transform=transform,
        ))
        scene.add_mesh(vertices, indices, ol.Material(
            base_color_texture=texture,
            base_color_transform=ol.TextureTransform(scale=(3.0, 1.0)),
        ))
        self.assertEqual(len(scene.textures), 1)
        self.assertEqual(len(scene.texture_bindings), 2)
        packed = scene.texture_binding_data()
        np.testing.assert_allclose(packed[0, 0], (0.0, 0.0, 1.0, 1.0), atol=1e-6)
        np.testing.assert_allclose(packed[0, 1], (0.25, -0.5, 2.0, 0.5))
        material_data = scene.triangle_material_data()
        np.testing.assert_allclose(material_data[0, 4], (0, -1, 0, -1))
        self.assertEqual(material_data[0, 3, 2], 0.25)
        self.assertEqual(material_data[1, 4, 0], 1.0)
        self.assertEqual(material_data[1, 3, 2], 0.0)

    def test_texture_transform_rejects_unknown_uv_set(self):
        with self.assertRaisesRegex(ValueError, "zero or one"):
            ol.TextureTransform(texcoord_set=2)


if __name__ == "__main__":
    unittest.main()
