import unittest
from base64 import b64encode
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import ordinarylight as ol
import ordinarylight.gltf as legacy_gltf
from ordinarylight import loaders
from ordinarylight.loaders import gltf


class GltfLoaderTests(unittest.TestCase):
    def test_loader_namespace_is_canonical_and_compatible(self):
        self.assertIs(gltf.load, gltf.load_gltf)
        self.assertIs(loaders.load_gltf, gltf.load)
        self.assertIs(ol.load_gltf, gltf.load)
        self.assertIs(legacy_gltf.load, gltf.load)
        self.assertIs(legacy_gltf.load_gltf, gltf.load)
        self.assertEqual(gltf.load.__module__, "ordinarylight.loaders.gltf")
        self.assertEqual(loaders.supported_formats(), (".glb", ".gltf"))
        with self.assertRaisesRegex(ValueError, "unsupported scene format"):
            loaders.load("scene.obj")

    def test_repeated_nodes_share_one_mesh_resource(self):
        positions = np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32
        )
        indices = np.asarray((0, 1, 2), np.uint16)
        animation_times = np.asarray((0, 2), np.float32)
        animation_translations = np.asarray(((3, 0, 0), (5, 0, 0)), np.float32)
        payload = (
            positions.tobytes() + indices.tobytes()
            + animation_times.tobytes() + animation_translations.tobytes()
        )
        times_offset = positions.nbytes + indices.nbytes
        translations_offset = times_offset + animation_times.nbytes
        document = {
            "asset": {"version": "2.0"},
            "extensionsUsed": ["KHR_lights_punctual"],
            "extensions": {"KHR_lights_punctual": {"lights": [
                {
                    "type": "point", "color": [1, 0.5, 0.25],
                    "intensity": 12, "range": 8,
                },
                {"type": "directional", "intensity": 3},
                {
                    "type": "spot", "intensity": 20,
                    "spot": {"innerConeAngle": 0.2, "outerConeAngle": 0.6},
                },
            ]}},
            "buffers": [{
                "byteLength": len(payload),
                "uri": "data:application/octet-stream;base64,"
                + b64encode(payload).decode("ascii"),
            }],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
                {
                    "buffer": 0, "byteOffset": positions.nbytes,
                    "byteLength": indices.nbytes,
                },
                {
                    "buffer": 0, "byteOffset": times_offset,
                    "byteLength": animation_times.nbytes,
                },
                {
                    "buffer": 0, "byteOffset": translations_offset,
                    "byteLength": animation_translations.nbytes,
                },
            ],
            "accessors": [
                {
                    "bufferView": 0, "componentType": 5126,
                    "count": 3, "type": "VEC3",
                },
                {
                    "bufferView": 1, "componentType": 5123,
                    "count": 3, "type": "SCALAR",
                },
                {
                    "bufferView": 2, "componentType": 5126,
                    "count": 2, "type": "SCALAR",
                },
                {
                    "bufferView": 3, "componentType": 5126,
                    "count": 2, "type": "VEC3",
                },
            ],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0}, "indices": 1, "material": 0,
            }]}],
            "materials": [{
                "emissiveFactor": [0.25, 0.5, 1.0],
                "extensions": {
                    "KHR_materials_emissive_strength": {"emissiveStrength": 4.0},
                    "KHR_materials_unlit": {},
                },
            }],
            "nodes": [
                {
                    "name": "left", "mesh": 0,
                    "translation": [0, 0, 0], "extras": {"group": 1},
                },
                {
                    "name": "right", "mesh": 0,
                    "translation": [3, 0, 0], "extras": {"group": 2},
                },
                {
                    "name": "point", "translation": [1, 2, 3],
                    "extensions": {"KHR_lights_punctual": {"light": 0}},
                },
                {
                    "name": "sun",
                    "extensions": {"KHR_lights_punctual": {"light": 1}},
                },
                {
                    "name": "spot", "translation": [0, 4, 0],
                    "extensions": {"KHR_lights_punctual": {"light": 2}},
                },
            ],
            "scenes": [{"nodes": [0, 1, 2, 3, 4]}],
            "scene": 0,
            "animations": [{
                "name": "move-right",
                "samplers": [{"input": 2, "output": 3, "interpolation": "LINEAR"}],
                "channels": [{
                    "sampler": 0,
                    "target": {"node": 1, "path": "translation"},
                }],
            }],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "instances.gltf"
            path.write_text(json.dumps(document))
            scene = loaders.load(path)

        self.assertEqual(scene.instance_count, 2)
        self.assertEqual(scene.metadata["source"], "gltf")
        self.assertIn("KHR_lights_punctual", scene.metadata["extensions_used"])
        np.testing.assert_allclose(scene.instances[0].material.emission, (1, 2, 4))
        self.assertIs(scene.instances[0].material.program, ol.unlit_material)
        self.assertEqual(len(scene.mesh_resources), 1)
        self.assertIs(scene.instances[0].resource, scene.instances[1].resource)
        self.assertEqual(scene.instancing_statistics()["shared_blas_savings"], 1)
        self.assertEqual(scene.instances[1].name, "right")
        self.assertEqual(
            [type(light) for light in scene.lights],
            [ol.PointLight, ol.DirectionalLight, ol.SpotLight],
        )
        np.testing.assert_allclose(scene.lights[0].position, (1, 2, 3))
        np.testing.assert_allclose(scene.lights[1].direction, (0, 0, -1))
        self.assertAlmostEqual(scene.lights[2].outer_cone_angle, 0.6)
        self.assertEqual(scene.animations[0].name, "move-right")
        scene.apply_animation(0, 1.0)
        np.testing.assert_allclose(scene.instances[1].transform.matrix[:3, 3], (4, 0, 0))
        self.assertEqual(scene.snapshot()["instances"][1]["metadata"]["extras"], {
            "group": 2,
        })
        np.testing.assert_allclose(
            scene.instances[1].world_vertices,
            scene.instances[0].world_vertices + (4, 0, 0),
        )

    def test_rejects_required_unsupported_extensions(self):
        document = {
            "asset": {"version": "2.0"},
            "extensionsUsed": ["EXT_imaginary_compression"],
            "extensionsRequired": ["EXT_imaginary_compression"],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unsupported.gltf"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(
                ValueError, "requires unsupported extensions"
            ):
                loaders.load(path)

    def test_loads_normalized_skin_weights_and_joint_animation(self):
        arrays = (
            np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
            np.zeros((3, 4), np.uint8),
            np.asarray(((255, 0, 0, 0),) * 3, np.uint8),
            np.asarray((0, 1, 2), np.uint16),
            np.eye(4, dtype=np.float32).T.reshape((1, 16)),
            np.asarray((0, 1), np.float32),
            np.asarray(((0, 0, 0), (1, 0, 0)), np.float32),
        )
        payload = bytearray()
        views = []
        for array in arrays:
            while len(payload) % 4:
                payload.append(0)
            offset = len(payload)
            data = array.tobytes()
            payload.extend(data)
            views.append({
                "buffer": 0, "byteOffset": offset, "byteLength": len(data),
            })
        document = {
            "asset": {"version": "2.0"},
            "buffers": [{
                "byteLength": len(payload),
                "uri": "data:application/octet-stream;base64,"
                + b64encode(payload).decode("ascii"),
            }],
            "bufferViews": views,
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3,
                 "type": "VEC3"},
                {"bufferView": 1, "componentType": 5121, "count": 3,
                 "type": "VEC4"},
                {"bufferView": 2, "componentType": 5121, "count": 3,
                 "type": "VEC4", "normalized": True},
                {"bufferView": 3, "componentType": 5123, "count": 3,
                 "type": "SCALAR"},
                {"bufferView": 4, "componentType": 5126, "count": 1,
                 "type": "MAT4"},
                {"bufferView": 5, "componentType": 5126, "count": 2,
                 "type": "SCALAR"},
                {"bufferView": 6, "componentType": 5126, "count": 2,
                 "type": "VEC3"},
            ],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                "indices": 3,
            }]}],
            "skins": [{"joints": [0], "inverseBindMatrices": 4}],
            "nodes": [{"name": "joint"}, {"name": "mesh", "mesh": 0, "skin": 0}],
            "scenes": [{"nodes": [0, 1]}],
            "scene": 0,
            "animations": [{
                "samplers": [{"input": 5, "output": 6}],
                "channels": [{
                    "sampler": 0,
                    "target": {"node": 0, "path": "translation"},
                }],
            }],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "skin.gltf"
            path.write_text(json.dumps(document))
            scene = ol.load_gltf(path)
        self.assertTrue(scene.instances[0].deformable)
        scene.apply_animation(0, 1.0)
        np.testing.assert_allclose(
            scene.instances[0].vertices,
            ((1, 0, 0), (2, 0, 0), (1, 1, 0)), atol=1e-6,
        )

    def test_loads_sparse_only_position_accessor(self):
        sparse_indices = np.asarray((0, 1, 2), np.uint8)
        sparse_positions = np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32
        )
        triangle_indices = np.asarray((0, 1, 2), np.uint16)
        payload = bytearray(sparse_indices.tobytes())
        while len(payload) % 4:
            payload.append(0)
        positions_offset = len(payload)
        payload.extend(sparse_positions.tobytes())
        indices_offset = len(payload)
        payload.extend(triangle_indices.tobytes())
        document = {
            "asset": {"version": "2.0"},
            "buffers": [{
                "byteLength": len(payload),
                "uri": "data:application/octet-stream;base64,"
                + b64encode(payload).decode("ascii"),
            }],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0,
                 "byteLength": sparse_indices.nbytes},
                {"buffer": 0, "byteOffset": positions_offset,
                 "byteLength": sparse_positions.nbytes},
                {"buffer": 0, "byteOffset": indices_offset,
                 "byteLength": triangle_indices.nbytes},
            ],
            "accessors": [{
                "componentType": 5126, "count": 3, "type": "VEC3",
                "sparse": {
                    "count": 3,
                    "indices": {"bufferView": 0, "componentType": 5121},
                    "values": {"bufferView": 1},
                },
            }, {
                "bufferView": 2, "componentType": 5123,
                "count": 3, "type": "SCALAR",
            }],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": 0}, "indices": 1,
            }]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}], "scene": 0,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sparse.gltf"
            path.write_text(json.dumps(document))
            scene = ol.loaders.gltf.load(path)
        np.testing.assert_allclose(scene.instances[0].vertices, sparse_positions)

    def test_loads_transmission_fixture(self):
        path = Path(__file__).parents[1] / "assets" / "TransmissionTest.glb"
        scene = ol.load_gltf(path)
        triangles, _, _ = scene.triangles()

        self.assertEqual(len(scene.meshes), 22)
        self.assertEqual(len(triangles), 128775)
        self.assertTrue(any(mesh.material.transmission > 0.0 for mesh in scene.meshes))
        self.assertTrue(any(mesh.normals is not None for mesh in scene.meshes))
        self.assertTrue(any(mesh.texcoords is not None for mesh in scene.meshes))
        self.assertEqual(scene.triangle_material_data().shape, (128775, 6, 4))
        self.assertTrue(all(mesh.tangents.shape[1] == 4 for mesh in scene.meshes))
        self.assertTrue(any(
            mesh.material.metallic_roughness_texture is not None
            for mesh in scene.meshes
        ))
        self.assertEqual(sum(
            mesh.material.transmission_texture is not None
            for mesh in scene.meshes
        ), 6)


if __name__ == "__main__":
    unittest.main()
