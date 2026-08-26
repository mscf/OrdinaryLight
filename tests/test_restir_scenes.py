import math
import types
import unittest

from tests.gates.restir_matrix import CASES, build_command
from ordinarylight.showcases.rooms import SCENES, get_restir_scene


class RestirSceneTests(unittest.TestCase):
    def test_every_scene_has_geometry_and_sampleable_emission(self):
        for name, spec in SCENES.items():
            with self.subTest(scene=name):
                scene = spec.build()
                self.assertGreater(len(scene.meshes), 0)
                self.assertGreater(scene.emissive_triangle_data().shape[0], 0)

    def test_textured_fixture_contains_texture_data(self):
        scene = get_restir_scene("textured").build()
        self.assertGreater(len(scene.textures), 0)

    def test_nested_glass_and_dense_fixtures_stress_scene_structure(self):
        nested = get_restir_scene("nested_glass").build()
        self.assertGreaterEqual(sum(
            mesh.material.transmission > 0.0 for mesh in nested.meshes), 3)
        dense = get_restir_scene("dense").build()
        self.assertGreaterEqual(len(dense.meshes), 45)
        self.assertGreaterEqual(
            sum(len(mesh.indices) for mesh in dense.meshes), 20_000
        )
        statistics = dense.instancing_statistics()
        self.assertEqual(statistics["instance_count"], len(dense.meshes))
        self.assertGreaterEqual(statistics["shared_blas_savings"], 39)
        self.assertLess(
            statistics["unique_geometry_bytes"],
            statistics["expanded_geometry_bytes"] // 10,
        )

    def test_room_presentation_orbits_remain_in_front_of_opening(self):
        for name, spec in SCENES.items():
            with self.subTest(scene=name):
                for phase in (0.0, math.pi * 0.5, math.pi, math.pi * 1.5):
                    angle = spec.presentation_arc_radians * math.sin(phase)
                    camera_z = -spec.orbit_radius * math.cos(angle)
                    self.assertLess(camera_z, -5.0)

    def test_matrix_command_selects_scene_and_motion_case(self):
        args = types.SimpleNamespace(
            width=640, height=360, frames=4,
            reference_samples=8, bounces=6,
            include_generalized=False,
            gate_max_abs_bias=0.0125, gate_max_mae_ratio=1.04,
            generalized_balance_cap=2.0,
            gate_max_generalized_gpu_ratio=2.0,
        )
        command = build_command(args, "diffuse", "moving", "result/prefix")
        self.assertEqual(command[command.index("--scene") + 1], "diffuse")
        self.assertEqual(
            float(command[command.index("--motion-radians") + 1]),
            CASES["moving"],
        )
        modes = command.index("--candidate-modes")
        self.assertEqual(command[modes + 1:modes + 3], ["canonical", "pairwise"])
        fast = build_command(args, "dense", "fast", "result/fast")
        self.assertIn("--require-motion-rejection", fast)


if __name__ == "__main__":
    unittest.main()
