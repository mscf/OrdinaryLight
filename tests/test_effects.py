import unittest

import ordinarylight as ol


class ObjectEffectTests(unittest.TestCase):
    def test_outline_is_an_immutable_validated_effect(self):
        effect = ol.effects.Outline(color=(0.25, 0.5, 1), width=4)
        self.assertIsInstance(effect, ol.effects.ObjectEffect)
        self.assertEqual(effect.color, (0.25, 0.5, 1.0))
        self.assertEqual(effect.width, 4)
        with self.assertRaises(ValueError):
            ol.effects.Outline(color=(1.1, 0, 0))
        with self.assertRaises(ValueError):
            ol.effects.Outline(color=(1, 0))
        with self.assertRaises(TypeError):
            ol.effects.Outline(width=True)
        with self.assertRaises(ValueError):
            ol.effects.Outline(width=0)

    def test_picking_does_not_create_effect_state(self):
        scene = ol.Scene()
        scene.add_mesh(
            ((-1, -1, 0), (1, -1, 0), (0, 1, 0)), ((0, 1, 2),),
        )
        hit = ol.pick(
            scene, ol.PerspectiveCamera((0, 0, 3), (0, 0, 0)),
            (101, 101), (50, 50),
        )
        self.assertIsNotNone(hit)
        self.assertFalse(hasattr(hit, "effect"))


if __name__ == "__main__":
    unittest.main()
