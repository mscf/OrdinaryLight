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

    def test_surface_effects_are_immutable_and_validated(self):
        tint = ol.effects.Tint(color=(0.1, 0.2, 0.3), strength=0.4)
        glow = ol.effects.EmissiveHighlight(color=(1, 0.5, 0), strength=0.7)
        isolation = ol.effects.Isolation(dimming=0.8)
        bounds = ol.effects.BoundingBox(color=(1, 1, 0), width=3)
        xray = ol.effects.XRay(color=(0, 1, 1), strength=0.2, width=2)
        self.assertEqual(tint.strength, 0.4)
        self.assertEqual(glow.color, (1.0, 0.5, 0.0))
        self.assertEqual(isolation.dimming, 0.8)
        self.assertEqual(bounds.width, 3)
        self.assertEqual(xray.strength, 0.2)
        with self.assertRaises(ValueError):
            ol.effects.Tint(strength=-0.1)
        with self.assertRaises(ValueError):
            ol.effects.EmissiveHighlight(color=(2, 0, 0))
        with self.assertRaises(ValueError):
            ol.effects.Isolation(dimming=float("nan"))
        with self.assertRaises(ValueError):
            ol.effects.BoundingBox(width=0)
        with self.assertRaises(ValueError):
            ol.effects.XRay(strength=2)


if __name__ == "__main__":
    unittest.main()
