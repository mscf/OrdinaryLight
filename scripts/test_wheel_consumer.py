"""Exercise an installed Ordinary Light wheel as a downstream application."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import struct
import tempfile

import numpy as np

import ordinarylight as ol


def _scene_and_camera():
    scene = ol.Scene()
    scene.add_mesh(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        ((0, 1, 2),),
        ol.Material(base_color=(0.8, 0.25, 0.1)),
    )
    camera = ol.PerspectiveCamera((0, 0, -3), (0, 0, 0))
    return scene, camera


class _ProductsBackend:
    available_outputs = ("color", "depth", "object_id")
    config = None
    device = "consumer-contract"
    last_timings = {"total_ms": 1.0}

    def __init__(self):
        self.resident_scene = None
        self.settings = {}
        self.object_effect = None

    @property
    def capabilities(self):
        return {
            "backend": "consumer-contract",
            "outputs": self.available_outputs,
            "features": {"offscreen_rendering"},
        }

    def render_frame(
        self, scene, camera, width, height, *, samples=None, frame_index=0,
    ):
        return np.zeros((height, width, 4), np.float32)

    def render_products(
        self, scene, camera, width, height, *, outputs,
        samples=None, frame_index=0,
    ):
        products = {
            "color": np.zeros((height, width, 4), np.float32),
            "depth": np.full((height, width), 2.0, np.float32),
            "object_id": np.full((height, width), 7, np.uint32),
        }
        return {name: products[name] for name in outputs}

    def close(self):
        pass

    def replace_scene(self, scene):
        self.resident_scene = scene

    def reconfigure(self, **changes):
        self.settings.update(changes)
        return dict(self.settings)

    def apply_object_effect(self, scene, reference, effect):
        self.object_effect = (scene.object_triangle_range(reference), effect)

    def set_object_effects(self, scene, bindings):
        self.object_effect = tuple(
            (scene.object_triangle_range(reference), effect)
            for reference, effect in bindings
        )
        return self.object_effect

    def clear_object_effect(self):
        self.object_effect = None


def _write_minimal_gltf(path):
    positions = struct.pack("<9f", -1, -1, 0, 1, -1, 0, 0, 1, 0)
    indices = struct.pack("<3H", 0, 1, 2)
    payload = positions + indices
    encoded = base64.b64encode(payload).decode("ascii")
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{
            "byteLength": len(payload),
            "uri": f"data:application/octet-stream;base64,{encoded}",
        }],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(indices),
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [-1, -1, 0],
                "max": [1, 1, 0],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
        ],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0}, "indices": 1,
        }]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    path.write_text(json.dumps(document))


async def _render_awaitable(scene, camera):
    with ol.Renderer(backend=_ProductsBackend()) as renderer:
        return await renderer.render_async(scene, camera, (8, 6))


def main():
    scene, camera = _scene_and_camera()

    output = np.empty((6, 8, 4), np.float32)
    with ol.Renderer(
        backend=ol.backends.ReferenceBackend(samples_per_pixel=1, seed=7)
    ) as renderer:
        result = renderer.render(scene, camera, (8, 6), out=output)
        assert result is output
        assert result.shape == (6, 8, 4) and result.dtype == np.float32

    products_backend = _ProductsBackend()
    with ol.Renderer(backend=products_backend) as renderer:
        renderer.replace_scene(scene)
        renderer.reconfigure(samples_per_pixel=2)
        frame = renderer.render(
            scene, camera, (8, 6), outputs=("color", "depth", "object_id")
        )
        assert isinstance(frame, ol.RenderFrame)
        assert frame.depth.shape == (6, 8)
        assert frame.object_id.dtype == np.uint32
        assert renderer.capabilities.supports_output("depth")
        assert products_backend.resident_scene is scene
        assert products_backend.settings == {"samples_per_pixel": 2}
        effect = ol.effects.Outline(color=(0.2, 0.4, 0.8), width=3)
        tint = ol.effects.Tint(color=(0.8, 0.2, 0.1), strength=0.4)
        renderer.set_object_effects(
            scene, ((scene.meshes[0], effect), (scene.meshes[0], tint)),
        )
        assert products_backend.object_effect == (
            ((0, 1), effect), ((0, 1), tint),
        )
        renderer.clear_object_effect()
        assert products_backend.object_effect is None

    asynchronous = asyncio.run(_render_awaitable(scene, camera))
    assert asynchronous.shape == (6, 8, 4)
    mapping = ol.ViewportMapping(
        (16, 12), framebuffer_size=(8, 6), render_size=(8, 6),
    )
    assert mapping.map_pixel((8, 6)) == (4.0, 3.0)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "consumer.gltf"
        _write_minimal_gltf(path)
        loaded = ol.loaders.load(path)
        assert len(loaded.meshes) == 1

    print(json.dumps({
        "package": str(Path(ol.__file__).resolve()),
        "hdr_shape": list(output.shape),
        "named_outputs": ["color", "depth", "object_id"],
        "async": True,
        "resident_transitions": True,
        "object_effects": True,
        "gltf": True,
    }, indent=2))


if __name__ == "__main__":
    main()
