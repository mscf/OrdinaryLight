"""Visually compare Ordinary Light's Vulkan and WebGPU raster scene paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

import ordinarylight as ol


def _load_ordinaryshade():
    try:
        import ordinaryshade as osh
        return osh
    except ImportError:
        sibling = Path(__file__).resolve().parents[2] / "ordinaryshade"
        if sibling.is_dir():
            sys.path.insert(0, str(sibling))
            import ordinaryshade as osh
            return osh
        raise RuntimeError(
            "The showcase requires Ordinary Shade; install it or place its "
            "repository beside Ordinary Light."
        )


osh = _load_ordinaryshade()


@osh.structure
class VertexOutput:
    position: osh.builtin(osh.vec4, "position")
    color: osh.location(osh.vec3, 0)


@osh.vertex
def vertex_shader(position: osh.location(osh.vec2, 0)) -> VertexOutput:
    color = osh.vec3(position * 0.5 + osh.vec2(0.5), 0.35)
    return VertexOutput(osh.vec4(position, 0.0, 1.0), color)


@osh.fragment
def fragment_shader(color: osh.location(osh.vec3, 0)) -> osh.location(osh.vec4, 0):
    return osh.vec4(color, 1.0)


def _volume_scene():
    z, y, x = np.mgrid[-1:1:48j, -1:1:48j, -1:1:48j]
    density = np.exp(-6.0 * (x*x + y*y + z*z)).astype(np.float32)
    transfer = ol.Texture1D(np.array(((0,0,0,0),(0.1,0.3,1,0.08),(1,0.2,0.05,0.35)), np.float32))
    material = ol.VolumeMaterial(transfer_function=transfer, density_scale=10.0)
    transform = ol.Transform.translation((-0.75, -0.75, -0.75)) @ ol.Transform.scale(1.5)
    volume = ol.Volume(density, material, transform=transform)
    return ol.Scene(volumes=[volume]), ol.PerspectiveCamera((1.8, 1.2, 2.4), (0, 0, 0))


def _render(backend_name: str, width: int, height: int, mode: str) -> np.ndarray:
    target = "spirv" if backend_name == "vulkan" else "wgsl"
    program = (
        ol.RasterProgram.compile(vertex_shader, fragment_shader, target=target, validate=backend_name == "webgpu")
        if mode == "triangle" else ol.RasterProgram.scene(target=target, validate=backend_name == "webgpu")
    )
    backend_type = (
        ol.VulkanRasterBackend
        if backend_name == "vulkan" else ol.WebGpuRasterBackend
    )
    config = (
        ol.RasterConfig(
            state=ol.RasterState(cull_mode="none", depth_write=False, blend_mode="alpha"),
            direct_lighting=False, volume_slices=64,
        ) if mode == "volume" else
        ol.RasterConfig(state=ol.RasterState(cull_mode="none"))
    )
    backend = backend_type(program, config=config)
    try:
        if mode == "triangle":
            return backend.render(ol.triangle_mesh(), width, height)
        scene, camera = (
            _volume_scene() if mode == "volume" else
            (ol.build_feature_parity_scene(), ol.feature_parity_camera())
        )
        renderer = ol.Renderer(backend=backend)
        hdr = renderer.render(scene, camera, (width, height))
        return np.rint(np.clip(hdr, 0.0, 1.0) * 255.0).astype(np.uint8)
    finally:
        backend.close()


def _comparison(images: list[tuple[str, np.ndarray]]) -> Image.Image:
    margin = 16
    label_height = 42
    width = sum(image.shape[1] for _name, image in images) + margin * (len(images) + 1)
    height = max(image.shape[0] for _name, image in images) + label_height + margin * 2
    result = Image.new("RGB", (width, height), (28, 30, 36))
    draw = ImageDraw.Draw(result)
    x = margin
    for name, pixels in images:
        draw.text((x, margin), name, fill=(240, 240, 244))
        image = Image.fromarray(pixels, "RGBA").convert("RGB")
        result.paste(image, (x, margin + label_height))
        x += image.width + margin
    return result


def _show_qt(image: Image.Image):
    try:
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
    except ImportError as error:
        raise RuntimeError(
            "Qt display requires: pip install 'ordinarylight[qt]'"
        ) from error
    pixels = np.ascontiguousarray(np.asarray(image.convert("RGBA")))
    application = QApplication.instance() or QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Ordinary Light raster showcase")
    label = QLabel()
    qimage = QImage(
        pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()
    label.setPixmap(QPixmap.fromImage(qimage))
    window.setCentralWidget(label)
    window.resize(image.width, image.height)
    window.show()
    # Retain the NumPy owner for bindings that do not eagerly detach QImage.
    window._ordinarylight_pixels = pixels
    return application.exec()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("both", "vulkan", "webgpu"), default="both",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mode", choices=("scene", "volume", "triangle"), default="scene")
    parser.add_argument(
        "--output", type=Path, default=Path("raster_showcase.png"),
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="save the comparison without opening the Qt viewer",
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    selected = (
        ("vulkan", "webgpu") if args.backend == "both" else (args.backend,)
    )
    images = []
    for backend in selected:
        print(f"Rendering {backend}...")
        images.append((backend.upper(), _render(backend, args.width, args.height, args.mode)))
    comparison = _comparison(images)
    comparison.save(args.output)
    print(f"Wrote {args.output.resolve()}")
    if not args.no_window:
        raise SystemExit(_show_qt(comparison))


if __name__ == "__main__":
    main()
