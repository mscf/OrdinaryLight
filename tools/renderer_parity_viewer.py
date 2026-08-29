"""Render and inspect Ordinary Light raster/GI visual parity in Qt."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ordinarylight as ol
from ordinarylight.outputs import to_sdr
from tests.gates.renderer_visual_parity import _render_gi, _render_raster


def _comparison(gi, raster, metrics):
    gi_sdr = to_sdr(gi.color)
    raster_sdr = to_sdr(
        raster.color, exposure=metrics["exposure_scale"],
    )
    height, width = gi_sdr.shape[:2]
    header = 58
    canvas = Image.new("RGB", (width * 2, height + header), (28, 30, 36))
    canvas.paste(Image.fromarray(gi_sdr, "RGB"), (0, header))
    canvas.paste(Image.fromarray(raster_sdr, "RGB"), (width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "GI REFERENCE", fill=(245, 245, 248))
    draw.text((width + 12, 10), "RASTER (EXPOSURE MATCHED)", fill=(245, 245, 248))
    summary = (
        f"color RMSE {metrics['log_color_rmse']:.3f}  |  "
        f"edge {metrics['edge_correlation']:.3f}  |  "
        f"coverage {metrics['coverage_iou']:.3f}  |  "
        f"exposure {metrics['exposure_scale']:.2f}x"
    )
    draw.text((12, 34), summary, fill=(190, 198, 212))
    return canvas


def _show(image):
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QScrollArea
    except ImportError as error:
        raise RuntimeError(
            "Qt display requires: pip install 'ordinarylight[qt]'"
        ) from error
    pixels = np.ascontiguousarray(np.asarray(image.convert("RGBA")))
    application = QApplication.instance() or QApplication(sys.argv)
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    qimage = QImage(
        pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()
    label.setPixmap(QPixmap.fromImage(qimage))
    scroll = QScrollArea()
    scroll.setWidget(label)
    scroll.setWidgetResizable(True)
    window = QMainWindow()
    window.setWindowTitle("Ordinary Light — raster / GI parity")
    window.setCentralWidget(scroll)
    window.resize(min(image.width, 1800), min(image.height, 1000))
    window._ordinarylight_pixels = pixels
    window.show()
    return application.exec()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_renderer_parity.png"),
    )
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.samples < 1:
        parser.error("dimensions and samples must be positive")
    extent = (args.width, args.height)
    scene = ol.build_feature_parity_scene()
    camera = ol.feature_parity_camera()
    print("Rendering GI reference...")
    gi = _render_gi(scene, camera, extent, args.samples)
    print("Rendering raster candidate...")
    raster = _render_raster(scene, camera, extent)
    metrics = ol.renderer_visual_metrics(
        gi.color, raster.color,
        reference_mask=gi.object_id > 0,
        candidate_mask=raster.object_id > 0,
    )
    comparison = _comparison(gi, raster, metrics)
    comparison.save(args.output)
    print(f"Wrote {args.output.resolve()}")
    print(
        f"color={metrics['log_color_rmse']:.4f} "
        f"edge={metrics['edge_correlation']:.4f} "
        f"coverage={metrics['coverage_iou']:.4f} "
        f"exposure={metrics['exposure_scale']:.4f}x"
    )
    if not args.no_window:
        raise SystemExit(_show(comparison))


if __name__ == "__main__":
    main()
