"""Create synchronized, lossless animated PNG comparisons from mirror replays."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def animate(capture, output):
    with np.load(capture) as data:
        keys = ("reference", "primary", "reflected", "depth_footprint")
        count, height, width, _ = data["reference"].shape
        frames = []
        for index in range(count):
            canvas = Image.new("RGB", (4 * width, height + 40))
            draw = ImageDraw.Draw(canvas)
            phase = "initial" if index < 2 else ("moving" if index < 6 else "settling")
            draw.text((3, 2), f"Frame {index}: {phase} (slowed diagnostic sequence)", fill="white")
            for col, key in enumerate(keys):
                draw.text((col * width + 3, 20), key, fill="white")
                rgb = np.maximum(data[key][index], 0)
                rgb = np.clip(rgb / (1 + rgb), 0, 1) ** (1 / 2.2)
                canvas.paste(Image.fromarray((rgb * 255).astype(np.uint8)),
                             (col * width, 40))
            frames.append(canvas.resize((8 * width, 2 * (height + 40)), Image.Resampling.NEAREST))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, format="PNG", save_all=True, append_images=frames[1:],
                   duration=[300] * (count - 1) + [1200], loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    animate(args.capture, args.output)
