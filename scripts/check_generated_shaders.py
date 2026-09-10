"""Verify every text shader generator without requiring native GPU tools.

Raster/denoiser binary artifact checks remain in their dedicated generators;
those also require glslang and Naga and run in the packaging CI job.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NATIVE_ARTIFACTS = {'generate_raster_artifacts.py', 'generate_denoiser_artifacts.py'}


def main():
    for generator in sorted((ROOT / 'scripts').glob('generate_*.py')):
        if generator.name not in NATIVE_ARTIFACTS:
            subprocess.run([sys.executable, str(generator), '--check'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
