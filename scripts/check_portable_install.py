"""Build wheels and prepare the browser example in an isolated installation.

Requires Python >=3.12, pip, and Git access to the pinned upstream repositories.
The output directory must not exist; it retains wheels, an environment, and the
prepared example for inspection. No native GPU dependency is installed.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ordinaryshade-source", type=Path,
                        help="Existing OrdinaryShade 0.1.0a5 or newer compiler checkout")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("the scientific model dependencies require Python >=3.12")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    example = repo / "examples" / "portable_volume"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"

    def run(*command):
        subprocess.run(command, cwd=root, env=env, check=True)

    # Use the compiler release validated with these shader sources.
    # Other upstream pins are unchanged.
    shade = args.ordinaryshade_source
    if shade is None:
        shade = root / "OrdinaryShade"
        run("git", "clone", "--no-checkout", "https://github.com/mscf/OrdinaryShade.git", str(shade))
        run("git", "-C", str(shade), "checkout", "98d12db5dfef408b32134bf5aa0c4a2798eab247")
    requirements = root / "wheel-requirements.txt"
    requirements.write_text("\n".join(
        line for line in (example / "upstream-requirements.txt").read_text().splitlines()
        if not line.strip().startswith("ordinaryshade @")
    ) + "\n" + str(shade.resolve()) + "\n")
    wheels = root / "wheels"
    wheels.mkdir()
    run(sys.executable, "-m", "pip", "wheel", "--wheel-dir", str(wheels),
        "-r", str(requirements), str(repo))
    run(sys.executable, str(repo / "scripts" / "verify_wheel.py"),
        *map(str, wheels.glob("ordinarylight-*.whl")))
    venv.create(root / "venv", with_pip=True)
    python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(str(python), "-I", "-m", "pip", "install", "--no-index", "--find-links",
        str(wheels), *map(str, wheels.glob("*.whl")))
    run(str(python), "-I", "-m", "pip", "check")
    for name in ("serve.py", "dmc.json"):
        shutil.copy2(example / name, root / name)
    shutil.copy2(example / "upstream-requirements.txt", root)
    run(str(python), "-I", str(root / "serve.py"), "--prepare-only", "--output",
        str(root / "viewer"))
    run(str(python), "-I", "-c", """
import importlib.util
import json
from pathlib import Path
from ordinarylight.portable import PortablePackage
assert importlib.util.find_spec('wgpu') is None, 'unexpected native GPU dependency'
root = Path('viewer')
package = PortablePackage.read(root / 'initial')
assert all(v != 'source-checkout' for v in package.manifest['producers'].values())
assert (root / 'runtime.js').stat().st_size > 0
assert (root / 'index.html').stat().st_size > 0
print(json.dumps(package.manifest['producers'], indent=2))
""")
    print(f"Clean installation and device-free export passed: {root}")


if __name__ == "__main__":
    main()
