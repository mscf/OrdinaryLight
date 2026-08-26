"""Require a GitHub release tag to match the static project version."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1:
        raise SystemExit("usage: check_release_version.py TAG")
    tag = args[0]
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    expected = f"v{version}"
    if tag != expected:
        raise SystemExit(
            f"release tag {tag!r} does not match project version; expected {expected!r}"
        )
    print(f"Release tag {tag} matches ordinarylight {version}")


if __name__ == "__main__":
    main()

