"""Compatibility launcher for the packaged Ordinary Light raster workbench."""

from ordinarylight.integrations.raster_workbench import main


if __name__ == "__main__":
    raise SystemExit(main())
