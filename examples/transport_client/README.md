# External transport client

This separately installable package depends only on OrdinaryLight's public APIs.
It renders three rows of application-indexed samples: an exact two-diffuse-bounce
cavity result, a refracting/absorbing SDF sphere, and a triangle glass box.
The sphere reads a declared application storage buffer. Two surface samples per
diffuse output are reduced explicitly using SampleReduction, and their GPU
input allocation is updated between frames without recreating integrators.
The application owns sample identities and display layout. There are no voxel
reconstruction or artistic averaging algorithms here.

Install the current OrdinaryLight source (the new APIs are unreleased), then this
package into the same environment:

```bash
python -m pip install -e /path/to/wave-render
python -m pip install -e /path/to/wave-render/examples/transport_client
ordinarylight-transport-demo --output /tmp/transport.png
# Optional GLFW native presentation:
ordinarylight-transport-demo --present --frames 16 --output /tmp/transport.png
```

Custom intersection/integrator compilation needs `glslangValidator` or `glslc`.
Native presentation requires GLFW, available with this package's `present` extra.
The final PNG/JSON/NPZ export intentionally reads the GPU results; live native
presentation uses the resident HDR image. The diffuse row deliberately stops
after two bounces, so its truncation counter is expected and its value is checked
against the finite cavity series. Nonzero invalid-path status raises an error.

Version 0.3.0 composes transport, reduction, HDR resolve, tone mapping and optional
presentation into an application graph. It keeps a persistent tone-map target
and uses a two-frame ring. CPU sample updates in the cavity demo intentionally
remain synchronized uploads.

The second entry point exercises GPU animation in a fixed sparse grid:

```bash
ordinarylight-animated-grid --frames 8 --output /tmp/grid.png
ordinarylight-animated-grid --present --frames 120 --output /tmp/grid.png
# From the source package:
python -m ordinarylight_transport_demo.animated_grid --frames 8
```

OrdinaryShade generates occupancy and exports access reflection. OrdinaryLight
orders the producer, custom-geometry updates, transport, reset/resolve, tone
mapping and presentation. Chunk capacity grows once; slots are activated and
removed without moving grid cells. This is a small execution fixture, not a
voxel authoring or reconstruction engine. Final RGB values are checked against
an exact emission/environment reference, and only final verification/export
reads back GPU results. Build from the current unreleased OrdinaryLight source.
