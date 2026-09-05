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
