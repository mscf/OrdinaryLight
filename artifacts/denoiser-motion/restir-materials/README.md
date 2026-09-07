# Material-path isolation of native viewer device loss

The same animated 1280x720 Qt viewer harness forces ReSTIR on (four streams,
normal specialization and pacing, mirror guides off), then records actual
pipeline bindings. No shader or core instrumentation is applied.

| Non-mirror materials | Result |
| --- | --- |
| Built-in material (`program=None`), custom mirror retained | 125 frames, clean exit |
| Original custom diffuse, custom mirror retained | Device loss after 4 completed frames |

Both runs bound `custom_primary_pipeline` and `custom_shade_pipeline`, followed
by the same named denoising and presentation stages. These names identify
pipeline roles, not identical shader binaries: changing material programs
changes the generated shader. The built-in material also changes scattering
behavior, so this is an isolation result, not a quality-equivalent fix.

The custom primary shader is compiled from `wavefront_primary.comp` without
passing the ReSTIR specialization switch. Dispatch explicitly chooses it over
the stock primary when present. Thus the earlier generic-specialization clean
run does not establish a fault in an executed production-specialized primary.

Run from the repository root with a working X11/Vulkan desktop:

```bash
QT_QPA_PLATFORM=xcb PYTHONPATH=. MATERIALS=builtin-diffuse .venv/bin/python artifacts/denoiser-motion/restir-materials/check_materials.py
QT_QPA_PLATFORM=xcb PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/restir-materials/check_materials.py
```

The second command intentionally reproduces device loss and exits immediately
on failure. The harness stops after at least 120 frames or 600 timer ticks.
`MODE` must be unset for this paired comparison; other modes are retained from
the earlier diagnostic harness. Logs are retained alongside this document.

Next isolate custom diffuse on the floor, emitter and spheres individually,
and inspect the corresponding generated primary/secondary scattering paths.
A short successful run is not proof of long-term stability. The existing
scene-specific ReSTIR-off workaround remains in place; no root-cause fix or
GPU driver/compiler fault is claimed.

## Surface isolation follow-up

Custom mirror retained in every run; only the listed diffuse groups use the
custom diffuse program. Other diffuse groups use the built-in material.

| Custom diffuse groups | Completed frames | Result |
| --- | --- | --- |
| floor | 120 | Clean exit |
| emitter | 121 | Clean exit |
| spheres | 121 | Clean exit |
| floor-spheres | 122 | Clean exit |
| floor-emitter | 124 | Clean exit |
| emitter-spheres | 124 | Clean exit |

Reproduce with `MATERIALS=floor`, `emitter`, `spheres`, `floor-spheres`,
`floor-emitter`, or `emitter-spheres` using the harness above. Mesh-selection
logging verifies the chosen fixture surfaces. No production shader changes
were made for these runs.

No single group or pair reproduced the failure. The failing all-custom case
also differs in its generated program set: `Scene.material_programs` includes
only custom diffuse and mirror, whereas every mixed case includes the built-in
program. This changes generated dispatch code and potentially compiler
optimization, as well as executed scattering. The results therefore do not
establish that a particular geometric surface is faulty.

Next separate program-set differences from visible scattering changes: retain
all visible custom materials while adding the built-in program to diagnostic
shader generation with stable existing material IDs. Compare generated SPIR-V
and rerun the failing control. Keep the scene workaround until a causal fix is
validated.
