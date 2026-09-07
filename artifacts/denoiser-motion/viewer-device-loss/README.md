# Viewer device-loss investigation

Reported invocation:
```bash
.venv/bin/python tools/raster_feature_viewer.py --target wavefront-gi --showcase planar-mirror-guides
```

The user reported VkErrorDeviceLost at vkWaitForFences, followed by a killed
process. Kernel evidence from the current boot:
```text
Sep 07 10:08:03 NVRM: Xid (PCI:0000:01:00): 109, pid=727313,
name=python, channel 0x00000035, errorString CTX SWITCH TIMEOUT, Info 0xdc026
```

This identifies a GPU context-switch timeout, but does not identify the
offending shader or establish that the experimental guide branch caused it.
The guide checkbox defaults off.

A full-resolution Qt diagnostic at 1280x720 completed eight warmup frames and
four captured frames. Its report is retained here. That run uses a fixed
camera, HDR capture, and diagnostic pacing, so it does not reproduce every
normal-startup condition.

A separate bounded invocation of the exact reported command emitted no
device-loss traceback during the test interval. Its eventual exit 137 came
from the explicitly requested timeout/kill escalation, not evidence that the
user's original 'Killed' message had the same cause.

No root-cause fix is claimed. Need to establish whether the original failure
preceded or followed enabling the guide toggle, then isolate frame pacing,
camera motion, and guide mode. Do not treat the earlier small native smoke
test as validation of this failure case.

## Reproduction and scoped workaround

User confirmed the crash occurred immediately, before enabling mirror guides.
A Qt timer harness using normal animation and pacing reproduced device loss
after four completed frames. Waiting for both in-flight frame fences did not
prevent it; that speculative change was removed.

Disabling ReSTIR DI allowed the same startup to complete normally. The viewer
now honors a showcase's wavefront_restir_di setting, and the planar-mirror
showcase sets it false. Other showcases retain their prior default. This is
a scoped workaround; the offending ReSTIR shader/resource interaction remains
unidentified.

At 1280x720 with animation enabled and diagnostic pacing disabled, final runs
completed 131 frames with mirror guides off and 124 with guides on, then closed
cleanly. Reports are retained beside this file. Their live status FPS values
are not GPU performance measurements.

84 focused tests and eight subtests passed. Retry the original command; no
additional launch flag is needed. Root-cause ReSTIR investigation remains
separate from this startup workaround.
