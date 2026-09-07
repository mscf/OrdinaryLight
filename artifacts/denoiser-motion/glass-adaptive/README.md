# Adaptive radiance rejection remains experimental

Two diagnostic policies operate only on explicitly marked glass (motion.w),
with the opt-in four-frame motion cap enabled in the baseline and candidates.
Both use reactive sigma 1 but require additional evidence before rejection:

- supported: preserve history when the current center sample differs from its
  neighborhood mean by more than half the local luminance deviation (with a
  small brightness-relative floor). This avoids some isolated sample outliers.
- consensus: reject only when at least three and at least 75% of compatible
  neighbors disagree with history in the same brightness direction.

Native captures test camera, glass, and target translation at 320x240 and
640x480, against independent 512-sample final-pose references. At 640x480 the
metrics patch is doubled in each dimension to cover the same image region.
It includes some opaque background at the corners; it is not a glass mask.
Eight 64-sample reference batches are used, with noise still present.

## Decision

Neither adaptive policy is promoted. At both resolutions, both help moving
backgrounds substantially but worsen settled camera/glass noise and error.
For example, at 640x480 consensus cuts target moving error from .1568 to .0355,
but camera settled error rises from .0136 to .0176 and stationary temporal
noise rises from .00116 to .00342. Supported rejection is less aggressive but
retains the same tradeoff. The contact sheets use a common display transform;
640x480 images are reduced to 320x240 panels for display, not for metrics.

A separate uncapped 640x480 control confirms the four-frame cap still improves
moving-glass error from .1313 to .1020 (22%) and settled error from .0307 to
.0157 (49%). Camera and target results remain unchanged by the cap. This checks
two resolutions; it does not establish a resolution-independent threshold.

In all three motion modes and at both resolutions, the opaque top target band
(top one-sixth of image height) is byte-identical between adaptive candidates
and the cap baseline. This is a mixed opaque/glass check, not a broad opaque
scene suite. Production rejection rules and defaults remain unchanged.

## Reproduce

Run from the repository root with PYTHONPATH=. and a working Vulkan desktop.
First generate /tmp/glass-detail-fixed via tools.denoiser_motion.glass_detail.
Then run capture_320.py, capture_640.py, and capture_uncapped_640.py in that order.
The 320 script uses those prior references; the 640 script creates its own;
the uncapped script reuses the 640 references. Outputs are under
/tmp/glass-adaptive, /tmp/glass-adaptive-640, and /tmp/glass-uncapped-640.
Temporary shaders are validated during compilation and never replace packaged
production shaders. Raw sequences remain in the temporary directories.
