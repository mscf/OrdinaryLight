# Depth-footprint false-acceptance stress test

**The adaptive policy fails this synthetic same-object occlusion test. Do not
promote it to the native renderer on the basis of the mirror noise reduction.**

The GPU replay runs 36 combinations: three resolutions (64x32, 160x80,
320x160), four signal fixtures, and three policies. These are constructed
two-frame signal inputs, not ray-traced scenes. Constant radiance and matching
normals deliberately prevent photometric/normal rejection from hiding depth
validation failures. The synthetic depth ramp isolates the depth gate; it is
not a physically calibrated perspective-plane capture.

Valid samples differ by 0.4 pixels of the depth ramp. The invalid samples
represent a newly revealed parallel surface 0.12 depth units behind the
previous surface, in a narrow strip. Their correspondence is known invalid
by construction. In one case the identity changes; in the difficult case it
does not. The fixture supplies zero screen motion and expected previous depth
directly to isolate this ambiguity.

| Width | Original false acceptance | Blanket 2% | Adaptive |
| --- | ---: | ---: | ---: |
| 64 | 0% | 100% | 100% |
| 160 | 0% | 100% | 100% |
| 320 | 0% | 100% | 50% |

These percentages apply only to the intentionally invalid same-identity strip,
not to an overall scene population. All policies reject the distinct-identity
strip. All accept valid flat-surface history. At width 64, adaptive and blanket
accept all valid sloped-surface samples while the original accepts 53.3%;
at the higher resolutions all accept the valid slope samples.

This demonstrates a failure mechanism, not a predicted real-scene failure
rate. Depth differences alone cannot distinguish the constructed nearby hidden
surface from permissible depth variation. Next: physically rendered sloped
and self-occluding fixtures, then evaluate geometric plane-distance validation
with camera/jitter information and stronger correspondence evidence where
available. Do not simply increase the tolerance again.

Validation: all 36 GPU runs completed; the existing 35 signal/diagnostic tests
passed. The replay runner explicitly releases device references between cases
to avoid exhausting GPU allocations.

Reproduce:
```bash
.venv/bin/python -m tools.denoiser_motion.validate_footprint --output /tmp/footprint-stress.json
```
