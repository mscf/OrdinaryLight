# Primary transmission capture for custom-material denoising

The generated custom-material primary shader now retains primary geometry for
transmitted samples when denoising signal capture is active. The ordinary
non-denoising shader keeps its previous secondary-capture eligibility. Both
OrdinaryShade and fallback primary call sites are covered. No packaged shader
binaries change: this shader is compiled dynamically for custom materials.

This removes the sample-dependent zero depth that caused most direct glass
pixels to retain raw path-traced output. The same glass comparison now has zero
invalid-depth pixels in the direct interior patch and 32 frames of history.

| Direct-glass metric (log luminance) | Before | After |
| --- | --- | --- |
| Last moving frame RMSE | 0.1283 | 0.0447 |
| After 32 stopped frames RMSE | 0.1420 | 0.0220 |
| Last 16 frames temporal standard deviation | 0.1126 | 0.00226 |

Reference split disagreement is 0.0306. The reflected-glass patch is byte-identical
throughout both sequences, with guides both off and on. The independent raw
reference differs in one channel of one sample-batch pixel by 0.000061;
no exact global bitwise-equivalence claim is made. See the earlier
[glass comparison](../glass-reflection/README.md) for camera, resolution and ROI.

This supplies **front-surface guides**, not refracted-background motion or
multi-interface path tracking. The fixture has little refracted detail;
moving/high-contrast backgrounds and coupled indirect-reuse behavior have not
been validated. The change is limited to dynamically compiled custom-material
primary pipelines, including the showcase glass. It does not expand every
stock/native executor's capture contract.

Validation: 70 focused material/config/viewer tests and ten subtests passed,
including compilation of the transmitted-material capture variant and checks
that the non-denoising eligibility remains unchanged. Native captures are finite.
Run capture.py and analyze.py from the repository root with PYTHONPATH=.; raw
outputs go to /tmp/glass-transmission-capture. The enlarged comparisons retain
one common display transform and nearest-neighbor scaling.
