"""Real split tracing feeds motion-aware temporal history without a renderer."""

import os
import numpy as np
import pytest
from examples.composed_motion import run


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_traced_camera_motion_history():
    data = run(frames=4, width=8, height=8)
    assert data["hdr"].shape == (4, 8, 8, 3)
    assert np.all(np.isfinite(data["hdr"]))
    # This checks a rendered surface rather than a cleared or invalid image;
    # radiance varies with the existing estimator and denoiser.
    assert np.all(data["hdr"] > 0)
    np.testing.assert_allclose(data["hdr"].mean(axis=(0, 1, 2)), [2, 1, 0.5], atol=0.2)
    np.testing.assert_allclose(data["motion"][0, 4, 4, :2], [0, 0], atol=0.001)
    np.testing.assert_allclose(
        data["motion"][1:, 4, 4, :2], np.tile([0.25, 0], (3, 1)), atol=0.001
    )
    np.testing.assert_allclose(data["motion"][:, 4, 4, 2], 2, atol=0.001)
    np.testing.assert_array_equal(data["history_length"][:, 4, 4], [1, 2, 3, 4])
