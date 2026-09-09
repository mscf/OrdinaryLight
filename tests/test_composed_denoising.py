"""Runnable example preserves radiance while history grows and resets."""

import os
import numpy as np
import pytest
from examples.composed_denoising import run


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_composed_denoising_history_reset():
    measurements = run()
    np.testing.assert_allclose(
        measurements[:, :3], np.tile([3, 6, 9], (6, 1)), atol=0.02
    )
    np.testing.assert_array_equal(measurements[:, 3], [1, 2, 3, 4, 1, 2])
