import unittest

import numpy as np

from tests.gates.relax_tessellation_quality import summarize


class RelaxTessellationGateTests(unittest.TestCase):
    def test_projected_edge_metric_detects_dark_triangle_lines(self):
        reference = np.ones((3, 32, 32, 4), np.float32)
        reference[..., 0] = 0.2
        reference[..., 1] = 0.6
        reference[..., 2] = 1.0
        candidate = reference.copy()
        edges = np.zeros((32, 32), bool)
        edges[8:24, 16] = True
        candidate[:, edges, :3] *= 0.4
        result = summarize(reference, candidate, edges)
        self.assertGreater(result["triangle_edge_dark_excess_p95"], 0.1)


if __name__ == "__main__":
    unittest.main()
