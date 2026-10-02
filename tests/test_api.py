import unittest

from neural_cost.adapters.base import FrameworkAdapter
from neural_cost.api import estimate_model
from neural_cost.estimate import CostEstimate
from neural_cost.operations import Operation


class MockAdapter(FrameworkAdapter):
    def operations(self, model, example_inputs):
        yield Operation("fc", "linear", ((4, 8), (8, 16)), (4, 16), dtype_bytes=4)


class ApiTests(unittest.TestCase):
    def test_estimate_model(self):
        adapter = MockAdapter()
        estimate = estimate_model("model", [], adapter)
        self.assertIsInstance(estimate, CostEstimate)
        self.assertEqual(estimate.flops, 1024)
        self.assertEqual(estimate.read_bytes, (32 + 128) * 4)
        self.assertEqual(estimate.write_bytes, 64 * 4)


if __name__ == "__main__":
    unittest.main()
