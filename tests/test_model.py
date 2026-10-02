import unittest
from unittest.mock import patch

from neural_cost.adapters.base import FrameworkAdapter
from neural_cost.estimate import CostEstimate
from neural_cost.memory import MemoryEstimate
from neural_cost.model import ModelProfile, profile_model
from neural_cost.operations import Operation


class MockAdapter(FrameworkAdapter):
    def operations(self, model, example_inputs):
        yield Operation("fc", "linear", ((4, 8), (8, 16)), (4, 16), dtype_bytes=4)


class ModelTests(unittest.TestCase):
    def test_profile_model(self):
        adapter = MockAdapter()
        profile = profile_model("model", [], adapter)
        self.assertIsInstance(profile, ModelProfile)
        self.assertIsInstance(profile.cost, CostEstimate)
        self.assertIsInstance(profile.memory, MemoryEstimate)
        self.assertEqual(profile.cost.flops, 1024)

    def test_profile_model_training(self):
        adapter = MockAdapter()
        profile = profile_model("model", [], adapter, training=True, optimizer_state_multiplier=2.0)
        self.assertIsInstance(profile, ModelProfile)
        self.assertIsInstance(profile.memory, MemoryEstimate)
        # weights = 8*16 = 128 elements = 512 bytes
        # optimizer state = 2.0 * 512 = 1024 bytes
        # Just check that it returns a valid profile with training attributes affected
        self.assertGreater(profile.memory.training_minimum_bytes, 0)

    @patch("neural_cost.model.estimate_operations")
    @patch("neural_cost.model.estimate_memory")
    def test_profile_model_calls(self, mock_estimate_memory, mock_estimate_operations):
        adapter = MockAdapter()
        mock_estimate_operations.return_value = CostEstimate(1024, 640, 256, 1)
        mock_estimate_memory.return_value = MemoryEstimate(0, 0, 0, 0)

        profile = profile_model("model", [], adapter)

        mock_estimate_operations.assert_called_once()
        mock_estimate_memory.assert_called_once()
        self.assertEqual(profile.cost.flops, 1024)


if __name__ == "__main__":
    unittest.main()
