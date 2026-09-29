import unittest

from neural_cost import HardwareSpec, Operation, analyze_gap, estimate_operations
from neural_cost.profiler import Measurement


class EstimateTests(unittest.TestCase):
    def test_linear_cost_and_roofline_analysis(self) -> None:
        estimate = estimate_operations(
            [Operation("fc", "linear", ((4, 8), (8, 16)), (4, 16), dtype_bytes=4)]
        )
        self.assertEqual(estimate.flops, 1024)
        self.assertEqual(estimate.read_bytes, (32 + 128) * 4)
        self.assertEqual(estimate.write_bytes, 64 * 4)
        self.assertAlmostEqual(estimate.arithmetic_intensity, 1024 / ((32 + 128 + 64) * 4))

        result = analyze_gap(
            estimate,
            Measurement(median_seconds=0.01, samples_seconds=(0.01,)),
            HardwareSpec("test", peak_flops=1_000_000, memory_bandwidth=1_000_000),
        )
        self.assertAlmostEqual(result.lower_bound_seconds, 0.001024)
        self.assertAlmostEqual(result.efficiency, 0.1024)
        self.assertEqual(result.bottleneck, "compute")
        self.assertIn("roofline efficiency", result.render())

    def test_conv_and_custom_validation(self) -> None:
        conv = Operation("conv", "conv2d", ((2, 3, 8, 8), (16, 3, 3, 3)), (2, 16, 6, 6))
        self.assertEqual(estimate_operations([conv]).flops, 2 * 2 * 16 * 6 * 6 * 3 * 3 * 3)
        with self.assertRaisesRegex(ValueError, "require attrs"):
            estimate_operations([Operation("unknown", "custom", ((1,),), (1,))])
