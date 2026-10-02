import unittest

from neural_cost import HardwareSpec, Operation, analyze_gap, estimate_operations
from neural_cost.estimate import CostEstimate
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

    def test_matmul_operation(self) -> None:
        op = Operation("mm", "matmul", ((4, 8), (8, 16)), (4, 16), dtype_bytes=4)
        est = estimate_operations([op])
        self.assertEqual(est.flops, 4 * 8 * 16 * 2)

    def test_elementwise_operation(self) -> None:
        op1 = Operation("add", "elementwise", ((10,), (10,)), (10,), dtype_bytes=4)
        est1 = estimate_operations([op1])
        self.assertEqual(est1.flops, 10)  # default 1 flop per element

        op2 = Operation(
            "add",
            "elementwise",
            ((10,), (10,)),
            (10,),
            dtype_bytes=4,
            attrs={"flops_per_element": 3},
        )
        est2 = estimate_operations([op2])
        self.assertEqual(est2.flops, 30)

    def test_custom_operation(self) -> None:
        op = Operation("custom", "custom", ((10,),), (10,), dtype_bytes=4, attrs={"flops": 150})
        est = estimate_operations([op])
        self.assertEqual(est.flops, 150)

    def test_custom_missing_flops_raises(self) -> None:
        op = Operation("custom", "custom", ((10,),), (10,), dtype_bytes=4)
        with self.assertRaises(ValueError):
            estimate_operations([op])

    def test_estimate_operations_aggregation(self) -> None:
        op1 = Operation("add", "elementwise", ((10,),), (10,), dtype_bytes=4)
        op2 = Operation("add", "elementwise", ((20,),), (20,), dtype_bytes=4)
        est = estimate_operations([op1, op2])
        self.assertEqual(est.flops, 30)
        self.assertEqual(est.read_bytes, 30 * 4)
        self.assertEqual(est.write_bytes, 30 * 4)

    def test_arithmetic_intensity_property(self) -> None:
        est = CostEstimate(flops=1000, read_bytes=150, write_bytes=50, operations=1)
        self.assertEqual(est.arithmetic_intensity, 1000.0 / 200.0)

    def test_cost_estimate_addition(self) -> None:
        est1 = CostEstimate(flops=100, read_bytes=50, write_bytes=50, operations=1)
        est2 = CostEstimate(flops=200, read_bytes=100, write_bytes=100, operations=2)
        est3 = est1 + est2
        self.assertEqual(est3.flops, 300)
        self.assertEqual(est3.read_bytes, 150)
        self.assertEqual(est3.write_bytes, 150)
        self.assertEqual(est3.operations, 3)

    def test_dimension_mismatch_raises(self) -> None:
        op = Operation("bad_mm", "linear", ((4, 8), (9, 16)), (4, 16))
        with self.assertRaises(ValueError):
            estimate_operations([op])

    def test_conv2d_channel_mismatch_raises(self) -> None:
        op = Operation("bad_conv", "conv2d", ((2, 3, 8, 8), (16, 4, 3, 3)), (2, 16, 6, 6))
        with self.assertRaises(ValueError):
            estimate_operations([op])

    def test_estimate_fused_operations(self) -> None:
        from neural_cost.estimate import estimate_fused_operations

        # Linear followed by ReLU and LayerNorm
        linear = Operation("fc", "linear", ((32, 128), (128, 64)), (32, 64), dtype_bytes=4)
        relu = Operation("relu", "elementwise", ((32, 64),), (32, 64), dtype_bytes=4)
        norm = Operation("norm", "layernorm", ((32, 64),), (32, 64), dtype_bytes=4)

        unfused = estimate_operations([linear, relu, norm])
        fused = estimate_fused_operations([linear, relu, norm])

        self.assertEqual(fused.flops, unfused.flops)
        self.assertLess(fused.total_bytes, unfused.total_bytes)
        self.assertGreater(fused.arithmetic_intensity, unfused.arithmetic_intensity)
        self.assertEqual(fused.fused_groups_count, 1)
        self.assertGreater(fused.eliminated_bytes, 0)
        self.assertGreater(fused.traffic_reduction_ratio, 0.0)

        # Reads: linear inputs (32*128 + 128*64) * 4 bytes.
        # Writes: final norm output (32*64) * 4 bytes.
        expected_fused_reads = (32 * 128 + 128 * 64) * 4
        expected_fused_writes = 32 * 64 * 4
        self.assertEqual(fused.fused_read_bytes, expected_fused_reads)
        self.assertEqual(fused.fused_write_bytes, expected_fused_writes)

    def test_estimate_fused_operations_empty(self) -> None:
        from neural_cost.estimate import estimate_fused_operations

        fused = estimate_fused_operations([])
        self.assertEqual(fused.flops, 0)
        self.assertEqual(fused.total_bytes, 0)
        self.assertEqual(fused.eliminated_bytes, 0)
        self.assertEqual(fused.fused_groups_count, 0)
