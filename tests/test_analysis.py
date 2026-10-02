import unittest

from neural_cost.analysis import (
    GapAnalysis,
    MemoryGapAnalysis,
    ModelGapAnalysis,
    analyze_gap,
    analyze_memory_gap,
    analyze_model_gap,
)
from neural_cost.estimate import CostEstimate
from neural_cost.hardware import HardwareSpec
from neural_cost.memory import MemoryEstimate
from neural_cost.model import ModelProfile
from neural_cost.profiler import Measurement


class AnalysisTests(unittest.TestCase):
    def test_efficiency_clamping(self):
        estimate = CostEstimate(flops=1000, read_bytes=100, write_bytes=100, operations=1)
        measurement = Measurement(median_seconds=0.0001, samples_seconds=(0.0001,))
        # hardware peak_flops = 10000 -> compute bound = 0.1s
        hardware = HardwareSpec("test", peak_flops=10000, memory_bandwidth=1000000)
        result = analyze_gap(estimate, measurement, hardware)
        self.assertEqual(result.efficiency, 1.0)
        self.assertEqual(result.bottleneck, "compute")

    def test_efficiency_less_than_one(self):
        estimate = CostEstimate(flops=1000, read_bytes=100, write_bytes=100, operations=1)
        measurement = Measurement(median_seconds=1.0, samples_seconds=(1.0,))
        hardware = HardwareSpec("test", peak_flops=10000, memory_bandwidth=1000000)
        result = analyze_gap(estimate, measurement, hardware)
        self.assertLess(result.efficiency, 1.0)
        self.assertEqual(result.bottleneck, "compute")

    def test_bottleneck_classification_memory(self):
        estimate = CostEstimate(flops=10, read_bytes=1000, write_bytes=1000, operations=1)
        measurement = Measurement(median_seconds=1.0, samples_seconds=(1.0,))
        hardware = HardwareSpec("test", peak_flops=100000, memory_bandwidth=1000)
        result = analyze_gap(estimate, measurement, hardware)
        self.assertEqual(result.bottleneck, "memory")
        self.assertTrue(any("Memory-bound" in f for f in result.findings))

    def test_findings_generation_large_gap(self):
        estimate = CostEstimate(flops=1000, read_bytes=100, write_bytes=100, operations=1)
        measurement = Measurement(median_seconds=10.0, samples_seconds=(10.0,))
        hardware = HardwareSpec("test", peak_flops=10000, memory_bandwidth=1000000)
        result = analyze_gap(estimate, measurement, hardware)
        self.assertTrue(any("Large roofline gap" in f for f in result.findings))
        self.assertTrue(any("Compute-bound" in f for f in result.findings))

    def test_findings_generation_capacity_exceeded(self):
        estimate = CostEstimate(flops=1000, read_bytes=1000, write_bytes=1000, operations=1)
        measurement = Measurement(median_seconds=1.0, samples_seconds=(1.0,))
        hardware = HardwareSpec("test", peak_flops=10000, memory_bandwidth=1000000, memory_capacity=1000)
        result = analyze_gap(estimate, measurement, hardware)
        self.assertTrue(any("exceeds device memory capacity" in f for f in result.findings))

    def test_render_output(self):
        estimate = CostEstimate(flops=1000, read_bytes=100, write_bytes=100, operations=1)
        measurement = Measurement(median_seconds=0.1, samples_seconds=(0.1,))
        hardware = HardwareSpec("test", peak_flops=10000, memory_bandwidth=1000000)
        result = analyze_gap(estimate, measurement, hardware)
        rendered = result.render()
        self.assertTrue(len(rendered) > 0)
        self.assertIn("Neural cost gap analysis", rendered)
        self.assertIn("roofline efficiency", rendered)
        self.assertIn("GFLOP/s", rendered)

    def test_analyze_memory_gap_none_telemetry(self):
        estimate = MemoryEstimate(parameter_bytes=100, activation_bytes=100, minimum_peak_activation_bytes=50, conservative_peak_activation_bytes=100)
        measurement = Measurement(median_seconds=1.0, samples_seconds=(1.0,), 
                                  peak_memory_bytes=None, allocated_memory_bytes=None, reserved_memory_bytes=None)
        result = analyze_memory_gap(estimate, measurement)
        self.assertTrue(any("No allocator telemetry" in f for f in result.findings))
        self.assertIsNone(result.overhead_ratio)

    def test_analyze_memory_gap_high_overhead_and_reserved(self):
        estimate = MemoryEstimate(parameter_bytes=100, activation_bytes=100, minimum_peak_activation_bytes=50, conservative_peak_activation_bytes=100)
        measurement = Measurement(median_seconds=1.0, samples_seconds=(1.0,),
                                  peak_memory_bytes=400, allocated_memory_bytes=400, reserved_memory_bytes=500)
        result = analyze_memory_gap(estimate, measurement)
        self.assertTrue(any("exceeds conservative tensor storage" in f for f in result.findings))
        self.assertTrue(any("Reserved memory exceeds allocated memory" in f for f in result.findings))
        self.assertEqual(result.overhead_ratio, 2.0)

    def test_analyze_gap_cache_residency(self):
        from neural_cost.hardware import CacheSpec

        l2 = CacheSpec("L2", bandwidth=10000000.0, capacity=1000)
        hardware = HardwareSpec("test", peak_flops=10000, memory_bandwidth=1000000, caches=(l2,))
        estimate = CostEstimate(flops=1000, read_bytes=100, write_bytes=100, operations=1)
        measurement = Measurement(median_seconds=0.01, samples_seconds=(0.01,))
        result = analyze_gap(estimate, measurement, hardware)
        self.assertEqual(result.resident_cache_level, "L2")
        self.assertIsNotNone(result.cache_bound_seconds)
        self.assertIsNotNone(result.cache_efficiency)
        self.assertTrue(any("Cache-resident" in f for f in result.findings))

        rendered = result.render()
        self.assertIn("cache residency: L2", rendered)

    def test_analyze_model_gap(self):
        cost_est = CostEstimate(flops=1000, read_bytes=100, write_bytes=100, operations=1)
        mem_est = MemoryEstimate(parameter_bytes=100, activation_bytes=100, minimum_peak_activation_bytes=50, conservative_peak_activation_bytes=100)
        profile = ModelProfile(cost=cost_est, memory=mem_est)
        measurement = Measurement(median_seconds=0.1, samples_seconds=(0.1,))
        hardware = HardwareSpec("test", peak_flops=10000, memory_bandwidth=1000000)
        result = analyze_model_gap(profile, measurement, hardware)
        self.assertIsInstance(result, ModelGapAnalysis)
        self.assertIsInstance(result.performance, GapAnalysis)
        self.assertIsInstance(result.memory, MemoryGapAnalysis)

if __name__ == "__main__":
    unittest.main()

