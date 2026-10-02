import unittest

from neural_cost import (
    HardwareSpec,
    Measurement,
    Operation,
    analyze_memory_gap,
    analyze_model_gap,
    estimate_memory,
    profile_model,
)
from neural_cost.adapters.base import FrameworkAdapter


class StaticAdapter(FrameworkAdapter):
    def operations(
        self, model: object, example_inputs: tuple[object, ...]
    ) -> tuple[Operation, ...]:
        return (Operation("fc", "linear", ((2, 4), (4, 3)), (2, 3)),)


class MemoryEstimateTests(unittest.TestCase):
    def test_counts_shared_parameters_once_and_models_training_state(self) -> None:
        operation = Operation(
            "shared_fc",
            "linear",
            ((2, 4), (4, 3)),
            (2, 3),
            attrs={"parameter_bytes": 60, "parameter_id": 7},
        )
        estimate = estimate_memory(
            (operation, operation), training=True, optimizer_state_multiplier=2
        )

        self.assertEqual(estimate.parameter_bytes, 60)
        self.assertEqual(estimate.activation_bytes, 48)
        self.assertEqual(estimate.minimum_peak_activation_bytes, 24)
        self.assertEqual(estimate.conservative_peak_activation_bytes, 48)
        self.assertEqual(estimate.gradient_bytes, 60)
        self.assertEqual(estimate.optimizer_state_bytes, 120)
        self.assertEqual(estimate.training_minimum_bytes, 264)

    def test_memory_gap_reports_allocator_overhead(self) -> None:
        estimate = estimate_memory([Operation("fc", "linear", ((2, 4), (4, 3)), (2, 3))])
        result = analyze_memory_gap(
            estimate,
            Measurement(
                median_seconds=0.001,
                samples_seconds=(0.001,),
                peak_memory_bytes=1_000,
                allocated_memory_bytes=900,
                reserved_memory_bytes=1_200,
            ),
        )

        self.assertEqual(result.observed_peak_bytes, 1_000)
        self.assertGreater(result.overhead_ratio or 0, 1.5)
        self.assertEqual(len(result.findings), 2)

    def test_model_profile_and_combined_gap_work_for_custom_adapters(self) -> None:
        profile = profile_model(object(), (), StaticAdapter(), training=True)
        result = analyze_model_gap(
            profile,
            Measurement(median_seconds=0.001, samples_seconds=(0.001,)),
            HardwareSpec("test", peak_flops=1_000_000, memory_bandwidth=1_000_000),
        )

        self.assertEqual(profile.cost.flops, 48)
        self.assertEqual(profile.memory.gradient_bytes, 48)
        self.assertIsNone(result.memory.observed_peak_bytes)
        self.assertEqual(result.performance.bottleneck, "memory")
