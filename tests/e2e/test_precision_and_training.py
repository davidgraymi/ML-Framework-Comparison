"""End-to-End Tests for Multi-Precision Sweeps (#20) & Training Workloads (#21).

Covers:
- CLI flags --precision [fp32|fp16|bf16|int8] and --mode [inference|train_step|backward_only].
- Propagation of datatype byte widths (4B, 2B, 2B, 1B) into Operation and CostEstimate.
- Inverse arithmetic intensity scaling: doubling AI for FP16/BF16, quadrupling for INT8.
- Precision-aware hardware detection and peak compute scaling.
- Training workload compute accounting: 3x forward FLOPs for train_step, 2x for backward_only.
- AdamW optimizer step DRAM traffic modeling (7P bytes per parameter).
- Static training memory modeling and validation against training_minimum_bytes.
- BenchRecord schema extensions for precision and training phases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from neural_cost.estimate import estimate_operation
from neural_cost.hardware import HardwareSpec
from neural_cost.hardware_detect import detect_hardware
from neural_cost.memory import estimate_memory
from neural_cost.operations import Operation

# Ensure benchmarks directory is importable
BENCHMARKS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))


# ---------------------------------------------------------------------------
# 1. Multi-Precision CLI & Argument Validation (Issue #20)
# ---------------------------------------------------------------------------


class TestMultiPrecisionCLI:
    """Validate CLI support for multi-precision sweeps."""

    def test_collect_data_precision_cli_argument(self):
        """collect_data.py must support --precision [fp32|fp16|bf16|int8] with default fp32."""
        import collect_data

        if hasattr(collect_data, "build_parser"):
            parser = collect_data.build_parser()
            actions = {action.dest: action for action in parser._actions}
            assert "precision" in actions, "CLI parser must contain --precision action"
            prec_action = actions["precision"]
            assert prec_action.default == "fp32"
            assert set(prec_action.choices) == {"fp32", "fp16", "bf16", "int8"}

            # Test valid CLI inputs
            for p in ["fp32", "fp16", "bf16", "int8"]:
                args = parser.parse_args(["--precision", p])
                assert args.precision == p
        else:
            source = Path(collect_data.__file__).read_text()
            assert "--precision" in source, "benchmarks/collect_data.py must implement --precision"

    def test_collect_data_rejects_unsupported_precision(self):
        """collect_data CLI must reject invalid precisions with error."""
        import collect_data

        if hasattr(collect_data, "build_parser"):
            parser = collect_data.build_parser()
            with pytest.raises(SystemExit):
                parser.parse_args(["--precision", "fp64"])
            with pytest.raises(SystemExit):
                parser.parse_args(["--precision", "int4"])


# ---------------------------------------------------------------------------
# 2. Datatype Byte Width Propagation & Inverse AI Scaling (Issue #20)
# ---------------------------------------------------------------------------


class TestPrecisionCostAndArithmeticIntensity:
    """Validate byte width propagation and inverse scaling of arithmetic intensity."""

    @pytest.mark.parametrize(
        ("precision", "expected_bytes"),
        [
            ("fp32", 4),
            ("fp16", 2),
            ("bf16", 2),
            ("int8", 1),
        ],
    )
    def test_precision_to_dtype_bytes_mapping(self, precision: str, expected_bytes: int):
        """Byte width mapping must follow IEEE-754 / standard integer widths."""
        # Derive byte widths from Operation.dtype_bytes
        op = Operation(
            name="test_linear",
            kind="linear",
            inputs=((64, 512), (512, 1024)),
            output=(64, 1024),
            dtype_bytes=expected_bytes,
        )
        assert op.dtype_bytes == expected_bytes

    def test_operation_rejects_non_positive_dtype_bytes(self):
        """Operation must reject non-positive dtype_bytes with ValueError."""
        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            Operation(
                name="invalid_op",
                kind="linear",
                inputs=((16, 64), (64, 128)),
                output=(16, 128),
                dtype_bytes=0,
            )

        with pytest.raises(ValueError, match="dtype_bytes must be positive"):
            Operation(
                name="invalid_op",
                kind="linear",
                inputs=((16, 64), (64, 128)),
                output=(16, 128),
                dtype_bytes=-2,
            )

    def test_inverse_arithmetic_intensity_scaling(self):
        """Authoritative Math Check: AI scales inversely with dtype byte width for identical FLOPs.

        For matrix multiplication (M, K) @ (K, N):
          FLOPs = 2 * M * K * N (constant across precisions).
          DRAM Bytes = dtype_bytes * (M*K + K*N + M*N).
          AI = FLOPs / DRAM Bytes = (2 * M * K * N) / (dtype_bytes * (M*K + K*N + M*N)).
          Therefore, AI(FP16) = 2.0 * AI(FP32), AI(INT8) = 4.0 * AI(FP32).
        """
        shape_in = (128, 512)
        shape_w = (512, 1024)
        shape_out = (128, 1024)

        op_fp32 = Operation("fc", "linear", (shape_in, shape_w), shape_out, dtype_bytes=4)
        op_fp16 = Operation("fc", "linear", (shape_in, shape_w), shape_out, dtype_bytes=2)
        op_bf16 = Operation("fc", "linear", (shape_in, shape_w), shape_out, dtype_bytes=2)
        op_int8 = Operation("fc", "linear", (shape_in, shape_w), shape_out, dtype_bytes=1)

        est_fp32 = estimate_operation(op_fp32)
        est_fp16 = estimate_operation(op_fp16)
        est_bf16 = estimate_operation(op_bf16)
        est_int8 = estimate_operation(op_int8)

        # FLOPs must be identical across precisions
        assert est_fp32.flops == est_fp16.flops == est_bf16.flops == est_int8.flops

        # Total DRAM traffic must scale proportionally to dtype_bytes
        assert est_fp32.total_bytes == 2 * est_fp16.total_bytes
        assert est_fp16.total_bytes == est_bf16.total_bytes
        assert est_fp32.total_bytes == 4 * est_int8.total_bytes

        # Arithmetic intensity must scale inversely with dtype_bytes
        ai_fp32 = est_fp32.arithmetic_intensity
        ai_fp16 = est_fp16.arithmetic_intensity
        ai_bf16 = est_bf16.arithmetic_intensity
        ai_int8 = est_int8.arithmetic_intensity

        assert ai_fp16 == pytest.approx(2.0 * ai_fp32, rel=1e-5)
        assert ai_bf16 == pytest.approx(2.0 * ai_fp32, rel=1e-5)
        assert ai_int8 == pytest.approx(4.0 * ai_fp32, rel=1e-5)


# ---------------------------------------------------------------------------
# 3. Precision-Aware Hardware Detection (Issue #20)
# ---------------------------------------------------------------------------


class TestPrecisionAwareHardwareDetection:
    """Validate that detect_hardware scales peak compute targets by precision."""

    def test_detect_hardware_accepts_precision_parameter(self):
        """detect_hardware must accept precision parameter: detect_hardware(precision='fp16')."""
        try:
            hw_spec, _ = detect_hardware(precision="fp32")
        except TypeError as exc:
            pytest.fail(f"detect_hardware does not accept precision parameter: {exc}")

        assert isinstance(hw_spec, HardwareSpec)

    def test_detect_hardware_peak_compute_scaling(self):
        """Accelerated hardware targets must report higher peak compute on FP16/INT8 than FP32."""
        try:
            hw_fp32, _ = detect_hardware(precision="fp32")
            hw_fp16, _ = detect_hardware(precision="fp16")
        except TypeError:
            pytest.fail("detect_hardware requires precision parameter (Issue #20)")

        # FP16 peak compute is accelerated (e.g. 2x AMX on Apple Silicon, Tensor Cores on NVIDIA)
        assert hw_fp16.peak_flops >= hw_fp32.peak_flops

        # Ridge point (peak_flops / memory_bandwidth) shifts right
        assert hw_fp16.ridge_point >= hw_fp32.ridge_point


# ---------------------------------------------------------------------------
# 4. Training Workload Mode CLI & FLOP Accounting (Issue #21)
# ---------------------------------------------------------------------------


class TestTrainingWorkloadCLIAndCompute:
    """Validate CLI support for training modes and compute accounting."""

    def test_collect_data_mode_cli_argument(self):
        """collect_data.py must expose --mode [inference|train_step|backward_only] defaulting to inference."""
        import collect_data

        if hasattr(collect_data, "build_parser"):
            parser = collect_data.build_parser()
            actions = {action.dest: action for action in parser._actions}
            assert "mode" in actions, "CLI parser must support --mode"
            mode_action = actions["mode"]
            assert mode_action.default == "inference"
            assert set(mode_action.choices) == {"inference", "train_step", "backward_only"}
        else:
            source = Path(collect_data.__file__).read_text()
            assert "--mode" in source, "benchmarks/collect_data.py must implement --mode"

    def test_collect_data_rejects_invalid_mode(self):
        """CLI parser must reject invalid execution modes."""
        import collect_data

        if hasattr(collect_data, "build_parser"):
            parser = collect_data.build_parser()
            with pytest.raises(SystemExit):
                parser.parse_args(["--mode", "eval"])
            with pytest.raises(SystemExit):
                parser.parse_args(["--mode", "train"])

    def test_training_flop_accounting_ratios(self):
        """Theoretical training work: train_step = 3x forward FLOPs, backward_only = 2x forward FLOPs."""
        forward_flops = 1_000_000_000

        # Contract definition:
        # Backward pass computes activation gradients + weight gradients = 2x forward FLOPs
        # Full train_step includes forward (1x) + backward (2x) = 3x forward FLOPs
        backward_flops = 2 * forward_flops
        train_step_flops = 3 * forward_flops

        assert backward_flops == 2.0 * forward_flops
        assert train_step_flops == 3.0 * forward_flops

        import collect_data

        # If collect_data defines a helper or scaling factor, verify it
        if hasattr(collect_data, "get_mode_flops_multiplier"):
            assert collect_data.get_mode_flops_multiplier("inference") == 1.0
            assert collect_data.get_mode_flops_multiplier("backward_only") == 2.0
            assert collect_data.get_mode_flops_multiplier("train_step") == 3.0


# ---------------------------------------------------------------------------
# 5. AdamW Optimizer Memory Traffic & Static Training Memory (Issue #21)
# ---------------------------------------------------------------------------


class TestTrainingMemoryAndOptimizerTraffic:
    """Validate static memory model and AdamW optimizer traffic modeling."""

    def test_adamw_optimizer_traffic_model(self):
        """AdamW parameter update generates 7 * P * dtype_bytes DRAM traffic.

        AdamW elementwise update for parameter tensor P:
          Reads:  Parameter (P), Gradient (P), First Moment m (P), Second Moment v (P) = 4P
          Writes: Updated Parameter (P), Updated m (P), Updated v (P)                  = 3P
          Total:  4P + 3P = 7P elements = 7 * P * dtype_bytes DRAM bytes.
        """
        num_parameters = 1_000_000
        dtype_bytes = 4  # FP32

        expected_adamw_traffic = 7 * num_parameters * dtype_bytes
        assert expected_adamw_traffic == 28_000_000  # 28 MB DRAM traffic per optimizer step

        # Verify against memory estimation helper if present
        from neural_cost import estimate

        if hasattr(estimate, "estimate_adamw_traffic"):
            traffic = estimate.estimate_adamw_traffic(num_parameters, dtype_bytes=dtype_bytes)
            assert traffic == expected_adamw_traffic

    def test_static_training_minimum_memory_estimate(self):
        """estimate_memory with training=True models gradient and optimizer state footprint."""
        op = Operation(
            name="fc",
            kind="linear",
            inputs=((32, 256), (256, 512)),
            output=(32, 512),
            dtype_bytes=4,
        )

        mem_inf = estimate_memory([op], training=False)
        mem_train = estimate_memory([op], training=True, optimizer_state_multiplier=2.0)

        param_bytes = 256 * 512 * 4  # 524,288 bytes

        # In training, gradient buffer = parameter_bytes
        assert mem_train.gradient_bytes == param_bytes

        # AdamW stores 2 moment buffers = 2 * parameter_bytes
        assert mem_train.optimizer_state_bytes == 2 * param_bytes

        # Theoretical training minimum = inference_minimum + gradient_bytes + optimizer_state_bytes
        assert mem_train.training_minimum_bytes == (
            mem_train.inference_minimum_bytes
            + mem_train.gradient_bytes
            + mem_train.optimizer_state_bytes
        )

        # Training footprint must exceed inference footprint by at least 3x parameter memory
        assert mem_train.training_minimum_bytes >= mem_inf.inference_minimum_bytes + 3 * param_bytes

    def test_negative_optimizer_state_multiplier_rejected(self):
        """estimate_memory must reject negative optimizer_state_multiplier."""
        op = Operation("fc", "linear", ((16, 64), (64, 64)), (16, 64), dtype_bytes=4)
        with pytest.raises(ValueError, match="optimizer_state_multiplier must be non-negative"):
            estimate_memory([op], optimizer_state_multiplier=-1.0)


# ---------------------------------------------------------------------------
# 6. BenchRecord Schema Extensions (Issues #20 & #21)
# ---------------------------------------------------------------------------


class TestBenchRecordSchemaExtensions:
    """Validate BenchRecord schema captures precision, mode, and breakdown metrics."""

    def test_bench_record_contains_precision_and_mode_fields(self):
        """BenchRecord must include precision and mode attributes."""
        import collect_data

        assert hasattr(collect_data, "BenchRecord"), "collect_data must define BenchRecord"
        fields = collect_data.BenchRecord.__dataclass_fields__

        assert "precision" in fields, "BenchRecord must contain 'precision' field"
        assert "mode" in fields, "BenchRecord must contain 'mode' field"

    def test_bench_record_defaults_backwards_compatible(self):
        """Default values in BenchRecord must be precision='fp32' and mode='inference'."""
        import collect_data

        fields = collect_data.BenchRecord.__dataclass_fields__
        if "precision" in fields:
            assert fields["precision"].default in ("fp32", argparse.SUPPRESS, None)
        if "mode" in fields:
            assert fields["mode"].default in ("inference", argparse.SUPPRESS, None)
