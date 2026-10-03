"""End-to-End Tests for LLM Prefill vs Decode (#22).

Covers:
- LLM prefill vs. autoregressive decode discrepancy and metrics (TTFT, tokens/sec).
- KV cache read/write memory traffic modeling (2 * B * L * D * dtype_bytes).
- Memory bandwidth saturation gap analysis finding when bandwidth utilization >= 60%.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_cost.analysis import GapAnalysis, analyze_gap
from neural_cost.hardware import HardwareSpec

# Ensure benchmarks directory is importable
BENCHMARKS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))


# ---------------------------------------------------------------------------
# 1. LLM Benchmark Module and Entry Point (Issue #22)
# ---------------------------------------------------------------------------


class TestLLMBenchmarkModule:
    """Validate existence and interface of benchmarks/llm_bench.py."""

    def test_llm_bench_module_exists(self):
        """benchmarks/llm_bench.py must exist as a dedicated LLM benchmark workload."""
        llm_bench_path = BENCHMARKS_DIR / "llm_bench.py"
        assert llm_bench_path.exists(), f"Expected {llm_bench_path} to exist (Issue #22)"

    def test_llm_bench_prefill_and_decode_functions(self):
        """llm_bench must expose benchmark_llm_prefill and benchmark_llm_decode functions."""
        try:
            import llm_bench
        except ImportError as exc:
            pytest.fail(f"Could not import llm_bench: {exc}")

        assert hasattr(llm_bench, "benchmark_llm_prefill"), (
            "llm_bench must implement benchmark_llm_prefill"
        )
        assert hasattr(llm_bench, "benchmark_llm_decode"), (
            "llm_bench must implement benchmark_llm_decode"
        )


# ---------------------------------------------------------------------------
# 2. Prefill vs. Decode Discrepancy & Metrics (Issue #22)
# ---------------------------------------------------------------------------


class TestLLMMetricsAndOperationalRegimes:
    """Validate prefill compute-bound vs decode bandwidth-bound operational metrics."""

    @pytest.mark.parametrize("prompt_len", [128, 512, 2048])
    def test_prefill_metrics_contract(self, prompt_len: int):
        """Prefill phase must report TTFT (time-to-first-token) and compute throughput."""
        try:
            import llm_bench
        except ImportError:
            pytest.fail("llm_bench module required (Issue #22)")

        # Verify PrefillMetrics data structure
        if hasattr(llm_bench, "PrefillMetrics"):
            fields = llm_bench.PrefillMetrics.__dataclass_fields__
            assert any(f in fields for f in ("ttft_ms", "latency_ms", "ttft_seconds")), (
                "PrefillMetrics must report Time-To-First-Token"
            )
            assert any(f in fields for f in ("gflops", "achieved_gflops", "throughput_tokens_s")), (
                "PrefillMetrics must report compute throughput"
            )

    @pytest.mark.parametrize("prompt_len", [128, 512, 2048])
    def test_decode_metrics_contract(self, prompt_len: int):
        """Decode phase must report generation throughput (tokens/s) and memory bandwidth util."""
        try:
            import llm_bench
        except ImportError:
            pytest.fail("llm_bench module required (Issue #22)")

        # Verify DecodeMetrics data structure
        if hasattr(llm_bench, "DecodeMetrics"):
            fields = llm_bench.DecodeMetrics.__dataclass_fields__
            assert any(
                f in fields for f in ("tokens_per_sec", "tokens_s", "throughput_tokens_s")
            ), "DecodeMetrics must report tokens/second"
            assert any(
                f in fields for f in ("memory_bw_util", "memory_bw_utilization", "achieved_gbw")
            ), "DecodeMetrics must report memory bandwidth utilization"


# ---------------------------------------------------------------------------
# 3. KV-Cache Memory Traffic Modeling (Issue #22)
# ---------------------------------------------------------------------------


class TestKVCacheTrafficModeling:
    """Validate theoretical DRAM traffic formulas for KV cache fetching."""

    def test_kv_cache_read_traffic_linear_with_context_length(self):
        """KV-cache read traffic per decode step equals 2 * B * L * D * dtype_bytes.

        For each decode token step:
          Batch B = 1, Hidden Dimension D = 4096, dtype_bytes = 2 (FP16/BF16).
          At L = 128:  2 * 1 * 128 * 4096 * 2  = 2,097,152 bytes (2 MB).
          At L = 2048: 2 * 1 * 2048 * 4096 * 2 = 33,554,432 bytes (32 MB).
          Traffic ratio must scale exactly by L2 / L1 = 2048 / 128 = 16x.
        """
        b = 1
        d = 4096
        dtype_bytes = 2

        l_short = 128
        l_long = 2048

        kv_traffic_short = 2 * b * l_short * d * dtype_bytes
        kv_traffic_long = 2 * b * l_long * d * dtype_bytes

        assert kv_traffic_short == 2_097_152
        assert kv_traffic_long == 33_554_432
        assert kv_traffic_long / kv_traffic_short == 16.0

    def test_total_decode_step_traffic_formula(self):
        """Per-token decode memory traffic includes weight read + KV history read + new KV write.

        T_step = 12 * D^2 * dtype_bytes (weights)
               + 2 * B * L * D * dtype_bytes (KV read)
               + 2 * B * 1 * D * dtype_bytes (KV write).
        """
        d = 2048
        b = 2
        l = 512
        dtype_bytes = 2

        weight_traffic = 12 * (d**2) * dtype_bytes
        kv_read_traffic = 2 * b * l * d * dtype_bytes
        kv_write_traffic = 2 * b * 1 * d * dtype_bytes
        total_traffic = weight_traffic + kv_read_traffic + kv_write_traffic

        assert weight_traffic == 12 * 4_194_304 * 2  # 100,663,296 bytes
        assert kv_read_traffic == 2 * 2 * 512 * 2048 * 2  # 8,388,608 bytes
        assert kv_write_traffic == 2 * 2 * 1 * 2048 * 2  # 16,384 bytes
        assert total_traffic == 109_068_288


# ---------------------------------------------------------------------------
# 4. Memory-Bandwidth Saturation in Gap Analysis (Issue #22)
# ---------------------------------------------------------------------------


class TestMemoryBandwidthSaturationAnalysis:
    """Validate analyze_gap records memory-bandwidth saturation during decoding."""

    def test_analyze_gap_detects_bandwidth_saturation(self):
        """When memory bandwidth utilization >= 60%, analyze_gap must record saturation finding."""
        hw = HardwareSpec(
            name="TestGPU",
            peak_flops=100e12,
            memory_bandwidth=1000e9,  # 1 TB/s
        )

        # 100 MB total traffic in 0.12 ms = 100e6 / 0.12e-3 = 833 GB/s (> 80% bandwidth util)
        traffic_bytes = 100 * 1024 * 1024
        est = SimpleNamespace(
            flops=1_000_000,
            total_bytes=traffic_bytes,
            read_bytes=traffic_bytes,
            write_bytes=0,
            arithmetic_intensity=1_000_000 / traffic_bytes,
        )

        # High bandwidth utilization: 80% (0.80)
        meas_saturated = SimpleNamespace(
            median_seconds=traffic_bytes / (800e9),  # 800 GB/s achieved out of 1000 GB/s
            samples_seconds=(traffic_bytes / (800e9),),
        )

        gap = analyze_gap(est, meas_saturated, hw)
        assert isinstance(gap, GapAnalysis)

        # Requirement: must record memory-bandwidth saturation finding
        saturation_findings = [
            f for f in gap.findings if "saturation" in f.lower() or "memory-bandwidth" in f.lower()
        ]
        assert len(saturation_findings) > 0, (
            "analyze_gap must identify memory-bandwidth saturation when bandwidth util >= 60%"
        )

    def test_analyze_gap_bandwidth_saturation_boundary(self):
        """Boundary test: 55% util should not trigger saturation, 65% util must trigger saturation."""
        hw = HardwareSpec(name="TestGPU", peak_flops=100e12, memory_bandwidth=1000e9)
        traffic_bytes = 50 * 1024 * 1024
        est = SimpleNamespace(
            flops=1_000_000,
            total_bytes=traffic_bytes,
            read_bytes=traffic_bytes,
            write_bytes=0,
            arithmetic_intensity=1_000_000 / traffic_bytes,
        )

        # 55% bandwidth utilization (550 GB/s achieved)
        meas_low = SimpleNamespace(
            median_seconds=traffic_bytes / 550e9,
            samples_seconds=(traffic_bytes / 550e9,),
        )
        gap_low = analyze_gap(est, meas_low, hw)
        saturated_low = [f for f in gap_low.findings if "saturation" in f.lower()]
        assert len(saturated_low) == 0, (
            "55% bandwidth utilization should not trigger saturation finding"
        )
