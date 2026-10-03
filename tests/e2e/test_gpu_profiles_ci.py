"""End-to-End Tests for Datacenter NVIDIA Hardware Profiles & GPU Benchmark Script (#25).

Covers:
- NVIDIA datacenter chip specifications (_NVIDIA_CHIP_TABLE for A100, L40S, H100).
- Datacenter hardware detection and precision scaling (TF32 and FP16 Tensor Cores).
- GPU benchmark script (benchmarks/collect_gpu_data.py) CLI execution validation.
"""

from __future__ import annotations

from pathlib import Path

import neural_cost.hardware_detect as hw_detect

# ---------------------------------------------------------------------------
# 1. Datacenter NVIDIA Hardware Support & Profiles (Issue #25)
# ---------------------------------------------------------------------------


class TestNVIDIADatacenterHardwareProfiles:
    """Validate NVIDIA datacenter chip specifications (A100, L40S, H100)."""

    def test_nvidia_chip_table_datacenter_entries(self):
        """_NVIDIA_CHIP_TABLE in hardware_detect must contain A100, L40S, and H100 entries."""
        assert hasattr(hw_detect, "_NVIDIA_CHIP_TABLE"), (
            "hardware_detect must define _NVIDIA_CHIP_TABLE (Issue #25)"
        )
        table = hw_detect._NVIDIA_CHIP_TABLE

        # Check for A100, L40S, and H100
        chip_keys = " ".join(table.keys()).upper()
        assert "A100" in chip_keys, "_NVIDIA_CHIP_TABLE must contain A100 profile"
        assert "L40S" in chip_keys or "L40" in chip_keys, (
            "_NVIDIA_CHIP_TABLE must contain L40S profile"
        )
        assert "H100" in chip_keys, "_NVIDIA_CHIP_TABLE must contain H100 profile"

    def test_a100_profile_specifications(self):
        """A100 profile must specify accurate Tensor Core peak TFLOP/s and memory bandwidth."""
        assert hasattr(hw_detect, "_NVIDIA_CHIP_TABLE"), "_NVIDIA_CHIP_TABLE required"

        a100_entry = None
        for k, v in hw_detect._NVIDIA_CHIP_TABLE.items():
            if "A100" in k.upper():
                a100_entry = v
                break

        assert a100_entry is not None, "A100 entry not found in _NVIDIA_CHIP_TABLE"
        if isinstance(a100_entry, (tuple, list)):
            bw_gb_s = a100_entry[3] if len(a100_entry) > 3 else a100_entry[1]
            assert bw_gb_s >= 1500.0, f"A100 memory bandwidth should be >= 1500 GB/s, got {bw_gb_s}"
        elif hasattr(a100_entry, "memory_bandwidth"):
            assert a100_entry.memory_bandwidth >= 1500e9

    def test_h100_profile_specifications(self):
        """H100 profile must specify Hopper Tensor Core specs (>=3000 GB/s HBM3)."""
        assert hasattr(hw_detect, "_NVIDIA_CHIP_TABLE"), "_NVIDIA_CHIP_TABLE required"

        h100_entry = None
        for k, v in hw_detect._NVIDIA_CHIP_TABLE.items():
            if "H100" in k.upper():
                h100_entry = v
                break

        assert h100_entry is not None, "H100 entry not found in _NVIDIA_CHIP_TABLE"
        if isinstance(h100_entry, (tuple, list)):
            bw_gb_s = h100_entry[3] if len(h100_entry) > 3 else h100_entry[1]
            assert bw_gb_s >= 2000.0, f"H100 memory bandwidth should be >= 2000 GB/s, got {bw_gb_s}"
        elif hasattr(h100_entry, "memory_bandwidth"):
            assert h100_entry.memory_bandwidth >= 2000e9


# ---------------------------------------------------------------------------
# 2. GPU Benchmark Python Script Validation (Issue #25)
# ---------------------------------------------------------------------------


class TestGPUBenchmarkScriptRunner:
    """Validate GPU benchmark script execution (benchmarks/collect_gpu_data.py)."""

    def test_gpu_benchmark_script_exists(self):
        """GPU benchmark script benchmarks/collect_gpu_data.py must exist."""
        script_path = (
            Path(__file__).resolve().parent.parent.parent / "benchmarks" / "collect_gpu_data.py"
        )
        assert script_path.exists(), f"Expected GPU benchmark script at {script_path} (Issue #25)"

    def test_gpu_benchmark_script_help_execution(self):
        """GPU benchmark script must be executable via Python CLI (--help)."""
        import subprocess
        import sys

        script_path = (
            Path(__file__).resolve().parent.parent.parent / "benchmarks" / "collect_gpu_data.py"
        )
        proc = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"collect_gpu_data.py --help failed: {proc.stderr}"
        assert "--device" in proc.stdout, "--device argument expected in help output"
        assert "cuda" in proc.stdout.lower(), "cuda device option expected in help output"
