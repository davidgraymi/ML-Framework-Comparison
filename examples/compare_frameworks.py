"""Compare identical matrix-multiply workloads across installed frameworks.

Run after installing one or more framework extras, for example:

    pip install -e '.[torch,jax,tensorflow]'
    python examples/compare_frameworks.py

The supplied hardware values are only roofline assumptions.  Pass values for
the hardware actually running the process for useful efficiency comparisons.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from neural_cost import HardwareSpec, analyze_gap, estimate_model
from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchAdapter
from neural_cost.adapters.base import FrameworkAdapter


@dataclass(frozen=True)
class Result:
    framework: str
    flops: int
    bytes: int
    median_ms: float
    efficiency: float


def installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def evaluate(
    name: str,
    model: Callable[..., Any],
    inputs: tuple[Any, ...],
    adapter: FrameworkAdapter,
    hardware: HardwareSpec,
) -> Result:
    estimate = estimate_model(model, inputs, adapter)
    measurement = adapter.benchmark(model, *inputs, warmup=5, repeats=20)
    gap = analyze_gap(estimate, measurement, hardware)
    return Result(
        name,
        estimate.flops,
        estimate.total_bytes,
        measurement.median_seconds * 1e3,
        gap.efficiency,
    )


def run_torch(hardware: HardwareSpec) -> Result:
    import torch

    model = torch.nn.Linear(1024, 1024, bias=False).eval()
    return evaluate("PyTorch", model, (torch.ones((64, 1024)),), TorchAdapter(), hardware)


def run_jax(hardware: HardwareSpec) -> Result:
    import jax.numpy as jnp

    def model(left: Any, right: Any) -> Any:
        return jnp.matmul(left, right)

    return evaluate(
        "JAX", model, (jnp.ones((64, 1024)), jnp.ones((1024, 1024))), JaxAdapter(), hardware
    )


def run_tensorflow(hardware: HardwareSpec) -> Result:
    import tensorflow as tf

    model = tf.keras.Sequential([tf.keras.layers.Dense(1024, use_bias=False)])
    return evaluate("TensorFlow", model, (tf.ones((64, 1024)),), TensorFlowAdapter(), hardware)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-flops", type=float, default=150e9)
    parser.add_argument("--memory-bandwidth", type=float, default=50e9)
    args = parser.parse_args()
    hardware = HardwareSpec("assumed hardware", args.peak_flops, args.memory_bandwidth)

    runners = (("torch", run_torch), ("jax", run_jax), ("tensorflow", run_tensorflow))
    results = [runner(hardware) for package, runner in runners if installed(package)]
    if not results:
        raise SystemExit("Install at least one framework extra: torch, jax, or tensorflow.")

    print(f"Assumed roofline: {hardware.peak_flops / 1e9:.1f} GFLOP/s, "
          f"{hardware.memory_bandwidth / 1e9:.1f} GB/s")
    print(f"{'framework':<12} {'FLOPs':>14} {'bytes':>14} {'median ms':>12} {'efficiency':>12}")
    for result in results:
        print(f"{result.framework:<12} {result.flops:>14,d} {result.bytes:>14,d} "
              f"{result.median_ms:>12.3f} {result.efficiency:>11.1%}")


if __name__ == "__main__":
    main()
