"""Runtime measurement primitives independent of ML frameworks."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter_ns
from typing import Any


@dataclass(frozen=True, slots=True)
class Measurement:
    """A measured execution result."""

    median_seconds: float
    samples_seconds: tuple[float, ...]
    peak_memory_bytes: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.median_seconds <= 0:
            raise ValueError("median_seconds must be positive")


def benchmark(
    function: Callable[..., Any], *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
) -> Measurement:
    """Benchmark synchronous Python or CPU functions with warmup iterations.

    GPU frameworks should use their adapter's ``benchmark`` method so device
    synchronization and memory telemetry are handled correctly.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be at least one")
    for _ in range(warmup):
        function(*args, **kwargs)
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        function(*args, **kwargs)
        samples.append((perf_counter_ns() - start) / 1_000_000_000)
    return Measurement(median(samples), tuple(samples))
