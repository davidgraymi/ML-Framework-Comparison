"""Hardware descriptions used by the roofline model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """A device's relevant performance limits.

    Values are expressed in SI units: FLOP/s, bytes/s, and bytes.  Use a
    precision-specific compute peak (for example, FP16 tensor-core peak) when
    analyzing a workload at that precision.
    """

    name: str
    peak_flops: float
    memory_bandwidth: float
    memory_capacity: int | None = None

    def __post_init__(self) -> None:
        if self.peak_flops <= 0 or self.memory_bandwidth <= 0:
            raise ValueError("peak_flops and memory_bandwidth must be positive")

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) at the compute/memory boundary."""
        return self.peak_flops / self.memory_bandwidth
