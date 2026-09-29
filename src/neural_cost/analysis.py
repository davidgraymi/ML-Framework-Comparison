"""Compare a measurement with the roofline lower bound and explain the gap."""

from dataclasses import dataclass

from .estimate import CostEstimate
from .hardware import HardwareSpec
from .profiler import Measurement


@dataclass(frozen=True, slots=True)
class GapAnalysis:
    lower_bound_seconds: float
    compute_bound_seconds: float
    bandwidth_bound_seconds: float
    observed_seconds: float
    efficiency: float
    achieved_flops: float
    achieved_bandwidth: float
    bottleneck: str
    findings: tuple[str, ...]

    def render(self) -> str:
        """Return a compact, terminal-friendly performance-gap report."""
        lines = [
            "Neural cost gap analysis",
            f"  bound: {self.lower_bound_seconds * 1e3:.3f} ms "
            f"(compute {self.compute_bound_seconds * 1e3:.3f} ms, "
            f"memory {self.bandwidth_bound_seconds * 1e3:.3f} ms)",
            f"  observed: {self.observed_seconds * 1e3:.3f} ms",
            f"  roofline efficiency: {self.efficiency:.1%} ({self.bottleneck}-bound)",
            f"  achieved: {self.achieved_flops / 1e9:.3f} GFLOP/s, "
            f"{self.achieved_bandwidth / 1e9:.3f} GB/s",
        ]
        lines.extend(f"  next: {finding}" for finding in self.findings)
        return "\n".join(lines)


def analyze_gap(
    estimate: CostEstimate, measurement: Measurement, hardware: HardwareSpec
) -> GapAnalysis:
    """Analyze observed runtime against a roofline lower bound.

    Findings are hypotheses to investigate, not assertions about kernel-level
    behavior.  The compulsory-traffic model makes the achieved bandwidth an
    effective value, especially where intermediates are materialized.
    """
    compute = estimate.flops / hardware.peak_flops
    bandwidth = estimate.total_bytes / hardware.memory_bandwidth
    lower_bound = max(compute, bandwidth)
    observed = measurement.median_seconds
    efficiency = min(1.0, lower_bound / observed)
    bottleneck = "compute" if compute >= bandwidth else "memory"
    findings: list[str] = []
    if bottleneck == "memory":
        findings.append("Memory-bound: consider fusion, reduced precision, or fewer materialized tensors.")
    else:
        findings.append("Compute-bound: consider faster kernels, tensor cores, or greater parallelism.")
    if efficiency < 0.5:
        findings.append("Large roofline gap: inspect launch overhead, synchronization, shape padding, and data movement.")
    if hardware.memory_capacity is not None and estimate.total_bytes > hardware.memory_capacity:
        findings.append("Compulsory traffic exceeds device memory capacity; partitioning or offload may be required.")
    return GapAnalysis(
        lower_bound,
        compute,
        bandwidth,
        observed,
        efficiency,
        estimate.flops / observed,
        estimate.total_bytes / observed,
        bottleneck,
        tuple(findings),
    )
