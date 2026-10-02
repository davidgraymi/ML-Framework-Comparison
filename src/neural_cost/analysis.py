"""Compare a measurement with the roofline lower bound and explain the gap."""

from dataclasses import dataclass

from .estimate import CostEstimate, FusedCostEstimate
from .hardware import HardwareSpec
from .memory import MemoryEstimate
from .model import ModelProfile
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
    cache_bound_seconds: float | None = None
    resident_cache_level: str | None = None
    cache_efficiency: float | None = None
    fused_lower_bound_seconds: float | None = None
    fused_efficiency: float | None = None

    def render(self) -> str:
        """Return a compact, terminal-friendly performance-gap report."""
        lines = [
            "Neural cost gap analysis",
            (
                f"  bound: {self.lower_bound_seconds * 1e3:.3f} ms "
                f"(compute {self.compute_bound_seconds * 1e3:.3f} ms, "
                f"memory {self.bandwidth_bound_seconds * 1e3:.3f} ms)"
            ),
        ]
        if self.fused_lower_bound_seconds is not None:
            lines.append(
                f"  fused bound: {self.fused_lower_bound_seconds * 1e3:.3f} ms"
                + (f" (efficiency {self.fused_efficiency:.1%})" if self.fused_efficiency is not None else "")
            )
        if self.resident_cache_level and self.cache_bound_seconds is not None:
            lines.append(
                f"  cache residency: {self.resident_cache_level} "
                f"(bound {self.cache_bound_seconds * 1e3:.3f} ms"
                + (f", efficiency {self.cache_efficiency:.1%}" if self.cache_efficiency is not None else "")
                + ")"
            )
        lines.extend([
            f"  observed: {self.observed_seconds * 1e3:.3f} ms",
            f"  roofline efficiency: {self.efficiency:.1%} ({self.bottleneck}-bound)",
            (
                f"  achieved: {self.achieved_flops / 1e9:.3f} GFLOP/s, "
                f"{self.achieved_bandwidth / 1e9:.3f} GB/s"
            ),
        ])
        lines.extend(f"  next: {finding}" for finding in self.findings)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MemoryGapAnalysis:
    """Comparison between idealized tensor storage and allocator telemetry."""

    theoretical_minimum_bytes: int
    theoretical_conservative_bytes: int
    observed_peak_bytes: int | None
    observed_reserved_bytes: int | None
    overhead_ratio: float | None
    findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelGapAnalysis:
    """Combined compute and memory gap analysis for one workload."""

    performance: GapAnalysis
    memory: MemoryGapAnalysis


def analyze_gap(
    estimate: CostEstimate | FusedCostEstimate, measurement: Measurement, hardware: HardwareSpec
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

    fused_lower_bound_seconds = None
    fused_efficiency = None
    if isinstance(estimate, FusedCostEstimate) and estimate.eliminated_bytes > 0:
        fused_bandwidth = estimate.total_bytes / hardware.memory_bandwidth
        fused_lower_bound_seconds = max(compute, fused_bandwidth)
        fused_efficiency = min(1.0, fused_lower_bound_seconds / observed)
        findings.append(
            f"Fusion optimization: kernel fusion eliminates {estimate.eliminated_bytes / 1024:.1f} KB of traffic "
            f"({estimate.traffic_reduction_ratio:.1%} reduction), raising arithmetic intensity to "
            f"{estimate.arithmetic_intensity:.1f} FLOP/byte and fused lower bound to {fused_lower_bound_seconds * 1e3:.3f} ms."
        )

    resident_cache = hardware.find_resident_cache(estimate.total_bytes)
    cache_bound_seconds = None
    resident_cache_level = None
    cache_efficiency = None

    if resident_cache is not None:
        resident_cache_level = resident_cache.name
        cache_mem_bound = estimate.total_bytes / resident_cache.bandwidth
        cache_bound_seconds = max(compute, cache_mem_bound)
        cache_efficiency = min(1.0, cache_bound_seconds / observed)
        findings.append(
            f"Cache-resident: total tensor traffic ({estimate.total_bytes / (1024 * 1024):.2f} MB) fits in {resident_cache.name} "
            f"({resident_cache.capacity / (1024 * 1024):.0f} MB, {resident_cache.bandwidth / 1e9:.0f} GB/s). "
            f"Hierarchical roofline bound is {cache_bound_seconds * 1e3:.3f} ms."
        )

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
        cache_bound_seconds=cache_bound_seconds,
        resident_cache_level=resident_cache_level,
        cache_efficiency=cache_efficiency,
        fused_lower_bound_seconds=fused_lower_bound_seconds,
        fused_efficiency=fused_efficiency,
    )



def analyze_memory_gap(estimate: MemoryEstimate, measurement: Measurement) -> MemoryGapAnalysis:
    """Explain allocator memory above a static tensor-storage estimate.

    If an adapter cannot obtain allocator telemetry, the result reports the
    static bounds and explicitly leaves the observed fields unset.
    """
    observed = measurement.peak_memory_bytes or measurement.allocated_memory_bytes
    reserved = measurement.reserved_memory_bytes
    findings: list[str] = []
    ratio = None
    if observed is None:
        findings.append("No allocator telemetry was collected for this device.")
    elif estimate.inference_conservative_bytes == 0:
        findings.append("No tensor storage was attributed to the static operation trace.")
    else:
        ratio = observed / estimate.inference_conservative_bytes
        if ratio > 1.5:
            findings.append(
                "Observed peak exceeds conservative tensor storage; inspect allocator pools, "
                "workspaces, retained tensors, and fragmentation."
            )
        else:
            findings.append("Observed peak is near the conservative static tensor-storage bound.")
    if reserved is not None and observed is not None and reserved > observed:
        findings.append("Reserved memory exceeds allocated memory; the caching allocator retains a pool.")
    return MemoryGapAnalysis(
        estimate.inference_minimum_bytes,
        estimate.inference_conservative_bytes,
        observed,
        reserved,
        ratio,
        tuple(findings),
    )


def analyze_model_gap(
    profile: ModelProfile, measurement: Measurement, hardware: HardwareSpec
) -> ModelGapAnalysis:
    """Analyze compute roofline efficiency and memory allocation in one call."""
    cost_to_analyze = profile.fused_cost if profile.fused_cost is not None else profile.cost
    return ModelGapAnalysis(
        analyze_gap(cost_to_analyze, measurement, hardware),
        analyze_memory_gap(profile.memory, measurement),
    )

