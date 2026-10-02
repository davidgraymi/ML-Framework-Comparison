"""Framework-neutral neural-network compute and memory cost analysis."""

from .adapters import available_adapters, get_adapter
from .analysis import (
    GapAnalysis,
    MemoryGapAnalysis,
    ModelGapAnalysis,
    analyze_gap,
    analyze_memory_gap,
    analyze_model_gap,
)
from .api import estimate_model
from .estimate import CostEstimate, estimate_operations
from .hardware import CacheSpec, HardwareSpec
from .hardware_detect import DetectionResult, detect_hardware
from .memory import MemoryEstimate, estimate_memory
from .model import ModelProfile, profile_model
from .operations import Operation
from .profiler import Measurement, benchmark

__all__ = [
    "CacheSpec",
    "CostEstimate",
    "DetectionResult",
    "GapAnalysis",
    "HardwareSpec",
    "Measurement",
    "MemoryEstimate",
    "MemoryGapAnalysis",
    "ModelGapAnalysis",
    "ModelProfile",
    "Operation",
    "analyze_gap",
    "analyze_memory_gap",
    "analyze_model_gap",
    "available_adapters",
    "benchmark",
    "detect_hardware",
    "estimate_memory",
    "estimate_model",
    "estimate_operations",
    "get_adapter",
    "profile_model",
]
