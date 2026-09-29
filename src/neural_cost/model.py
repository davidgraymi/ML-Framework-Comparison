"""Combined static compute and memory profiles for models."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .adapters.base import FrameworkAdapter
from .estimate import CostEstimate, estimate_operations
from .memory import MemoryEstimate, estimate_memory


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Theoretical compute and tensor-storage profile for concrete inputs."""

    cost: CostEstimate
    memory: MemoryEstimate


def profile_model(
    model: Any,
    example_inputs: Sequence[Any],
    adapter: FrameworkAdapter,
    *,
    training: bool = False,
    optimizer_state_multiplier: float = 0.0,
) -> ModelProfile:
    """Build a framework-neutral static profile through an adapter."""
    operations = tuple(adapter.operations(model, example_inputs))
    return ModelProfile(
        estimate_operations(operations),
        estimate_memory(
            operations,
            training=training,
            optimizer_state_multiplier=optimizer_state_multiplier,
        ),
    )
