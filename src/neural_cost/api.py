"""High-level convenience API for adapter-backed analyses."""

from collections.abc import Sequence
from typing import Any

from .adapters.base import FrameworkAdapter
from .estimate import CostEstimate, estimate_operations


def estimate_model(
    model: Any, example_inputs: Sequence[Any], adapter: FrameworkAdapter
) -> CostEstimate:
    """Estimate a model using a supplied framework adapter."""
    return estimate_operations(adapter.operations(model, example_inputs))
