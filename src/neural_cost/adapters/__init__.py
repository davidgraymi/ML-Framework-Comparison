"""Optional framework adapters and the custom-adapter contract."""

from .base import FrameworkAdapter
from .jax import JaxAdapter
from .registry import available_adapters, get_adapter
from .tensorflow import TensorFlowAdapter
from .torch import TorchAdapter, TorchFxAdapter

__all__ = [
    "FrameworkAdapter",
    "JaxAdapter",
    "TensorFlowAdapter",
    "TorchAdapter",
    "TorchFxAdapter",
    "available_adapters",
    "get_adapter",
]
