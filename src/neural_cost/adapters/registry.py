"""Adapter discovery and factory."""

from __future__ import annotations

import importlib.util

from .base import FrameworkAdapter

_ADAPTER_MAP: dict[str, tuple[str, str]] = {
    "torch": ("neural_cost.adapters.torch", "TorchAdapter"),
    "pytorch": ("neural_cost.adapters.torch", "TorchAdapter"),
    "torch-fx": ("neural_cost.adapters.torch", "TorchFxAdapter"),
    "pytorch-fx": ("neural_cost.adapters.torch", "TorchFxAdapter"),
    "jax": ("neural_cost.adapters.jax", "JaxAdapter"),
    "tensorflow": ("neural_cost.adapters.tensorflow", "TensorFlowAdapter"),
    "tf": ("neural_cost.adapters.tensorflow", "TensorFlowAdapter"),
}


_PACKAGE_MAP: dict[str, str] = {
    "torch": "torch",
    "pytorch": "torch",
    "torch-fx": "torch",
    "pytorch-fx": "torch",
    "jax": "jax",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
}


def get_adapter(name: str) -> FrameworkAdapter:
    """Return an adapter instance for the named framework.

    Accepted names: 'torch', 'pytorch', 'torch-fx', 'pytorch-fx', 'jax', 'tensorflow', 'tf'.
    Raises ValueError for unknown names and ImportError if the framework
    package is not installed.
    """
    key = name.lower().strip()
    if key not in _ADAPTER_MAP:
        raise ValueError(
            f"Unknown framework {name!r}. "
            f"Choose from: {', '.join(sorted({m for m, _ in _ADAPTER_MAP.values()}))}"
        )
    module_path, class_name = _ADAPTER_MAP[key]
    package = _PACKAGE_MAP.get(key, key)
    if importlib.util.find_spec(package) is None:
        raise ImportError(
            f"Framework {package!r} is not installed. Install it with: pip install neural-cost[{package}]"
        )
    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)
    return adapter_class()


def available_adapters() -> list[str]:
    """Return canonical names of frameworks whose packages are installed."""
    result = []
    seen = set()
    for key, (module_path, class_name) in _ADAPTER_MAP.items():
        if (module_path, class_name) in seen:
            continue
        seen.add((module_path, class_name))
        package = _PACKAGE_MAP.get(key, key)
        if importlib.util.find_spec(package) is not None:
            result.append(key)
    return result
