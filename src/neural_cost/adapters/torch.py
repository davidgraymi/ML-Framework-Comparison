"""PyTorch module adapter (available only when PyTorch is installed)."""

from collections.abc import Sequence
from statistics import median
from time import perf_counter_ns
from typing import Any

from ..operations import Operation
from ..profiler import Measurement
from .base import FrameworkAdapter


class TorchAdapter(FrameworkAdapter):
    """Capture Linear and Conv2d module calls through forward hooks."""

    name = "pytorch"

    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on optional package
            raise ImportError("Install neural-cost[torch] to use TorchAdapter") from error

        captured: list[Operation] = []
        hooks = []

        def dtype_bytes(tensor: Any) -> int:
            return tensor.element_size()

        def hook(name: str, module: Any):
            def record(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
                if not inputs or not isinstance(output, torch.Tensor):
                    return
                source = inputs[0]
                if not isinstance(source, torch.Tensor):
                    return
                if isinstance(module, torch.nn.Linear):
                    captured.append(
                        Operation(
                            name, "linear", (tuple(source.shape), tuple(module.weight.T.shape)),
                            tuple(output.shape), dtype_bytes(source),
                        )
                    )
                elif isinstance(module, torch.nn.Conv2d):
                    captured.append(
                        Operation(
                            name, "conv2d", (tuple(source.shape), tuple(module.weight.shape)),
                            tuple(output.shape), dtype_bytes(source),
                        )
                    )
            return record

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook(name or module.__class__.__name__, module)))
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                model(*example_inputs)
        finally:
            for registered_hook in hooks:
                registered_hook.remove()
            model.train(was_training)
        return captured

    def benchmark(
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise ImportError("Install neural-cost[torch] to use TorchAdapter") from error
        device = next((arg.device for arg in args if isinstance(arg, torch.Tensor)), None)
        is_cuda = device is not None and device.type == "cuda"
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            function(*args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            start = perf_counter_ns()
            function(*args, **kwargs)
            if is_cuda:
                torch.cuda.synchronize(device)
            samples.append((perf_counter_ns() - start) / 1_000_000_000)
        peak = torch.cuda.max_memory_allocated(device) if is_cuda else None
        return Measurement(median(samples), tuple(samples), peak, str(device) if device else "cpu")
