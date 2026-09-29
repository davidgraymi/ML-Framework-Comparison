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
                if not inputs:
                    return
                source = inputs[0]
                if not isinstance(source, torch.Tensor):
                    return
                if isinstance(module, torch.nn.Linear):
                    if not isinstance(output, torch.Tensor):
                        return
                    captured.append(
                        Operation(
                            name, "linear", (tuple(source.shape), tuple(module.weight.T.shape)),
                            tuple(output.shape),
                            dtype_bytes(source),
                            {
                                "parameter_bytes": sum(
                                    p.numel() * p.element_size()
                                    for p in module.parameters(recurse=False)
                                ),
                                "parameter_id": id(module.weight),
                            },
                        )
                    )
                elif isinstance(module, torch.nn.Conv2d):
                    if not isinstance(output, torch.Tensor):
                        return
                    captured.append(
                        Operation(
                            name, "conv2d", (tuple(source.shape), tuple(module.weight.shape)),
                            tuple(output.shape),
                            dtype_bytes(source),
                            {
                                "parameter_bytes": sum(
                                    p.numel() * p.element_size()
                                    for p in module.parameters(recurse=False)
                                ),
                                "parameter_id": id(module.weight),
                            },
                        )
                    )
                elif isinstance(module, torch.nn.Embedding):
                    if not isinstance(output, torch.Tensor):
                        return
                    captured.append(
                        Operation(
                            name, "embedding", (tuple(source.shape),),
                            tuple(output.shape),
                            dtype_bytes(module.weight),
                            {
                                "parameter_bytes": module.weight.numel() * module.weight.element_size(),
                                "parameter_id": id(module.weight),
                            },
                        )
                    )
                elif isinstance(module, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)):
                    # Unroll as input×hidden + hidden×hidden linears per layer per direction.
                    if not isinstance(output, (tuple, list)):
                        return
                    directions = 2 if module.bidirectional else 1
                    gates = 4 if isinstance(module, torch.nn.LSTM) else (3 if isinstance(module, torch.nn.GRU) else 1)
                    batch_seq = source.shape[0] * source.shape[1] if source.dim() >= 2 else source.shape[0]
                    in_size = source.shape[-1]
                    h = module.hidden_size
                    for layer in range(module.num_layers):
                        layer_in = in_size if layer == 0 else h * directions
                        # Input → hidden projection
                        captured.append(Operation(
                            f"{name}.layer{layer}.ih", "linear",
                            ((batch_seq, layer_in), (layer_in, gates * h)),
                            (batch_seq, gates * h),
                            source.element_size(),
                            {
                                "parameter_bytes": layer_in * gates * h * source.element_size(),
                                "parameter_id": id(getattr(module, f"weight_ih_l{layer}")),
                            },
                        ))
                        # Hidden → hidden projection
                        captured.append(Operation(
                            f"{name}.layer{layer}.hh", "linear",
                            ((batch_seq, h), (h, gates * h)),
                            (batch_seq, gates * h),
                            source.element_size(),
                            {
                                "parameter_bytes": h * gates * h * source.element_size(),
                                "parameter_id": id(getattr(module, f"weight_hh_l{layer}")),
                            },
                        ))
                elif isinstance(module, torch.nn.MultiheadAttention):
                    if not isinstance(output, (tuple, list, torch.Tensor)):
                        return
                    out_tensor = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(out_tensor, torch.Tensor):
                        return
                    embed_dim = module.embed_dim
                    num_heads = module.num_heads
                    seq_len = source.shape[0] if source.dim() >= 2 else 1
                    batch = source.shape[1] if source.dim() >= 3 else 1
                    captured.append(Operation(
                        name, "attention",
                        (tuple(source.shape),),
                        (seq_len, batch, embed_dim),
                        dtype_bytes(source),
                        {
                            "num_heads": num_heads,
                            "seq_len": seq_len,
                            "parameter_bytes": sum(
                                p.numel() * p.element_size()
                                for p in module.parameters(recurse=False)
                            ),
                            "parameter_id": id(module.in_proj_weight) if module.in_proj_weight is not None else id(module),
                        },
                    ))
                elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    if not isinstance(output, torch.Tensor):
                        return
                    kind = "layernorm" if isinstance(module, torch.nn.LayerNorm) else "batchnorm"
                    attrs: dict[str, int | float | tuple[int, ...]] = {}
                    if hasattr(module, "weight") and module.weight is not None:
                        attrs["parameter_bytes"] = sum(
                            p.numel() * p.element_size()
                            for p in module.parameters(recurse=False)
                        )
                        attrs["parameter_id"] = id(module.weight)
                    captured.append(Operation(name, kind, (tuple(source.shape),), tuple(output.shape),
                                              dtype_bytes(source), attrs))
            return record

        _TRACKED = (
            torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding,
            torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM,
            torch.nn.MultiheadAttention,
            torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
        )
        for name, module in model.named_modules():
            if isinstance(module, _TRACKED):
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
        """Benchmark execution and capture CUDA allocator statistics when present."""
        return self.trace(function, *args, warmup=warmup, repeats=repeats, **kwargs)

    def trace(
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        """Profile PyTorch events and CUDA memory for a representative workload.

        The trace is aggregated over ``repeats`` iterations.  CUDA timings are
        synchronized before every sample, and CUDA allocator counters are
        reported separately from the static tensor-memory estimate.
        """
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise ImportError("Install neural-cost[torch] to use TorchAdapter") from error
        if warmup < 0 or repeats < 1:
            raise ValueError("warmup must be non-negative and repeats must be at least one")
        device = next((arg.device for arg in args if isinstance(arg, torch.Tensor)), None)
        is_cuda = device is not None and device.type == "cuda"
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            function(*args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize(device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        samples: list[float] = []
        with torch.profiler.profile(activities=activities, profile_memory=is_cuda) as trace:
            for _ in range(repeats):
                start = perf_counter_ns()
                function(*args, **kwargs)
                if is_cuda:
                    torch.cuda.synchronize(device)
                samples.append((perf_counter_ns() - start) / 1_000_000_000)
        peak = torch.cuda.max_memory_allocated(device) if is_cuda else None
        allocated = torch.cuda.memory_allocated(device) if is_cuda else None
        reserved = torch.cuda.memory_reserved(device) if is_cuda else None
        events = trace.events()
        if is_cuda:
            device_time = sum(event.cuda_time_total for event in events) / 1_000_000
            event_count = sum(event.cuda_time_total > 0 for event in events)
        else:
            device_time = sum(event.cpu_time_total for event in events) / 1_000_000
            event_count = len(events)
        return Measurement(
            median(samples),
            tuple(samples),
            peak,
            str(device) if device else "cpu",
            allocated,
            reserved,
            event_count,
            device_time,
        )
