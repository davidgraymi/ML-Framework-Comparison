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


class TorchFxAdapter(TorchAdapter):
    """Capture PyTorch module and inline functional operations via torch.fx graph tracing."""

    name = "pytorch-fx"

    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        try:
            import torch
            import torch.fx
            from torch.fx.passes.shape_prop import ShapeProp
        except ImportError as error:  # pragma: no cover
            raise ImportError("Install neural-cost[torch] to use TorchFxAdapter") from error

        try:
            gm = torch.fx.symbolic_trace(model)
            ShapeProp(gm).propagate(*example_inputs)
        except Exception:
            # Fall back gracefully to hook-based capture if symbolic tracing fails
            return super().operations(model, example_inputs)

        captured: list[Operation] = []
        modules = dict(model.named_modules())

        def get_shape(val: Any) -> tuple[int, ...] | None:
            if hasattr(val, "shape"):
                return tuple(int(d) for d in val.shape)
            if isinstance(val, torch.fx.Node) and "tensor_meta" in val.meta:
                meta = val.meta["tensor_meta"]
                if hasattr(meta, "shape"):
                    return tuple(int(d) for d in meta.shape)
            return None

        def get_dtype_bytes(val: Any) -> int:
            if hasattr(val, "dtype"):
                return val.element_size() if hasattr(val, "element_size") else 4
            if isinstance(val, torch.fx.Node) and "tensor_meta" in val.meta:
                meta = val.meta["tensor_meta"]
                if hasattr(meta, "dtype"):
                    return torch.empty((), dtype=meta.dtype).element_size()
            return 4

        for node in gm.graph.nodes:
            out_shape = get_shape(node)
            if not out_shape:
                continue
            dtype_b = get_dtype_bytes(node)

            if node.op == "call_module":
                mod = modules.get(str(node.target))
                if mod is None:
                    continue
                in_shapes = tuple(s for s in (get_shape(arg) for arg in node.args) if s is not None)
                if isinstance(mod, torch.nn.Linear):
                    if in_shapes:
                        captured.append(Operation(
                            node.name, "linear",
                            (in_shapes[0], tuple(mod.weight.T.shape)),
                            out_shape, dtype_b,
                            {
                                "parameter_bytes": sum(p.numel() * p.element_size() for p in mod.parameters(recurse=False)),
                                "parameter_id": id(mod.weight),
                            }
                        ))
                elif isinstance(mod, torch.nn.Conv2d):
                    if in_shapes:
                        captured.append(Operation(
                            node.name, "conv2d",
                            (in_shapes[0], tuple(mod.weight.shape)),
                            out_shape, dtype_b,
                            {
                                "parameter_bytes": sum(p.numel() * p.element_size() for p in mod.parameters(recurse=False)),
                                "parameter_id": id(mod.weight),
                            }
                        ))
                elif isinstance(mod, torch.nn.Embedding):
                    captured.append(Operation(
                        node.name, "embedding", in_shapes or (out_shape,), out_shape, dtype_b,
                        {
                            "parameter_bytes": mod.weight.numel() * mod.weight.element_size(),
                            "parameter_id": id(mod.weight),
                        }
                    ))
                elif isinstance(mod, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Sigmoid, torch.nn.Tanh)):
                    captured.append(Operation(node.name, "elementwise", in_shapes or (out_shape,), out_shape, dtype_b))
                elif isinstance(mod, torch.nn.LayerNorm):
                    captured.append(Operation(node.name, "layernorm", in_shapes or (out_shape,), out_shape, dtype_b))
                elif isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    captured.append(Operation(node.name, "batchnorm", in_shapes or (out_shape,), out_shape, dtype_b))
                elif isinstance(mod, (torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d)):
                    k = getattr(mod, "kernel_size", (1, 1))
                    k_tuple = k if isinstance(k, tuple) else (k, k)
                    captured.append(Operation(node.name, "pooling", in_shapes or (out_shape,), out_shape, dtype_b, {"kernel_size": k_tuple}))

            elif node.op in {"call_function", "call_method"}:
                fn = node.target
                fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
                in_shapes = tuple(s for s in (get_shape(arg) for arg in node.args) if s is not None)

                if fn_name in {"matmul", "mm", "bmm", "linear"} or fn in {torch.matmul, torch.mm, torch.bmm, torch.nn.functional.linear}:
                    if len(in_shapes) >= 2:
                        captured.append(Operation(node.name, "matmul", in_shapes[:2], out_shape, dtype_b))
                elif fn_name in {
                    "add", "mul", "sub", "truediv", "div", "relu", "gelu", "silu", "sigmoid",
                    "tanh", "exp", "log", "sqrt", "neg", "abs"
                } or any(f in str(fn) for f in ["add", "mul", "sub", "div", "relu", "gelu", "silu", "sigmoid", "tanh", "exp"]):
                    captured.append(Operation(node.name, "elementwise", in_shapes or (out_shape,), out_shape, dtype_b))
                elif fn_name in {"layer_norm", "batch_norm"} or fn in {torch.nn.functional.layer_norm, torch.nn.functional.batch_norm}:
                    kind = "layernorm" if "layer" in fn_name else "batchnorm"
                    captured.append(Operation(node.name, kind, in_shapes or (out_shape,), out_shape, dtype_b))
                elif fn_name == "softmax" or fn == torch.nn.functional.softmax:
                    captured.append(Operation(node.name, "softmax", in_shapes or (out_shape,), out_shape, dtype_b))

        return captured if captured else super().operations(model, example_inputs)

