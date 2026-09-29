"""JAX adapter for common jaxpr dot and elementwise primitives."""

from collections.abc import Sequence
from statistics import median
from time import perf_counter_ns
from typing import Any

from ..operations import Operation, numel
from ..profiler import Measurement
from .base import FrameworkAdapter


class JaxAdapter(FrameworkAdapter):
    """Trace a JAX function and translate common primitives to operations."""

    name = "jax"

    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        try:
            import jax
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[jax] to use JaxAdapter") from error
        closed = jax.make_jaxpr(model)(*example_inputs)
        parameters = {id(var) for var in closed.jaxpr.invars[1:]}
        operations: list[Operation] = []
        for index, equation in enumerate(closed.jaxpr.eqns):
            primitive = equation.primitive.name
            in_avals = [value.aval for value in equation.invars if hasattr(value, "aval")]
            out_avals = [value.aval for value in equation.outvars if hasattr(value, "aval")]
            if not out_avals or not hasattr(out_avals[0], "shape"):
                continue
            input_shapes = tuple(tuple(int(d) for d in aval.shape) for aval in in_avals if hasattr(aval, "shape"))
            output_shape = tuple(int(d) for d in out_avals[0].shape)
            dtype_bytes = int(out_avals[0].dtype.itemsize)
            if primitive == "dot_general" and len(input_shapes) == 2:
                if len(input_shapes[0]) >= 2 and len(input_shapes[1]) == 2:
                    attrs = {}
                    if id(equation.invars[1]) in parameters:
                        attrs = {
                            "parameter_bytes": numel(input_shapes[1]) * dtype_bytes,
                            "parameter_id": id(equation.invars[1]),
                        }
                    operations.append(Operation(f"dot_{index}", "matmul", input_shapes, output_shape, dtype_bytes, attrs))
            elif primitive == "conv_general_dilated":
                if len(input_shapes) == 2 and len(input_shapes[0]) == 4 and len(input_shapes[1]) == 4:
                    attrs = {}
                    if id(equation.invars[1]) in parameters:
                        attrs = {
                            "parameter_bytes": numel(input_shapes[1]) * dtype_bytes,
                            "parameter_id": id(equation.invars[1]),
                        }
                    operations.append(Operation(f"conv_{index}", "conv2d", input_shapes, output_shape, dtype_bytes, attrs))
            elif primitive in {"add", "mul", "max", "exp", "tanh", "logistic", "sub", "neg", "sin", "cos", "rsqrt", "integer_pow", "abs", "log", "sqrt", "reduce_sum", "reduce_max"}:
                operations.append(Operation(f"{primitive}_{index}", "elementwise", input_shapes, output_shape, dtype_bytes))
        return operations

    def benchmark(
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        """Benchmark a JAX function while waiting for dispatched device work."""
        try:
            import jax
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[jax] to use JaxAdapter") from error

        def wait(value: Any) -> None:
            for leaf in jax.tree.leaves(value):
                if hasattr(leaf, "block_until_ready"):
                    leaf.block_until_ready()

        if warmup < 0 or repeats < 1:
            raise ValueError("warmup must be non-negative and repeats must be at least one")
        for _ in range(warmup):
            wait(function(*args, **kwargs))
        samples = []
        for _ in range(repeats):
            start = perf_counter_ns()
            wait(function(*args, **kwargs))
            samples.append((perf_counter_ns() - start) / 1_000_000_000)
        device = str(args[0].device) if args and hasattr(args[0], "device") else None
        return Measurement(median(samples), tuple(samples), device=device)
