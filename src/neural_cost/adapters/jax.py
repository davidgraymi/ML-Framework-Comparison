"""JAX adapter for common jaxpr dot and elementwise primitives."""

from collections.abc import Sequence
from typing import Any

from ..operations import Operation
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
                    operations.append(Operation(f"dot_{index}", "matmul", input_shapes, output_shape, dtype_bytes))
            elif primitive in {"add", "mul", "max", "exp", "tanh", "logistic"}:
                operations.append(Operation(f"{primitive}_{index}", "elementwise", input_shapes, output_shape, dtype_bytes))
        return operations
