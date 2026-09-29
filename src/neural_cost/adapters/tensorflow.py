"""TensorFlow/Keras adapter (available only when TensorFlow is installed)."""

from collections.abc import Sequence
from typing import Any

from ..operations import Operation
from .base import FrameworkAdapter


class TensorFlowAdapter(FrameworkAdapter):
    """Capture Keras Dense and Conv2D layer calls through temporary hooks."""

    name = "tensorflow"

    def operations(self, model: Any, example_inputs: Sequence[Any]) -> Sequence[Operation]:
        try:
            import tensorflow as tf
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[tensorflow] to use TensorFlowAdapter") from error
        captured: list[Operation] = []
        original_calls: list[tuple[Any, Any]] = []

        def shape(tensor: Any) -> tuple[int, ...]:
            values = tuple(tensor.shape)
            if any(value is None for value in values):
                raise ValueError("example inputs must provide concrete TensorFlow shapes")
            return tuple(int(value) for value in values)

        for layer in model.submodules:
            if not isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                continue
            previous_call = layer.call

            def wrapped(inputs: Any, *args: Any, _layer: Any = layer, _call: Any = previous_call, **kwargs: Any) -> Any:
                output = _call(inputs, *args, **kwargs)
                if isinstance(_layer, tf.keras.layers.Dense):
                    captured.append(Operation(_layer.name, "linear", (shape(inputs), shape(_layer.kernel)), shape(output), inputs.dtype.size))
                else:
                    # TF is NHWC while the portable Conv2D record is NCHW.
                    x = shape(inputs)
                    y = shape(output)
                    kernel = shape(_layer.kernel)  # HWIO -> OIHW
                    captured.append(Operation(_layer.name, "conv2d", ((x[0], x[3], x[1], x[2]), (kernel[3], kernel[2], kernel[0], kernel[1])), (y[0], y[3], y[1], y[2]), inputs.dtype.size))
                return output

            original_calls.append((layer, previous_call))
            layer.call = wrapped
        try:
            model(*example_inputs, training=False)
        finally:
            for layer, previous_call in original_calls:
                layer.call = previous_call
        return captured
