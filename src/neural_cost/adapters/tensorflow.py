"""TensorFlow/Keras adapter (available only when TensorFlow is installed)."""

from collections.abc import Sequence
from statistics import median
from time import perf_counter_ns
from typing import Any

from ..operations import Operation
from ..profiler import Measurement
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

        # Keras 3 removed ``Model.submodules``.  ``layers`` is the public API;
        # walking it also supports nested Sequential/Model containers.
        pending = list(getattr(model, "layers", ()))
        seen: set[int] = set()
        layers: list[Any] = []
        while pending:
            layer = pending.pop(0)
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            layers.append(layer)
            pending.extend(getattr(layer, "layers", ()))

        for layer in layers:
            if not isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                continue
            previous_call = layer.call

            def wrapped(inputs: Any, *args: Any, _layer: Any = layer, _call: Any = previous_call, **kwargs: Any) -> Any:
                output = _call(inputs, *args, **kwargs)
                if isinstance(_layer, tf.keras.layers.Dense):
                    captured.append(
                        Operation(
                            _layer.name,
                            "linear",
                            (shape(inputs), shape(_layer.kernel)),
                            shape(output),
                            inputs.dtype.size,
                            {
                                "parameter_bytes": int(_layer.count_params()) * inputs.dtype.size,
                                "parameter_id": id(_layer.kernel),
                            },
                        )
                    )
                else:
                    # TF is NHWC while the portable Conv2D record is NCHW.
                    x = shape(inputs)
                    y = shape(output)
                    kernel = shape(_layer.kernel)  # HWIO -> OIHW
                    captured.append(
                        Operation(
                            _layer.name,
                            "conv2d",
                            (
                                (x[0], x[3], x[1], x[2]),
                                (kernel[3], kernel[2], kernel[0], kernel[1]),
                            ),
                            (y[0], y[3], y[1], y[2]),
                            inputs.dtype.size,
                            {
                                "parameter_bytes": int(_layer.count_params()) * inputs.dtype.size,
                                "parameter_id": id(_layer.kernel),
                            },
                        )
                    )
                return output

            original_calls.append((layer, previous_call))
            layer.call = wrapped
        try:
            model(*example_inputs, training=False)
        finally:
            for layer, previous_call in original_calls:
                layer.call = previous_call
        return captured

    def benchmark(
        self, function: Any, *args: Any, warmup: int = 3, repeats: int = 10, **kwargs: Any
    ) -> Measurement:
        """Benchmark eager TensorFlow and collect supported GPU allocator telemetry."""
        try:
            import tensorflow as tf
        except ImportError as error:  # pragma: no cover - optional package
            raise ImportError("Install neural-cost[tensorflow] to use TensorFlowAdapter") from error
        if warmup < 0 or repeats < 1:
            raise ValueError("warmup must be non-negative and repeats must be at least one")

        gpus = tf.config.list_logical_devices("GPU")
        memory_device = "GPU:0" if gpus else None

        def wait() -> None:
            async_wait = getattr(tf.experimental, "async_wait", None)
            if async_wait is not None:
                async_wait()

        if memory_device is not None:
            try:
                tf.config.experimental.reset_memory_stats(memory_device)
            except (RuntimeError, ValueError):
                memory_device = None
        for _ in range(warmup):
            function(*args, **kwargs)
        wait()
        samples: list[float] = []
        for _ in range(repeats):
            start = perf_counter_ns()
            function(*args, **kwargs)
            wait()
            samples.append((perf_counter_ns() - start) / 1_000_000_000)
        info: dict[str, int] = {}
        if memory_device is not None:
            try:
                info = tf.config.experimental.get_memory_info(memory_device)
            except (RuntimeError, ValueError):
                pass
        return Measurement(
            median(samples),
            tuple(samples),
            info.get("peak"),
            gpus[0].name if gpus else "cpu",
            info.get("current"),
        )
