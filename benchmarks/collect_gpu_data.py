"""GPU benchmark: collect performance data across architectures × frameworks × batch sizes on GPU.

Writes results/benchmark_gpu_data.json when complete.

Key differences vs collect_data.py (CPU):
  - PyTorch: tensors/model moved to CUDA; CUDA events used for precise kernel timing
  - JAX: arrays placed on GPU device; XLA JIT targets GPU backend
  - TensorFlow: GPU device explicitly selected; tf.function + XLA JIT compile enabled
  - Larger batch sizes are swept (GPU favours high arithmetic intensity)
  - CUDA synchronisation barriers ensure only kernel time is measured
  - Gracefully falls back to CPU if no GPU accelerator is found

Run:
    python benchmarks/collect_gpu_data.py [--quick]
    python benchmarks/collect_gpu_data.py --device cuda  # explicit CUDA
    python benchmarks/collect_gpu_data.py --device mps   # Apple Silicon GPU
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

# Ensure src/ is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_cost import HardwareSpec, analyze_gap, profile_model
from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchAdapter
from neural_cost.hardware_detect import detect_hardware

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_DIM   = 128
NUM_HEADS   = 4
NUM_CLASSES = 10
IMG_SIZE    = 32

# GPU benefits most from larger batches (more parallelism).
BATCH_SIZES       = [8, 32, 128, 512]
QUICK_BATCH_SIZES = [32, 256]


# ---------------------------------------------------------------------------
# Result record (identical schema to CPU benchmark for easy comparison)
# ---------------------------------------------------------------------------

@dataclass
class BenchRecord:
    framework:          str
    variant:            str   # "baseline" | "compiled" | "jit" | "tf.function"
    architecture:       str
    batch:              int
    device:             str   # "cuda:0" / "mps" / "cpu-fallback" / "gpu" / etc.
    flops:              int
    param_bytes:        int
    total_bytes:        int
    arith_intensity:    float
    latency_median_ms:  float
    latency_mean_ms:    float
    latency_stddev_ms:  float
    latency_cv_pct:     float
    latency_p95_ms:     float
    roofline_efficiency: float
    achieved_gflops:    float
    achieved_gbw:       float
    bottleneck:         str


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _cuda_event_block(
    fn: Callable,
    args: tuple,
    warmup: int,
    repeats: int,
    device: Any,
) -> list[float]:
    """Use CUDA events for sub-millisecond GPU kernel timing."""
    import torch
    # Warmup — lets CUDA driver warm up and torch.compile finish tracing.
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize(device)
        samples.append(start.elapsed_time(end))  # milliseconds
    return samples


def _mps_block(
    fn: Callable,
    args: tuple,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Apple MPS has no CUDA events — use perf_counter with MPS sync."""
    import torch
    for _ in range(warmup):
        fn(*args)
    torch.mps.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args)
        torch.mps.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return samples


def _cpu_block(
    fn: Callable,
    args: tuple,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        fn(*args)
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return samples


def _jax_block(fn: Callable, args: tuple, warmup: int, repeats: int) -> list[float]:
    import jax
    def wait(v: Any) -> None:
        for leaf in jax.tree.leaves(v):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
    for _ in range(warmup):
        wait(fn(*args))
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        wait(fn(*args))
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return samples


def _tf_block(fn: Callable, args: tuple, warmup: int, repeats: int) -> list[float]:
    import tensorflow as tf
    async_wait = getattr(tf.experimental, "async_wait", None)
    def wait():
        if async_wait:
            async_wait()
    for _ in range(warmup):
        fn(*args); wait()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args); wait()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return samples


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    n = len(s)
    med = median(s)
    mn  = mean(s)
    sd  = stdev(s) if n > 1 else 0.0
    cv  = 100 * sd / mn if mn > 0 else 0.0
    p95 = s[min(int(0.95 * n), n - 1)]
    return dict(median=med, mean=mn, stddev=sd, cv=cv, p95=p95)


# ---------------------------------------------------------------------------
# Detect best available torch device
# ---------------------------------------------------------------------------

def _detect_torch_device(requested: str | None) -> tuple[Any, str]:
    """Return (torch.device, device_label) for the best available GPU."""
    import torch
    if requested:
        dev = torch.device(requested)
        return dev, str(dev)
    if torch.cuda.is_available():
        dev = torch.device("cuda", 0)
        name = torch.cuda.get_device_name(0)
        return dev, f"cuda:0 ({name})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps (Apple Silicon GPU)"
    return torch.device("cpu"), "cpu (no GPU found)"


# ---------------------------------------------------------------------------
# PyTorch GPU model builders
# ---------------------------------------------------------------------------

def _torch_ff_dnn(batch: int, device: Any):
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(784, EMBED_DIM), torch.nn.ReLU(),
        torch.nn.LayerNorm(EMBED_DIM),
        torch.nn.Linear(EMBED_DIM, EMBED_DIM), torch.nn.ReLU(),
        torch.nn.LayerNorm(EMBED_DIM),
        torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
    ).eval().to(device)
    x = torch.randn(batch, 784, device=device)
    return model, (x,)


def _torch_cnn(batch: int, device: Any):
    import torch
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, EMBED_DIM // 2, 3, padding=1), torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM // 2), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(EMBED_DIM // 2, EMBED_DIM, 3, padding=1), torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM), torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(), torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
    ).eval().to(device)
    x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=device)
    return model, (x,)


def _torch_rnn(batch: int, device: Any):
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.RNN(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc  = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1])
    m = M().eval().to(device)
    x = torch.randn(batch, 32, EMBED_DIM, device=device)
    return m, (x,)


def _torch_lstm(batch: int, device: Any):
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc   = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])
    m = M().eval().to(device)
    x = torch.randn(batch, 32, EMBED_DIM, device=device)
    return m, (x,)


def _torch_transformer(batch: int, device: Any):
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1  = torch.nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=False)
            self.norm1a = torch.nn.LayerNorm(EMBED_DIM)
            self.ff1    = torch.nn.Linear(EMBED_DIM, EMBED_DIM * 4)
            self.ff1b   = torch.nn.Linear(EMBED_DIM * 4, EMBED_DIM)
            self.norm1b = torch.nn.LayerNorm(EMBED_DIM)
            self.attn2  = torch.nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=False)
            self.norm2a = torch.nn.LayerNorm(EMBED_DIM)
            self.ff2    = torch.nn.Linear(EMBED_DIM, EMBED_DIM * 4)
            self.ff2b   = torch.nn.Linear(EMBED_DIM * 4, EMBED_DIM)
            self.norm2b = torch.nn.LayerNorm(EMBED_DIM)
            self.head   = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)
        def forward(self, x):
            xt = x.permute(1, 0, 2)
            a1, _ = self.attn1(xt, xt, xt)
            xt = self.norm1a(xt + a1)
            xt = self.norm1b(xt + self.ff1b(torch.relu(self.ff1(xt))))
            a2, _ = self.attn2(xt, xt, xt)
            xt = self.norm2a(xt + a2)
            xt = self.norm2b(xt + self.ff2b(torch.relu(self.ff2(xt))))
            return self.head(xt.mean(0))
    m = M().eval().to(device)
    x = torch.randn(batch, 32, EMBED_DIM, device=device)
    return m, (x,)


TORCH_GPU_BUILDERS = {
    "FF DNN":      _torch_ff_dnn,
    "CNN":         _torch_cnn,
    "RNN":         _torch_rnn,
    "LSTM":        _torch_lstm,
    "Transformer": _torch_transformer,
}

ARCHITECTURES = ["FF DNN", "CNN", "RNN", "LSTM", "Transformer"]


# ---------------------------------------------------------------------------
# JAX GPU model builders (identical math, jnp arrays on GPU device)
# ---------------------------------------------------------------------------

def _jax_ff_dnn_gpu(batch: int, jax_device: Any):
    import jax
    import jax.numpy as jnp
    def _put(x): return jax.device_put(x, jax_device)
    w1 = _put(jnp.ones((784, EMBED_DIM)))
    w2 = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    w3 = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))
    def model(x, _w1=w1, _w2=w2, _w3=w3):
        return jnp.tanh(jnp.tanh(x @ _w1) @ _w2) @ _w3
    return model, (_put(jnp.ones((batch, 784))), w1, w2, w3)


def _jax_cnn_gpu(batch: int, jax_device: Any):
    import jax
    import jax.lax as lax
    import jax.numpy as jnp
    def _put(x): return jax.device_put(x, jax_device)
    k1  = _put(jnp.ones((EMBED_DIM // 2, 3, 3, 3)))
    k2  = _put(jnp.ones((EMBED_DIM, EMBED_DIM // 2, 3, 3)))
    wfc = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))
    def model(x, _k1=k1, _k2=k2, _wfc=wfc):
        y = jnp.tanh(lax.conv_general_dilated(x, _k1, (1,1), "SAME", dimension_numbers=("NCHW","OIHW","NCHW")))
        y = jnp.tanh(lax.conv_general_dilated(y, _k2, (2,2), "SAME", dimension_numbers=("NCHW","OIHW","NCHW")))
        return y.mean(axis=(2, 3)) @ _wfc
    return model, (_put(jnp.ones((batch, 3, IMG_SIZE, IMG_SIZE))), k1, k2, wfc)


def _jax_rnn_gpu(batch: int, jax_device: Any):
    import jax
    import jax.numpy as jnp
    T = 32
    def _put(x): return jax.device_put(x, jax_device)
    wih = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    whh = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wfc = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))
    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], EMBED_DIM))
        for t in range(T):
            h = jnp.tanh(x[:, t] @ _wih + h @ _whh)
        return h @ _wfc
    return model, (_put(jnp.ones((batch, T, EMBED_DIM))), wih, whh, wfc)


def _jax_lstm_gpu(batch: int, jax_device: Any):
    import jax
    import jax.numpy as jnp
    T, G = 32, 4
    def _put(x): return jax.device_put(x, jax_device)
    wih = _put(jnp.ones((EMBED_DIM, G * EMBED_DIM)))
    whh = _put(jnp.ones((EMBED_DIM, G * EMBED_DIM)))
    wfc = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))
    def sigmoid(x): return 1.0 / (1.0 + jnp.exp(-x))
    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], EMBED_DIM))
        c = jnp.zeros((x.shape[0], EMBED_DIM))
        for t in range(T):
            gv = x[:, t] @ _wih + h @ _whh
            i, f, g, o = jnp.split(gv, 4, axis=-1)
            c = sigmoid(f) * c + sigmoid(i) * jnp.tanh(g)
            h = sigmoid(o) * jnp.tanh(c)
        return h @ _wfc
    return model, (_put(jnp.ones((batch, T, EMBED_DIM))), wih, whh, wfc)


def _jax_transformer_gpu(batch: int, jax_device: Any):
    import jax
    import jax.numpy as jnp
    T  = 32
    hd = EMBED_DIM // NUM_HEADS
    def _put(x): return jax.device_put(x, jax_device)
    wq  = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wk  = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wv  = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wo  = _put(jnp.ones((EMBED_DIM, EMBED_DIM)))
    wf1 = _put(jnp.ones((EMBED_DIM, EMBED_DIM * 4)))
    wf2 = _put(jnp.ones((EMBED_DIM * 4, EMBED_DIM)))
    wfc = _put(jnp.ones((EMBED_DIM, NUM_CLASSES)))
    scale = hd ** -0.5
    def attn(x, wq_, wk_, wv_, wo_):
        q, k, v = x @ wq_, x @ wk_, x @ wv_
        s = (q * scale) @ k.transpose(0, 2, 1)
        w = jnp.exp(s) / jnp.exp(s).sum(-1, keepdims=True)
        return (w @ v) @ wo_
    def model(x, _wq=wq, _wk=wk, _wv=wv, _wo=wo, _wf1=wf1, _wf2=wf2, _wfc=wfc):
        x = x + attn(x, _wq, _wk, _wv, _wo)
        x = x + jnp.tanh(x @ _wf1) @ _wf2
        return x.mean(1) @ _wfc
    return model, (_put(jnp.ones((batch, T, EMBED_DIM))), wq, wk, wv, wo, wf1, wf2, wfc)


JAX_GPU_BUILDERS = {
    "FF DNN":      _jax_ff_dnn_gpu,
    "CNN":         _jax_cnn_gpu,
    "RNN":         _jax_rnn_gpu,
    "LSTM":        _jax_lstm_gpu,
    "Transformer": _jax_transformer_gpu,
}


# ---------------------------------------------------------------------------
# TensorFlow GPU model builders
# ---------------------------------------------------------------------------

def _tf_ff_dnn_gpu(batch: int, tf_device: str):
    import tensorflow as tf
    with tf.device(tf_device):
        m = tf.keras.Sequential([
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(NUM_CLASSES),
        ])
        x = tf.ones((batch, 784))
        m(x)
    return m, (x,)


def _tf_cnn_gpu(batch: int, tf_device: str):
    import tensorflow as tf
    with tf.device(tf_device):
        m = tf.keras.Sequential([
            tf.keras.layers.Conv2D(EMBED_DIM // 2, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(EMBED_DIM, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(NUM_CLASSES),
        ])
        x = tf.ones((batch, IMG_SIZE, IMG_SIZE, 3))
        m(x)
    return m, (x,)


def _tf_rnn_gpu(batch: int, tf_device: str):
    import tensorflow as tf
    with tf.device(tf_device):
        inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
        x   = tf.keras.layers.GRU(EMBED_DIM, return_sequences=True)(inp)
        x   = tf.keras.layers.GRU(EMBED_DIM)(x)
        out = tf.keras.layers.Dense(NUM_CLASSES)(x)
        m   = tf.keras.Model(inp, out)
        xi  = tf.ones((batch, 32, EMBED_DIM))
        m(xi)
    return m, (xi,)


def _tf_lstm_gpu(batch: int, tf_device: str):
    import tensorflow as tf
    with tf.device(tf_device):
        inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
        x   = tf.keras.layers.LSTM(EMBED_DIM, return_sequences=True)(inp)
        x   = tf.keras.layers.LSTM(EMBED_DIM)(x)
        out = tf.keras.layers.Dense(NUM_CLASSES)(x)
        m   = tf.keras.Model(inp, out)
        xi  = tf.ones((batch, 32, EMBED_DIM))
        m(xi)
    return m, (xi,)


def _tf_transformer_gpu(batch: int, tf_device: str):
    import tensorflow as tf
    with tf.device(tf_device):
        inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
        x   = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)(inp, inp)
        x   = tf.keras.layers.LayerNormalization()(inp + x)
        ff  = tf.keras.layers.Dense(EMBED_DIM * 4, activation="relu")(x)
        ff  = tf.keras.layers.Dense(EMBED_DIM)(ff)
        x   = tf.keras.layers.LayerNormalization()(x + ff)
        x2  = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)(x, x)
        x2  = tf.keras.layers.LayerNormalization()(x + x2)
        ff2 = tf.keras.layers.Dense(EMBED_DIM * 4, activation="relu")(x2)
        ff2 = tf.keras.layers.Dense(EMBED_DIM)(ff2)
        x2  = tf.keras.layers.LayerNormalization()(x2 + ff2)
        p   = tf.keras.layers.GlobalAveragePooling1D()(x2)
        out = tf.keras.layers.Dense(NUM_CLASSES)(p)
        m   = tf.keras.Model(inp, out)
        xi  = tf.ones((batch, 32, EMBED_DIM))
        m(xi)
    return m, (xi,)


TF_GPU_BUILDERS = {
    "FF DNN":      _tf_ff_dnn_gpu,
    "CNN":         _tf_cnn_gpu,
    "RNN":         _tf_rnn_gpu,
    "LSTM":        _tf_lstm_gpu,
    "Transformer": _tf_transformer_gpu,
}


# ---------------------------------------------------------------------------
# GPU device detection helpers
# ---------------------------------------------------------------------------

def _detect_jax_gpu() -> tuple[Any | None, str]:
    """Return (jax.Device, label) for the first available GPU/accelerator."""
    try:
        import jax
        # Try GPU first, then TPU, fall back to CPU
        for backend in ("gpu", "tpu", "cpu"):
            try:
                devs = jax.devices(backend)
                if devs:
                    dev = devs[0]
                    return dev, f"{backend}:{dev.id} ({dev.device_kind})"
            except RuntimeError:
                pass
        return None, "cpu-fallback"
    except ImportError:
        return None, "jax-not-installed"


def _detect_tf_gpu() -> tuple[str, str]:
    """Return (tf_device_str, label) for the first available GPU."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return "/GPU:0", f"GPU:0 ({gpus[0].name})"
        return "/CPU:0", "cpu-fallback (no TF GPU)"
    except ImportError:
        return "/CPU:0", "tf-not-installed"


# ---------------------------------------------------------------------------
# Per-framework GPU evaluation
# ---------------------------------------------------------------------------

def _make_gap(cost: Any, st: dict, hardware: HardwareSpec) -> Any:
    """Build a fake measurement object compatible with analyze_gap."""
    return analyze_gap(
        cost,
        type("M", (), {
            "median_seconds":   st["median"] / 1e3,
            "samples_seconds":  tuple(s / 1e3 for s in [st["median"]] * 2),
        })(),
        hardware,
    )


def eval_torch_gpu(
    hardware: HardwareSpec,
    batches: list[int],
    warmup: int,
    repeats: int,
    torch_device: Any,
    device_label: str,
) -> list[BenchRecord]:
    import torch
    adapter = TorchAdapter()
    records: list[BenchRecord] = []

    is_cuda = str(torch_device).startswith("cuda")
    is_mps  = str(torch_device) == "mps"

    for arch in ARCHITECTURES:
        for batch in batches:
            try:
                model, inputs = TORCH_GPU_BUILDERS[arch](batch, torch_device)
            except Exception as exc:
                print(f"  PyTorch GPU builder {arch} B={batch}: {exc}")
                continue

            # Profile on CPU copies (adapter hooks work on any device)
            try:
                cpu_model, cpu_inputs = TORCH_GPU_BUILDERS[arch](batch, torch.device("cpu"))
                prof = profile_model(cpu_model, cpu_inputs, adapter)
                cost = prof.cost
                mem  = prof.memory
            except Exception:
                continue

            def _time(fn: Callable, args: tuple) -> list[float]:
                if is_cuda:
                    return _cuda_event_block(fn, args, warmup, repeats, torch_device)
                elif is_mps:
                    return _mps_block(fn, args, warmup, repeats)
                else:
                    return _cpu_block(fn, args, warmup, repeats)

            # Baseline (eager GPU)
            try:
                samp = _time(model, inputs)
                st   = _stats(samp)
                gap  = _make_gap(cost, st, hardware)
                records.append(BenchRecord(
                    "PyTorch", "baseline", arch, batch, device_label,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  PyTorch GPU baseline {arch} B={batch}: {exc}")

            # Optimised: torch.compile (Inductor backend, GPU)
            try:
                compiled = torch.compile(model)
                # Trigger compilation
                if is_cuda:
                    for _ in range(max(3, warmup)):
                        compiled(*inputs)
                    torch.cuda.synchronize(torch_device)
                elif is_mps:
                    for _ in range(max(3, warmup)):
                        compiled(*inputs)
                    torch.mps.synchronize()
                else:
                    for _ in range(max(3, warmup)):
                        compiled(*inputs)
                samp = _time(compiled, inputs)
                st   = _stats(samp)
                gap  = _make_gap(cost, st, hardware)
                records.append(BenchRecord(
                    "PyTorch", "compiled", arch, batch, device_label,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  PyTorch GPU compiled {arch} B={batch}: {exc}")

    return records


def eval_jax_gpu(
    hardware: HardwareSpec,
    batches: list[int],
    warmup: int,
    repeats: int,
    jax_device: Any,
    device_label: str,
) -> list[BenchRecord]:
    import jax
    adapter = JaxAdapter()
    records: list[BenchRecord] = []

    for arch in ARCHITECTURES:
        for batch in batches:
            try:
                model, inputs = JAX_GPU_BUILDERS[arch](batch, jax_device)
            except Exception as exc:
                print(f"  JAX GPU builder {arch} B={batch}: {exc}")
                continue

            # Static profile uses CPU-side JAX adapter (shape-only)
            try:
                import jax.numpy as jnp
                cpu_model, cpu_inputs = JAX_GPU_BUILDERS[arch](
                    batch, jax.devices("cpu")[0]
                )
                prof = profile_model(cpu_model, cpu_inputs, adapter)
                cost = prof.cost
                mem  = prof.memory
            except Exception:
                continue

            # Baseline (eager on GPU device)
            try:
                samp = _jax_block(model, inputs, warmup, repeats)
                st   = _stats(samp)
                gap  = _make_gap(cost, st, hardware)
                records.append(BenchRecord(
                    "JAX", "baseline", arch, batch, device_label,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  JAX GPU baseline {arch} B={batch}: {exc}")

            # Optimised: jax.jit
            try:
                jit_model = jax.jit(model)
                # Trigger compilation
                wait = lambda v: [
                    leaf.block_until_ready()
                    for leaf in jax.tree.leaves(v)
                    if hasattr(leaf, "block_until_ready")
                ]
                wait(jit_model(*inputs))
                samp = _jax_block(jit_model, inputs, warmup, repeats)
                st   = _stats(samp)
                gap  = _make_gap(cost, st, hardware)
                records.append(BenchRecord(
                    "JAX", "jit", arch, batch, device_label,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  JAX GPU jit {arch} B={batch}: {exc}")

    return records


def eval_tensorflow_gpu(
    hardware: HardwareSpec,
    batches: list[int],
    warmup: int,
    repeats: int,
    tf_device: str,
    device_label: str,
) -> list[BenchRecord]:
    import tensorflow as tf
    adapter = TensorFlowAdapter()
    records: list[BenchRecord] = []

    for arch in ARCHITECTURES:
        for batch in batches:
            try:
                model, inputs = TF_GPU_BUILDERS[arch](batch, tf_device)
            except Exception as exc:
                print(f"  TF GPU builder {arch} B={batch}: {exc}")
                continue

            try:
                prof = profile_model(model, inputs, adapter)
                cost = prof.cost
                mem  = prof.memory
            except Exception:
                continue

            fn = lambda *a: model(*a, training=False)

            # Baseline (eager GPU)
            try:
                samp = _tf_block(fn, inputs, warmup, repeats)
                st   = _stats(samp)
                gap  = _make_gap(cost, st, hardware)
                records.append(BenchRecord(
                    "TensorFlow", "baseline", arch, batch, device_label,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  TF GPU baseline {arch} B={batch}: {exc}")

            # Optimised: tf.function + XLA JIT compile
            try:
                # Enable XLA JIT for GPU — significant gains on matmul/conv
                tf_fn = tf.function(fn, jit_compile=True)
                for _ in range(3):
                    tf_fn(*inputs)
                samp = _tf_block(tf_fn, inputs, warmup, repeats)
                st   = _stats(samp)
                gap  = _make_gap(cost, st, hardware)
                records.append(BenchRecord(
                    "TensorFlow", "tf.function+XLA", arch, batch, device_label,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                # Fallback to graph mode without XLA if XLA compile fails
                try:
                    tf_fn = tf.function(fn, jit_compile=False)
                    for _ in range(3):
                        tf_fn(*inputs)
                    samp = _tf_block(tf_fn, inputs, warmup, repeats)
                    st   = _stats(samp)
                    gap  = _make_gap(cost, st, hardware)
                    records.append(BenchRecord(
                        "TensorFlow", "tf.function", arch, batch, device_label,
                        cost.flops, mem.parameter_bytes, cost.total_bytes,
                        cost.arithmetic_intensity,
                        st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                        gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                        gap.bottleneck,
                    ))
                except Exception as exc2:
                    print(f"  TF GPU tf.function {arch} B={batch}: {exc2}")

    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",  action="store_true", help="Fewer batches/repeats for fast iteration")
    parser.add_argument("--warmup",  type=int, default=15)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--device",  type=str, default=None,
                        help="PyTorch device override: 'cuda', 'cuda:1', 'mps', 'cpu'")
    parser.add_argument("--peak-flops",       type=float, default=None,
                        help="Override peak FLOP/s (e.g. 312e12 for A100)")
    parser.add_argument("--memory-bandwidth", type=float, default=None,
                        help="Override memory bandwidth in bytes/s (e.g. 2.0e12 for A100)")
    args = parser.parse_args()

    warmup  = 5  if args.quick else args.warmup
    repeats = 15 if args.quick else args.repeats
    batches = QUICK_BATCH_SIZES if args.quick else BATCH_SIZES

    # ------------------------------------------------------------------
    # Detect GPU and build a HardwareSpec for the GPU if one is found.
    # The CPU-targeted detect_hardware() is used as the base; GPU specs
    # can be supplied on the command line or will be estimated from the
    # GPU name when a CUDA device is available.
    # ------------------------------------------------------------------
    print("Detecting hardware…", flush=True)
    hardware, detection = detect_hardware()

    # Attempt to read better GPU specs from nvidia-smi / torch.cuda
    gpu_name = hardware.name
    peak_flops = hardware.peak_flops
    mem_bw     = hardware.memory_bandwidth

    if importlib.util.find_spec("torch") is not None:
        import torch
        torch_dev, torch_label = _detect_torch_device(args.device)
        if str(torch_dev).startswith("cuda") and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch_dev)
            gpu_name = props.name
            # SM count × 2 (FP32 ALUs per SM on Ampere/Ada) × 2 ops/clock × clock
            # This is an approximation; --peak-flops can override.
            clock_hz = props.max_clock_rate * 1000  # kHz → Hz
            n_sm     = props.multi_processor_count
            # Common CUDA arch: 128 CUDA cores/SM (Ampere) → 2 FP32 ops/clock/core
            peak_flops = n_sm * 128 * 2 * clock_hz
            mem_bw     = props.memory_bandwidth  # bytes/s (PyTorch 2.x)
            print(f"  GPU detected: {gpu_name}")
            print(f"  SMs={n_sm}  clock={clock_hz/1e9:.2f} GHz  "
                  f"est. peak={peak_flops/1e12:.1f} TFLOP/s  "
                  f"bw={mem_bw/1e9:.0f} GB/s")
        elif str(torch_dev) == "mps":
            gpu_name  = torch_label
            # Apple Silicon — use detected values (chip table has GPU specs)
            print(f"  Device: {torch_label}")
        else:
            torch_label = f"cpu-fallback ({hardware.name})"
            print(f"  No GPU found — running on {hardware.name}")
    else:
        torch_dev, torch_label = None, "torch-not-installed"

    if args.peak_flops:
        peak_flops = args.peak_flops
    if args.memory_bandwidth:
        mem_bw = args.memory_bandwidth

    gpu_hardware = HardwareSpec(gpu_name, peak_flops, mem_bw)

    hw_meta = {
        "name":               gpu_hardware.name,
        "peak_flops":         gpu_hardware.peak_flops,
        "memory_bandwidth":   gpu_hardware.memory_bandwidth,
        "ridge_point":        gpu_hardware.ridge_point,
        "source":             detection.source,
        "measured_bw_gb_s":   detection.measured_bandwidth_gb_s,
        "torch_device":       torch_label,
    }
    print(f"\n  {gpu_hardware.name}  "
          f"{gpu_hardware.peak_flops/1e12:.2f} TFLOP/s  "
          f"{gpu_hardware.memory_bandwidth/1e9:.0f} GB/s")
    print(f"  Batches={batches}  warmup={warmup}  repeats={repeats}\n")

    all_records: list[BenchRecord] = []

    # PyTorch GPU
    if importlib.util.find_spec("torch") is not None and torch_dev is not None:
        print(f"── PyTorch ({torch_label}) ─────────────────────────────────")
        recs = eval_torch_gpu(gpu_hardware, batches, warmup, repeats, torch_dev, torch_label)
        all_records.extend(recs)
        for r in recs:
            print(f"  {r.variant:<16} {r.architecture:<14} B={r.batch:<4}  "
                  f"{r.latency_median_ms:7.2f} ms  {r.roofline_efficiency:.1%}  {r.bottleneck}")
        print()

    # JAX GPU
    if importlib.util.find_spec("jax") is not None:
        jax_dev, jax_label = _detect_jax_gpu()
        if jax_dev is not None:
            print(f"── JAX ({jax_label}) ─────────────────────────────────")
            recs = eval_jax_gpu(gpu_hardware, batches, warmup, repeats, jax_dev, jax_label)
            all_records.extend(recs)
            for r in recs:
                print(f"  {r.variant:<16} {r.architecture:<14} B={r.batch:<4}  "
                      f"{r.latency_median_ms:7.2f} ms  {r.roofline_efficiency:.1%}  {r.bottleneck}")
            print()
        else:
            print("[skip] JAX: no GPU/accelerator device available")
    else:
        print("[skip] JAX not installed")

    # TensorFlow GPU
    if importlib.util.find_spec("tensorflow") is not None:
        tf_dev, tf_label = _detect_tf_gpu()
        print(f"── TensorFlow ({tf_label}) ─────────────────────────────────")
        recs = eval_tensorflow_gpu(gpu_hardware, batches, warmup, repeats, tf_dev, tf_label)
        all_records.extend(recs)
        for r in recs:
            print(f"  {r.variant:<16} {r.architecture:<14} B={r.batch:<4}  "
                  f"{r.latency_median_ms:7.2f} ms  {r.roofline_efficiency:.1%}  {r.bottleneck}")
        print()
    else:
        print("[skip] TensorFlow not installed")

    out_dir  = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "benchmark_gpu_data.json"
    payload  = {"hardware": hw_meta, "records": [asdict(r) for r in all_records]}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {len(all_records)} records → {out_path}")


if __name__ == "__main__":
    main()
