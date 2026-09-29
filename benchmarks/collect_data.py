"""Scientific evaluation: collect benchmark data across architectures × frameworks × batch sizes.

Writes results/benchmark_data.json when complete.

Key improvement over the basic example:
  - JAX models are wrapped with jax.jit() (the dominant performance lever)
  - PyTorch models are wrapped with torch.compile() where supported
  - Baseline (un-compiled) vs optimised variants are both measured
  - Multiple batch sizes sweep from memory-bound to compute-bound regimes
  - 15 warmup + 40 repeat iterations for stable statistics

Run:
    python benchmarks/collect_data.py [--quick]
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
from neural_cost.adapters.base import FrameworkAdapter
from neural_cost.hardware_detect import detect_hardware

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_DIM  = 128
NUM_HEADS  = 4
NUM_CLASSES = 10
IMG_SIZE   = 32

BATCH_SIZES = [1, 8, 32, 128]

# For --quick flag
QUICK_BATCH_SIZES = [8, 64]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class BenchRecord:
    framework: str
    variant: str          # "baseline" or "optimised"
    architecture: str
    batch: int
    flops: int
    param_bytes: int
    total_bytes: int
    arith_intensity: float
    latency_median_ms: float
    latency_mean_ms: float
    latency_stddev_ms: float
    latency_cv_pct: float  # coefficient of variation
    latency_p95_ms: float
    roofline_efficiency: float
    achieved_gflops: float
    achieved_gbw: float
    bottleneck: str


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _jax_block(fn: Callable, args: tuple, warmup: int, repeats: int) -> list[float]:
    import jax
    def wait(v: Any) -> None:
        for leaf in jax.tree.leaves(v):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
    for _ in range(warmup):
        wait(fn(*args))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        wait(fn(*args))
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return samples


def _torch_block(fn: Callable, args: tuple, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn(*args)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return samples


def _tf_block(fn: Callable, args: tuple, warmup: int, repeats: int) -> list[float]:
    import tensorflow as tf
    async_wait = getattr(tf.experimental, "async_wait", None)
    def wait():
        if async_wait:
            async_wait()
    for _ in range(warmup):
        fn(*args)
        wait()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args)
        wait()
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
# PyTorch model builders  (return model, call_fn, inputs_for_adapter)
# ---------------------------------------------------------------------------

def _torch_ff_dnn(batch: int):
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(784, EMBED_DIM), torch.nn.ReLU(),
        torch.nn.LayerNorm(EMBED_DIM),
        torch.nn.Linear(EMBED_DIM, EMBED_DIM), torch.nn.ReLU(),
        torch.nn.LayerNorm(EMBED_DIM),
        torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
    ).eval()
    x = torch.randn(batch, 784)
    return model, (x,)

def _torch_cnn(batch: int):
    import torch
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, EMBED_DIM // 2, 3, padding=1), torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM // 2), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(EMBED_DIM // 2, EMBED_DIM, 3, padding=1), torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM), torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(), torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
    ).eval()
    x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)
    return model, (x,)

def _torch_rnn(batch: int):
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.RNN(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc  = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1])
    m = M().eval()
    x = torch.randn(batch, 32, EMBED_DIM)
    return m, (x,)

def _torch_lstm(batch: int):
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc   = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])
    m = M().eval()
    x = torch.randn(batch, 32, EMBED_DIM)
    return m, (x,)

def _torch_transformer(batch: int):
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
    m = M().eval()
    x = torch.randn(batch, 32, EMBED_DIM)
    return m, (x,)


TORCH_BUILDERS = {
    "FF DNN": _torch_ff_dnn,
    "CNN": _torch_cnn,
    "RNN": _torch_rnn,
    "LSTM": _torch_lstm,
    "Transformer": _torch_transformer,
}


# ---------------------------------------------------------------------------
# JAX model builders
# ---------------------------------------------------------------------------

def _jax_ff_dnn(batch: int):
    import jax.numpy as jnp
    w1 = jnp.ones((784, EMBED_DIM))
    w2 = jnp.ones((EMBED_DIM, EMBED_DIM))
    w3 = jnp.ones((EMBED_DIM, NUM_CLASSES))
    def model(x, _w1=w1, _w2=w2, _w3=w3):
        return jnp.tanh(jnp.tanh(x @ _w1) @ _w2) @ _w3
    return model, (jnp.ones((batch, 784)), w1, w2, w3)

def _jax_cnn(batch: int):
    import jax.lax as lax
    import jax.numpy as jnp
    k1  = jnp.ones((EMBED_DIM // 2, 3, 3, 3))
    k2  = jnp.ones((EMBED_DIM, EMBED_DIM // 2, 3, 3))
    wfc = jnp.ones((EMBED_DIM, NUM_CLASSES))
    def model(x, _k1=k1, _k2=k2, _wfc=wfc):
        y = jnp.tanh(lax.conv_general_dilated(x, _k1, (1,1), "SAME", dimension_numbers=("NCHW","OIHW","NCHW")))
        y = jnp.tanh(lax.conv_general_dilated(y, _k2, (2,2), "SAME", dimension_numbers=("NCHW","OIHW","NCHW")))
        return y.mean(axis=(2, 3)) @ _wfc
    return model, (jnp.ones((batch, 3, IMG_SIZE, IMG_SIZE)), k1, k2, wfc)

def _jax_rnn(batch: int):
    import jax.numpy as jnp
    T = 32
    wih = jnp.ones((EMBED_DIM, EMBED_DIM))
    whh = jnp.ones((EMBED_DIM, EMBED_DIM))
    wfc = jnp.ones((EMBED_DIM, NUM_CLASSES))
    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], EMBED_DIM))
        for t in range(T):
            h = jnp.tanh(x[:, t] @ _wih + h @ _whh)
        return h @ _wfc
    return model, (jnp.ones((batch, T, EMBED_DIM)), wih, whh, wfc)

def _jax_lstm(batch: int):
    import jax.numpy as jnp
    T, G = 32, 4
    wih = jnp.ones((EMBED_DIM, G * EMBED_DIM))
    whh = jnp.ones((EMBED_DIM, G * EMBED_DIM))
    wfc = jnp.ones((EMBED_DIM, NUM_CLASSES))
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
    return model, (jnp.ones((batch, T, EMBED_DIM)), wih, whh, wfc)

def _jax_transformer(batch: int):
    import jax.numpy as jnp
    T       = 32
    hd      = EMBED_DIM // NUM_HEADS
    wq  = jnp.ones((EMBED_DIM, EMBED_DIM))
    wk  = jnp.ones((EMBED_DIM, EMBED_DIM))
    wv  = jnp.ones((EMBED_DIM, EMBED_DIM))
    wo  = jnp.ones((EMBED_DIM, EMBED_DIM))
    wf1 = jnp.ones((EMBED_DIM, EMBED_DIM * 4))
    wf2 = jnp.ones((EMBED_DIM * 4, EMBED_DIM))
    wfc = jnp.ones((EMBED_DIM, NUM_CLASSES))
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
    return model, (jnp.ones((batch, T, EMBED_DIM)), wq, wk, wv, wo, wf1, wf2, wfc)


JAX_BUILDERS = {
    "FF DNN":      _jax_ff_dnn,
    "CNN":         _jax_cnn,
    "RNN":         _jax_rnn,
    "LSTM":        _jax_lstm,
    "Transformer": _jax_transformer,
}


# ---------------------------------------------------------------------------
# TF model builders
# ---------------------------------------------------------------------------

def _tf_ff_dnn(batch: int):
    import tensorflow as tf
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

def _tf_cnn(batch: int):
    import tensorflow as tf
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

def _tf_rnn(batch: int):
    import tensorflow as tf
    inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
    x   = tf.keras.layers.GRU(EMBED_DIM, return_sequences=True)(inp)
    x   = tf.keras.layers.GRU(EMBED_DIM)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)
    m   = tf.keras.Model(inp, out)
    xi  = tf.ones((batch, 32, EMBED_DIM))
    m(xi)
    return m, (xi,)

def _tf_lstm(batch: int):
    import tensorflow as tf
    inp = tf.keras.layers.Input(shape=(32, EMBED_DIM))
    x   = tf.keras.layers.LSTM(EMBED_DIM, return_sequences=True)(inp)
    x   = tf.keras.layers.LSTM(EMBED_DIM)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)
    m   = tf.keras.Model(inp, out)
    xi  = tf.ones((batch, 32, EMBED_DIM))
    m(xi)
    return m, (xi,)

def _tf_transformer(batch: int):
    import tensorflow as tf
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


TF_BUILDERS = {
    "FF DNN":      _tf_ff_dnn,
    "CNN":         _tf_cnn,
    "RNN":         _tf_rnn,
    "LSTM":        _tf_lstm,
    "Transformer": _tf_transformer,
}

ARCHITECTURES = ["FF DNN", "CNN", "RNN", "LSTM", "Transformer"]


# ---------------------------------------------------------------------------
# Per-framework evaluation
# ---------------------------------------------------------------------------

def eval_torch(hardware: HardwareSpec, batches: list[int], warmup: int, repeats: int) -> list[BenchRecord]:
    import torch
    adapter = TorchAdapter()
    records: list[BenchRecord] = []
    for arch in ARCHITECTURES:
        for batch in batches:
            model, inputs = TORCH_BUILDERS[arch](batch)
            # Static profile
            try:
                prof = profile_model(model, inputs, adapter)
            except Exception:
                continue
            cost  = prof.cost
            mem   = prof.memory
            # Baseline
            try:
                samp  = _torch_block(model, inputs, warmup, repeats)
                st    = _stats(samp)
                gap   = analyze_gap(cost, type("M", (), {"median_seconds": st["median"] / 1e3, "samples_seconds": tuple(s / 1e3 for s in samp)})(), hardware)
                records.append(BenchRecord(
                    "PyTorch", "baseline", arch, batch,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  PyTorch baseline {arch} B={batch}: {exc}")
            # Optimised: torch.compile
            try:
                compiled = torch.compile(model)
                # Warmup compile
                for _ in range(max(3, warmup)):
                    compiled(*inputs)
                samp  = _torch_block(compiled, inputs, 0, repeats)
                st    = _stats(samp)
                gap   = analyze_gap(cost, type("M", (), {"median_seconds": st["median"] / 1e3, "samples_seconds": tuple(s / 1e3 for s in samp)})(), hardware)
                records.append(BenchRecord(
                    "PyTorch", "compiled", arch, batch,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  PyTorch compiled {arch} B={batch}: {exc}")
    return records


def eval_jax(hardware: HardwareSpec, batches: list[int], warmup: int, repeats: int) -> list[BenchRecord]:
    import jax
    adapter = JaxAdapter()
    records: list[BenchRecord] = []
    for arch in ARCHITECTURES:
        for batch in batches:
            model, inputs = JAX_BUILDERS[arch](batch)
            try:
                prof = profile_model(model, inputs, adapter)
            except Exception:
                continue
            cost  = prof.cost
            mem   = prof.memory
            # Baseline (eager)
            try:
                samp = _jax_block(model, inputs, warmup, repeats)
                st   = _stats(samp)
                gap  = analyze_gap(cost, type("M", (), {"median_seconds": st["median"] / 1e3, "samples_seconds": tuple(s / 1e3 for s in samp)})(), hardware)
                records.append(BenchRecord(
                    "JAX", "baseline", arch, batch,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  JAX baseline {arch} B={batch}: {exc}")
            # Optimised: jax.jit
            try:
                jit_model = jax.jit(model)
                # Trigger compilation
                import jax
                wait = lambda v: [leaf.block_until_ready() for leaf in jax.tree.leaves(v) if hasattr(leaf, "block_until_ready")]
                wait(jit_model(*inputs))
                samp = _jax_block(jit_model, inputs, warmup, repeats)
                st   = _stats(samp)
                gap  = analyze_gap(cost, type("M", (), {"median_seconds": st["median"] / 1e3, "samples_seconds": tuple(s / 1e3 for s in samp)})(), hardware)
                records.append(BenchRecord(
                    "JAX", "jit", arch, batch,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  JAX jit {arch} B={batch}: {exc}")
    return records


def eval_tensorflow(hardware: HardwareSpec, batches: list[int], warmup: int, repeats: int) -> list[BenchRecord]:
    import tensorflow as tf
    adapter = TensorFlowAdapter()
    records: list[BenchRecord] = []
    for arch in ARCHITECTURES:
        for batch in batches:
            model, inputs = TF_BUILDERS[arch](batch)
            try:
                prof = profile_model(model, inputs, adapter)
            except Exception:
                continue
            cost = prof.cost
            mem  = prof.memory
            fn   = lambda *a: model(*a, training=False)
            # Baseline (eager)
            try:
                samp = _tf_block(fn, inputs, warmup, repeats)
                st   = _stats(samp)
                gap  = analyze_gap(cost, type("M", (), {"median_seconds": st["median"] / 1e3, "samples_seconds": tuple(s / 1e3 for s in samp)})(), hardware)
                records.append(BenchRecord(
                    "TensorFlow", "baseline", arch, batch,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  TF baseline {arch} B={batch}: {exc}")
            # Optimised: tf.function (XLA)
            try:
                tf_fn = tf.function(fn, jit_compile=False)
                for _ in range(3):
                    tf_fn(*inputs)
                samp = _tf_block(tf_fn, inputs, warmup, repeats)
                st   = _stats(samp)
                gap  = analyze_gap(cost, type("M", (), {"median_seconds": st["median"] / 1e3, "samples_seconds": tuple(s / 1e3 for s in samp)})(), hardware)
                records.append(BenchRecord(
                    "TensorFlow", "tf.function", arch, batch,
                    cost.flops, mem.parameter_bytes, cost.total_bytes,
                    cost.arithmetic_intensity,
                    st["median"], st["mean"], st["stddev"], st["cv"], st["p95"],
                    gap.efficiency, gap.achieved_flops / 1e9, gap.achieved_bandwidth / 1e9,
                    gap.bottleneck,
                ))
            except Exception as exc:
                print(f"  TF tf.function {arch} B={batch}: {exc}")
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

RUNNERS = {
    "torch":       ("PyTorch",     eval_torch),
    "jax":         ("JAX",         eval_jax),
    "tensorflow":  ("TensorFlow",  eval_tensorflow),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Fewer batch sizes and repeats for fast iteration")
    parser.add_argument("--warmup",  type=int, default=15)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--peak-flops",       type=float, default=None)
    parser.add_argument("--memory-bandwidth", type=float, default=None)
    args = parser.parse_args()

    warmup  = 5  if args.quick else args.warmup
    repeats = 15 if args.quick else args.repeats
    batches = QUICK_BATCH_SIZES if args.quick else BATCH_SIZES

    print("Detecting hardware…", flush=True)
    hardware, detection = detect_hardware()
    if args.peak_flops:
        hardware = HardwareSpec(hardware.name, args.peak_flops, hardware.memory_bandwidth)
    if args.memory_bandwidth:
        hardware = HardwareSpec(hardware.name, hardware.peak_flops, args.memory_bandwidth)

    hw_meta = {
        "name": hardware.name,
        "peak_flops": hardware.peak_flops,
        "memory_bandwidth": hardware.memory_bandwidth,
        "ridge_point": hardware.ridge_point,
        "source": detection.source,
        "measured_bw_gb_s": detection.measured_bandwidth_gb_s,
    }
    print(f"  {hardware.name}  {hardware.peak_flops/1e12:.2f} TFLOP/s  {hardware.memory_bandwidth/1e9:.0f} GB/s")
    print(f"  Batches={batches}  warmup={warmup}  repeats={repeats}\n")

    all_records: list[BenchRecord] = []
    for pkg, (label, runner) in RUNNERS.items():
        if importlib.util.find_spec(pkg) is None:
            print(f"[skip] {label} not installed")
            continue
        print(f"── {label} ─────────────────────────────────")
        recs = runner(hardware, batches, warmup, repeats)
        all_records.extend(recs)
        for r in recs:
            print(f"  {r.variant:<12} {r.architecture:<14} B={r.batch:<4}  "
                  f"{r.latency_median_ms:7.2f} ms  {r.roofline_efficiency:.1%}  {r.bottleneck}")
        print()

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "benchmark_data.json"
    payload = {"hardware": hw_meta, "records": [asdict(r) for r in all_records]}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {len(all_records)} records → {out_path}")


if __name__ == "__main__":
    main()
