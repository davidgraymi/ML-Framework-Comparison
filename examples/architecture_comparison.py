"""End-to-end architecture comparison across PyTorch, JAX, and TensorFlow.

Benchmarks five canonical neural-network architecture families on a single
shared workload per family so framework overhead and roofline efficiency are
directly comparable:

  • FF DNN     – multi-layer feedforward classifier
  • CNN        – 3-block ConvNet with pooling
  • RNN        – 2-layer vanilla RNN encoder
  • LSTM       – 2-layer LSTM encoder
  • Transformer – 2-layer encoder (multi-head attention + FFN)

Run after installing one or more framework extras:

    pip install -e '.[torch,jax,tensorflow]'
    python examples/architecture_comparison.py

Hardware is auto-detected (Apple Silicon, NVIDIA, CPU fallback). Override:

    python examples/architecture_comparison.py --peak-flops 312e12 --memory-bandwidth 1.6e12

Usage flags:
    --warmup N        warm-up iterations per shape (default: 5)
    --repeats N       timed iterations per shape (default: 20)
    --batch N         batch size (default: 16)
    --seq-len N       sequence length for RNN/LSTM/Transformer (default: 32)
    --bw-bench-mb N   bandwidth benchmark working-set MiB (default: 256)
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

from neural_cost import HardwareSpec, analyze_gap, estimate_model, profile_model
from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchAdapter
from neural_cost.adapters.base import FrameworkAdapter
from neural_cost.hardware_detect import detect_hardware


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

EMBED_DIM = 128          # hidden/channel/embedding dimension
NUM_HEADS = 4            # attention heads (Transformer)
NUM_CLASSES = 10         # output classes


def installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    framework: str
    architecture: str
    flops: int
    bytes_moved: int
    param_bytes: int
    arith_intensity: float
    median_ms: float
    stddev_ms: float
    efficiency: float
    achieved_gflops: float
    achieved_gbw: float
    bottleneck: str
    fused_efficiency: float | None = None
    fused_lower_bound_ms: float | None = None
    cache_resident: bool = False
    cache_name: str | None = None
    top_layer_bottleneck: str | None = None
    top_layer_share_pct: float | None = None


def evaluate(
    framework: str,
    architecture: str,
    model: Callable[..., Any],
    inputs: tuple[Any, ...],
    adapter: FrameworkAdapter,
    hardware: HardwareSpec,
    warmup: int,
    repeats: int,
) -> Result:
    profile = profile_model(model, inputs, adapter)
    measurement = adapter.benchmark(model, *inputs, warmup=warmup, repeats=repeats)
    gap = analyze_gap(profile.cost, measurement, hardware, operations=profile.operations)

    samples_ms = [s * 1e3 for s in measurement.samples_seconds]
    sd = stdev(samples_ms) if len(samples_ms) > 1 else 0.0

    fused_lb_ms = (
        round(gap.fused_lower_bound_seconds * 1e3, 3)
        if gap.fused_lower_bound_seconds is not None
        else None
    )
    cache_resident = gap.resident_cache_level is not None
    cache_name = gap.resident_cache_level

    top_layer_bneck = None
    top_layer_share = None
    if gap.layer_analyses:
        top_l = max(gap.layer_analyses, key=lambda l: l.time_share_ratio)
        top_layer_bneck = f"{top_l.name} ({top_l.kind}, {top_l.bottleneck}-bound)"
        top_layer_share = round(top_l.time_share_ratio * 100, 1)

    return Result(
        framework=framework,
        architecture=architecture,
        flops=profile.cost.flops,
        bytes_moved=profile.cost.total_bytes,
        param_bytes=profile.memory.parameter_bytes,
        arith_intensity=profile.cost.arithmetic_intensity,
        median_ms=measurement.median_seconds * 1e3,
        stddev_ms=sd,
        efficiency=gap.efficiency,
        achieved_gflops=gap.achieved_flops / 1e9,
        achieved_gbw=gap.achieved_bandwidth / 1e9,
        bottleneck=gap.bottleneck,
        fused_efficiency=gap.fused_efficiency,
        fused_lower_bound_ms=fused_lb_ms,
        cache_resident=cache_resident,
        cache_name=cache_name,
        top_layer_bottleneck=top_layer_bneck,
        top_layer_share_pct=top_layer_share,
    )


# ---------------------------------------------------------------------------
# PyTorch model builders
# ---------------------------------------------------------------------------

def torch_ff_dnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(784, EMBED_DIM),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(EMBED_DIM),
        torch.nn.Linear(EMBED_DIM, EMBED_DIM),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(EMBED_DIM),
        torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
    ).eval()
    return model, (torch.randn(batch, 784),)


def torch_cnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, EMBED_DIM // 2, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM // 2),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(EMBED_DIM // 2, EMBED_DIM, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(EMBED_DIM),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(EMBED_DIM, NUM_CLASSES),
    ).eval()
    return model, (torch.randn(batch, 3, 32, 32),)


def torch_rnn(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch

    class RNNModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = torch.nn.RNN(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x: Any) -> Any:
            out, _ = self.rnn(x)
            return self.fc(out[:, -1])

    return RNNModel().eval(), (torch.randn(batch, seq, EMBED_DIM),)


def torch_lstm(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch

    class LSTMModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(EMBED_DIM, EMBED_DIM, num_layers=2, batch_first=True)
            self.fc = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x: Any) -> Any:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    return LSTMModel().eval(), (torch.randn(batch, seq, EMBED_DIM),)


def torch_transformer(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import torch

    class TransformerEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn1 = torch.nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=False)
            self.norm1a = torch.nn.LayerNorm(EMBED_DIM)
            self.ff1 = torch.nn.Linear(EMBED_DIM, EMBED_DIM * 4)
            self.ff1b = torch.nn.Linear(EMBED_DIM * 4, EMBED_DIM)
            self.norm1b = torch.nn.LayerNorm(EMBED_DIM)
            self.attn2 = torch.nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=False)
            self.norm2a = torch.nn.LayerNorm(EMBED_DIM)
            self.ff2 = torch.nn.Linear(EMBED_DIM, EMBED_DIM * 4)
            self.ff2b = torch.nn.Linear(EMBED_DIM * 4, EMBED_DIM)
            self.norm2b = torch.nn.LayerNorm(EMBED_DIM)
            self.head = torch.nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x: Any) -> Any:  # x: (B, T, D)
            x_t = x.permute(1, 0, 2)  # (T, B, D) for MHA
            a1, _ = self.attn1(x_t, x_t, x_t)
            x_t = self.norm1a(x_t + a1)
            f1 = torch.relu(self.ff1(x_t))
            x_t = self.norm1b(x_t + self.ff1b(f1))
            a2, _ = self.attn2(x_t, x_t, x_t)
            x_t = self.norm2a(x_t + a2)
            f2 = torch.relu(self.ff2(x_t))
            x_t = self.norm2b(x_t + self.ff2b(f2))
            return self.head(x_t.mean(0))

    return TransformerEncoder().eval(), (torch.randn(batch, seq, EMBED_DIM),)


# ---------------------------------------------------------------------------
# JAX model builders
# ---------------------------------------------------------------------------

def jax_ff_dnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import jax
    import jax.numpy as jnp

    w1 = jnp.ones((784, EMBED_DIM))
    w2 = jnp.ones((EMBED_DIM, EMBED_DIM))
    w3 = jnp.ones((EMBED_DIM, NUM_CLASSES))

    def model(x: Any, _w1: Any = w1, _w2: Any = w2, _w3: Any = w3) -> Any:
        x = jnp.tanh(x @ _w1)
        x = jnp.tanh(x @ _w2)
        return x @ _w3

    return model, (jnp.ones((batch, 784)), w1, w2, w3)


def jax_cnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    """Approximate CNN via a sequence of matmuls (JAX jaxpr traces raw primitives)."""
    import jax.numpy as jnp

    # Represent the two conv feature maps + final classifier as matmuls over flattened space.
    # The actual convolution kernel has shape [kH, kW, Cin, Cout] in JAX (HWIO).
    # We use conv_general_dilated via jax.lax for the jaxpr to capture conv2d.
    import jax.lax as lax

    k1 = jnp.ones((EMBED_DIM // 2, 3, 3, 3))   # (Cout, Cin, kH, kW)  — strides NCHW
    k2 = jnp.ones((EMBED_DIM, EMBED_DIM // 2, 3, 3))
    w_fc = jnp.ones((EMBED_DIM, NUM_CLASSES))

    def model(x: Any, _k1: Any = k1, _k2: Any = k2, _wfc: Any = w_fc) -> Any:
        # x: (B, C, H, W)  NCHW
        y = lax.conv_general_dilated(
            x, _k1, window_strides=(1, 1), padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        y = jnp.tanh(y)
        y = lax.conv_general_dilated(
            y, _k2, window_strides=(2, 2), padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        y = jnp.tanh(y)
        y = y.mean(axis=(2, 3))  # global avg pool -> (B, EMBED_DIM)
        return y @ _wfc

    return model, (jnp.ones((batch, 3, 32, 32)), k1, k2, w_fc)


def jax_rnn(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    """Single-step RNN unrolled via scan (approximated with per-step matmuls)."""
    import jax
    import jax.numpy as jnp

    w_ih = jnp.ones((EMBED_DIM, EMBED_DIM))
    w_hh = jnp.ones((EMBED_DIM, EMBED_DIM))
    w_fc = jnp.ones((EMBED_DIM, NUM_CLASSES))

    def model(x: Any, _wih: Any = w_ih, _whh: Any = w_hh, _wfc: Any = w_fc) -> Any:
        # x: (B, T, D)
        h = jnp.zeros((x.shape[0], EMBED_DIM))
        for t in range(x.shape[1]):
            h = jnp.tanh(x[:, t] @ _wih + h @ _whh)
        return h @ _wfc

    dummy = jnp.ones((batch, seq, EMBED_DIM))
    return model, (dummy, w_ih, w_hh, w_fc)


def jax_lstm(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    """LSTM approximated as 4-gate matmuls per timestep."""
    import jax.numpy as jnp

    gates = 4
    w_ih = jnp.ones((EMBED_DIM, gates * EMBED_DIM))
    w_hh = jnp.ones((EMBED_DIM, gates * EMBED_DIM))
    w_fc = jnp.ones((EMBED_DIM, NUM_CLASSES))

    def model(x: Any, _wih: Any = w_ih, _whh: Any = w_hh, _wfc: Any = w_fc) -> Any:
        h = jnp.zeros((x.shape[0], EMBED_DIM))
        c = jnp.zeros((x.shape[0], EMBED_DIM))
        for t in range(x.shape[1]):
            gates_val = x[:, t] @ _wih + h @ _whh  # (B, 4*D)
            i, f, g, o = jnp.split(gates_val, 4, axis=-1)
            i = jax_sigmoid(i)
            f = jax_sigmoid(f)
            g = jnp.tanh(g)
            o = jax_sigmoid(o)
            c = f * c + i * g
            h = o * jnp.tanh(c)
        return h @ _wfc

    dummy = jnp.ones((batch, seq, EMBED_DIM))
    return model, (dummy, w_ih, w_hh, w_fc)


def jax_sigmoid(x: Any) -> Any:
    import jax.numpy as jnp
    return 1.0 / (1.0 + jnp.exp(-x))


def jax_transformer(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    """Transformer approximated via Q/K/V projections + softmax per head."""
    import jax.numpy as jnp

    head_dim = EMBED_DIM // NUM_HEADS
    wq = jnp.ones((EMBED_DIM, EMBED_DIM))
    wk = jnp.ones((EMBED_DIM, EMBED_DIM))
    wv = jnp.ones((EMBED_DIM, EMBED_DIM))
    wo = jnp.ones((EMBED_DIM, EMBED_DIM))
    w_ff1 = jnp.ones((EMBED_DIM, EMBED_DIM * 4))
    w_ff2 = jnp.ones((EMBED_DIM * 4, EMBED_DIM))
    w_fc = jnp.ones((EMBED_DIM, NUM_CLASSES))

    def attention(x: Any, wq_: Any, wk_: Any, wv_: Any, wo_: Any) -> Any:
        # x: (B, T, D)
        q = x @ wq_
        k = x @ wk_
        v = x @ wv_
        scale = head_dim ** -0.5
        # Simplified single-head attention for jaxpr tracing
        scores = (q * scale) @ k.transpose(0, 2, 1)
        weights = jnp.exp(scores) / jnp.exp(scores).sum(axis=-1, keepdims=True)
        return (weights @ v) @ wo_

    def model(
        x: Any,
        _wq: Any = wq, _wk: Any = wk, _wv: Any = wv, _wo: Any = wo,
        _wff1: Any = w_ff1, _wff2: Any = w_ff2, _wfc: Any = w_fc,
    ) -> Any:
        x = x + attention(x, _wq, _wk, _wv, _wo)
        x = x + jnp.tanh(x @ _wff1) @ _wff2
        return x.mean(axis=1) @ _wfc

    dummy = jnp.ones((batch, seq, EMBED_DIM))
    return model, (dummy, wq, wk, wv, wo, w_ff1, w_ff2, w_fc)


# ---------------------------------------------------------------------------
# TensorFlow model builders
# ---------------------------------------------------------------------------

def tf_ff_dnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(EMBED_DIM, activation="relu"),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(NUM_CLASSES),
    ])
    inp = tf.ones((batch, 784))
    model(inp)  # build
    return model, (inp,)


def tf_cnn(batch: int, _seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(EMBED_DIM // 2, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(EMBED_DIM, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES),
    ])
    inp = tf.ones((batch, 32, 32, 3))
    model(inp)
    return model, (inp,)


def tf_rnn(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf
    # Use return_sequences=False for simplicity
    inp_layer = tf.keras.layers.Input(shape=(seq, EMBED_DIM))
    x = tf.keras.layers.GRU(EMBED_DIM, return_sequences=True)(inp_layer)
    x = tf.keras.layers.GRU(EMBED_DIM)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)
    model = tf.keras.Model(inp_layer, out)
    inp = tf.ones((batch, seq, EMBED_DIM))
    model(inp)
    return model, (inp,)


def tf_lstm(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf
    inp_layer = tf.keras.layers.Input(shape=(seq, EMBED_DIM))
    x = tf.keras.layers.LSTM(EMBED_DIM, return_sequences=True)(inp_layer)
    x = tf.keras.layers.LSTM(EMBED_DIM)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES)(x)
    model = tf.keras.Model(inp_layer, out)
    inp = tf.ones((batch, seq, EMBED_DIM))
    model(inp)
    return model, (inp,)


def tf_transformer(batch: int, seq: int) -> tuple[Any, tuple[Any, ...]]:
    import tensorflow as tf

    inp_layer = tf.keras.layers.Input(shape=(seq, EMBED_DIM))
    # Layer 1
    x = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)(inp_layer, inp_layer)
    x = tf.keras.layers.LayerNormalization()(inp_layer + x)
    ff = tf.keras.layers.Dense(EMBED_DIM * 4, activation="relu")(x)
    ff = tf.keras.layers.Dense(EMBED_DIM)(ff)
    x = tf.keras.layers.LayerNormalization()(x + ff)
    # Layer 2
    x2 = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)(x, x)
    x2 = tf.keras.layers.LayerNormalization()(x + x2)
    ff2 = tf.keras.layers.Dense(EMBED_DIM * 4, activation="relu")(x2)
    ff2 = tf.keras.layers.Dense(EMBED_DIM)(ff2)
    x2 = tf.keras.layers.LayerNormalization()(x2 + ff2)
    # Head
    pooled = tf.keras.layers.GlobalAveragePooling1D()(x2)
    out = tf.keras.layers.Dense(NUM_CLASSES)(pooled)

    model = tf.keras.Model(inp_layer, out)
    inp = tf.ones((batch, seq, EMBED_DIM))
    model(inp)
    return model, (inp,)


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ARCHITECTURES = ["FF DNN", "CNN", "RNN", "LSTM", "Transformer"]

BUILDERS: dict[str, dict[str, Any]] = {
    "PyTorch":     {"FF DNN": torch_ff_dnn, "CNN": torch_cnn, "RNN": torch_rnn, "LSTM": torch_lstm, "Transformer": torch_transformer},
    "JAX":         {"FF DNN": jax_ff_dnn,   "CNN": jax_cnn,   "RNN": jax_rnn,   "LSTM": jax_lstm,   "Transformer": jax_transformer},
    "TensorFlow":  {"FF DNN": tf_ff_dnn,    "CNN": tf_cnn,    "RNN": tf_rnn,    "LSTM": tf_lstm,    "Transformer": tf_transformer},
}

ADAPTERS: dict[str, Any] = {
    "PyTorch": TorchAdapter,
    "JAX": JaxAdapter,
    "TensorFlow": TensorFlowAdapter,
}

PACKAGES: dict[str, str] = {
    "PyTorch": "torch",
    "JAX": "jax",
    "TensorFlow": "tensorflow",
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SEP = "─" * 135


def _bar(fraction: float, width: int = 18) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_flops(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.3f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n:,}"


def print_hardware(hardware: HardwareSpec, detection: Any) -> None:
    print()
    print("┌─ Hardware ──────────────────────────────────────────────────────────────────")
    print(f"│  Device          : {hardware.name}")
    print(f"│  Peak FP32       : {hardware.peak_flops / 1e12:.2f} TFLOP/s")
    print(f"│  Peak bandwidth  : {hardware.memory_bandwidth / 1e9:.1f} GB/s  "
          f"(STREAM triad: {detection.measured_bandwidth_gb_s:.1f} GB/s)")
    print(f"│  Ridge point     : {hardware.ridge_point:.1f} FLOP/byte")
    print(f"│  Detection source: {detection.source}")
    print("└────────────────────────────────────────────────────────────────────────────")
    print()


def print_architecture_section(arch: str, results: list[Result], hardware: HardwareSpec) -> None:
    fw_results = [r for r in results if r.architecture == arch]
    if not fw_results:
        return

    print(f"\n{'━' * 135}")
    print(f"  {arch}")
    print(f"{'━' * 135}")

    col = dict(fw=12, flops=10, params=10, mbytes=9, ai=7, ms=9, sd=7, eff=8, bar=20, gflops=9, gbw=8, bot=8)
    hdr = (
        f"{'Framework':<{col['fw']}} "
        f"{'FLOPs':>{col['flops']}} "
        f"{'Params':>{col['params']}} "
        f"{'I/O MB':>{col['mbytes']}} "
        f"{'AI':>{col['ai']}} "
        f"{'ms(med)':>{col['ms']}} "
        f"{'±ms':>{col['sd']}} "
        f"{'effic.':>{col['eff']}} "
        f"{'roofline':^{col['bar']}} "
        f"{'GFLOP/s':>{col['gflops']}} "
        f"{'GB/s':>{col['gbw']}} "
        f"{'bound':<{col['bot']}}"
    )
    print(hdr)
    print("─" * 135)

    for r in fw_results:
        eff_clamped = min(r.efficiency, 1.0)
        bar = _bar(eff_clamped)
        eff_str = f"{r.efficiency:.1%}" if r.efficiency < 1.0 else f">{100:.0f}%*"
        print(
            f"{r.framework:<{col['fw']}} "
            f"{_fmt_flops(r.flops):>{col['flops']}} "
            f"{_fmt_flops(r.param_bytes):>{col['params']}} "
            f"{r.bytes_moved / 1e6:>{col['mbytes']}.1f} "
            f"{r.arith_intensity:>{col['ai']}.2f} "
            f"{r.median_ms:>{col['ms']}.3f} "
            f"{r.stddev_ms:>{col['sd']}.3f} "
            f"{eff_str:>{col['eff']}} "
            f"[{bar}] "
            f"{r.achieved_gflops:>{col['gflops']}.2f} "
            f"{r.achieved_gbw:>{col['gbw']}.2f} "
            f"{r.bottleneck:<{col['bot']}}"
        )

    print("─" * 135)
    print("  Diagnostics (top layer bottleneck & fusion):")
    for r in fw_results:
        fused_info = f", Fused Eff: {r.fused_efficiency:.1%}" if r.fused_efficiency is not None else ""
        cache_info = f", Resident in {r.cache_name}" if r.cache_resident and r.cache_name else ""
        top_info = f"Top Bottleneck: {r.top_layer_bottleneck} ({r.top_layer_share_pct}%)" if r.top_layer_bottleneck else "Top Bottleneck: N/A"
        print(f"    • {r.framework}: {top_info}{fused_info}{cache_info}")
    print("─" * 135)
    print(f"  AI = arithmetic intensity (FLOP/byte).  "
          f"Ridge point = {hardware.ridge_point:.1f} FLOP/byte  "
          "(above → compute-bound, below → memory-bound)")


def print_cross_framework_summary(results: list[Result]) -> None:
    print(f"\n{'━' * 135}")
    print("  Per-Architecture × Per-Framework Summary")
    print(f"{'━' * 135}")

    arch_col = 14
    fw_col = 28  # "framework: eff% (Xms)"

    header = f"  {'Architecture':<{arch_col}}"
    all_fws: list[str] = []
    seen: set[str] = set()
    for r in results:
        if r.framework not in seen:
            all_fws.append(r.framework)
            seen.add(r.framework)
    for fw in all_fws:
        header += f"  {fw:<{fw_col}}"
    print(header)
    print("  " + "─" * (arch_col + len(all_fws) * (fw_col + 2)))

    archs = list(dict.fromkeys(r.architecture for r in results))
    for arch in archs:
        row = f"  {arch:<{arch_col}}"
        for fw in all_fws:
            match = next((r for r in results if r.architecture == arch and r.framework == fw), None)
            if match:
                cell = f"{match.efficiency:.1%} eff  {match.median_ms:.2f}ms  {_fmt_flops(match.flops)} FLOPs"
            else:
                cell = "(skipped)"
            row += f"  {cell:<{fw_col}}"
        print(row)

    print(f"\n  *Efficiency >100% means hardware spec is conservative; pass --peak-flops to calibrate.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--peak-flops", type=float, default=None,
                        help="Override peak FP32 FLOP/s (e.g. 3.6e12)")
    parser.add_argument("--memory-bandwidth", type=float, default=None,
                        help="Override memory bandwidth bytes/s (e.g. 100e9)")
    parser.add_argument("--bw-bench-mb", type=int, default=256,
                        help="Working-set MiB for bandwidth benchmark (default 256)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warm-up iterations per workload (default 5)")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Timed iterations per workload (default 20)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default 16)")
    parser.add_argument("--seq-len", type=int, default=32,
                        help="Sequence length for RNN/LSTM/Transformer (default 32)")
    args = parser.parse_args()

    print("Detecting hardware…", flush=True)
    hardware, detection = detect_hardware(bandwidth_benchmark_mb=args.bw_bench_mb)
    if args.peak_flops is not None:
        hardware = HardwareSpec(
            hardware.name, args.peak_flops, hardware.memory_bandwidth, caches=hardware.caches
        )
    if args.memory_bandwidth is not None:
        hardware = HardwareSpec(
            hardware.name, hardware.peak_flops, args.memory_bandwidth, caches=hardware.caches
        )
    print_hardware(hardware, detection)

    all_results: list[Result] = []

    for fw_name, pkg in PACKAGES.items():
        if not installed(pkg):
            print(f"  [skipping {fw_name} – not installed]")
            continue
        if fw_name == "PyTorch":
            from neural_cost.adapters import TorchFxAdapter
            try:
                adapter: FrameworkAdapter = TorchFxAdapter()
            except Exception:  # noqa: BLE001
                adapter = ADAPTERS[fw_name]()
        else:
            adapter = ADAPTERS[fw_name]()
        builders = BUILDERS[fw_name]
        print(f"Benchmarking {fw_name}…", flush=True)
        for arch in ARCHITECTURES:
            builder = builders[arch]
            try:
                model, inputs = builder(args.batch, args.seq_len)
                result = evaluate(
                    fw_name, arch, model, inputs, adapter, hardware,
                    warmup=args.warmup, repeats=args.repeats,
                )
                all_results.append(result)
                status = f"{result.median_ms:.2f} ms  {result.efficiency:.1%} eff"
                print(f"    {arch:<14} {status}")
            except Exception as exc:  # noqa: BLE001
                print(f"    {arch:<14} ERROR: {exc}")

    if not all_results:
        raise SystemExit("Install at least one framework: pip install -e '.[torch,jax,tensorflow]'")

    print()
    for arch in ARCHITECTURES:
        arch_results = [r for r in all_results if r.architecture == arch]
        if arch_results:
            print_architecture_section(arch, all_results, hardware)

    print_cross_framework_summary(all_results)
    print()


if __name__ == "__main__":
    main()
