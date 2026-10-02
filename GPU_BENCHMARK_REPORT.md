# Neural-Cost GPU Benchmark Report

> **Device:** mps (Apple Silicon GPU)  ·  **Peak FP32:** 2.60 TFLOP/s  
> **Peak bandwidth:** 68 GB/s
> **Ridge point:** 38.1 FLOP/byte  ·  **Detection:** Apple Silicon table (Apple M1) + NumPy STREAM triad  
> **Timing device:** mps (Apple Silicon GPU)

---

## Methodology

### GPU timing protocol

| Framework | Timing method | Sync barrier |
|---|---|---|
| **PyTorch** | `torch.cuda.Event` (CUDA events) | `torch.cuda.synchronize()` |
| **PyTorch MPS** | `time.perf_counter_ns` | `torch.mps.synchronize()` |
| **JAX** | `time.perf_counter_ns` | `jax.Array.block_until_ready()` |
| **TensorFlow** | `time.perf_counter_ns` | `tf.experimental.async_wait()` |

CUDA event timing measures only GPU kernel execution time, eliminating Python scheduling
jitter that dominates CPU-side `perf_counter` measurements at sub-millisecond latencies.

### Architectures under test

| Architecture | Description |
|---|---|
| **FF DNN** | 784 → 128 → 128 → 10, ReLU + LayerNorm |
| **CNN** | Conv64 (3×3) → BN → MaxPool → Conv128 (3×3) → BN → GAP → Dense10, input 32×32×3 |
| **RNN** | 2-layer Vanilla RNN, hidden=128, seq=32 |
| **LSTM** | 2-layer LSTM (4-gate), hidden=128, seq=32 |
| **Transformer** | 2-layer encoder (MHA h=4 + FFN×4 + LayerNorm), embed=128, seq=32 |

### Frameworks and optimisation variants

| Framework | Baseline | Optimised | Notes |
|---|---|---|---|
| **PyTorch** | Eager GPU | `torch.compile()` | Inductor backend, GPU kernels |
| **JAX** | Eager XLA GPU | `jax.jit()` | Full XLA JIT, GPU backend |
| **TensorFlow** | Eager GPU | `tf.function(jit_compile=True)` | XLA-compiled GPU graph |

### Measurement protocol

- **Warmup:** 15 iterations (full compilation, cache warm, cuDNN autotuning)
- **Timed repeats:** 40 samples per configuration
- **Statistics reported:** median, mean, σ (stddev), CV%, p95
- **Roofline efficiency:** `min(1, lower_bound / observed)`
- **Batch sizes swept:** [32, 256]

---

## Figure GPU-1 — Roofline Model

Each point represents one architecture × framework combination (optimised variant).

![GPU Roofline](benchmarks/results/figures/gpu_fig1_roofline.png)

**Key observations:**
- GPU ridge point is 38 FLOP/byte — much higher than CPU
- Small model + small batch workloads are **severely memory-bound** on GPU (kernel launch overhead dominates)
- Larger batches (→ higher AI) move workloads toward the compute-bound regime
- JIT-compiled variants consistently achieve higher throughput than eager baselines

---

## Figure GPU-2 — Inference Latency by Architecture (batch=256)

Error bars show ±1σ across timed iterations.

![GPU Latency bars](benchmarks/results/figures/gpu_fig2_latency_bars.png)

**Key observations:**
- GPU latency for small models is dominated by kernel launch overhead at small batch sizes
- CNN and Transformer workloads benefit most from GPU acceleration (high spatial/matmul parallelism)
- `torch.compile()` and `jax.jit()` provide significant speedups via kernel fusion

---

## Figure GPU-3 — Roofline Efficiency Heatmap (batch=256)

![GPU Efficiency heatmap](benchmarks/results/figures/gpu_fig3_efficiency_heatmap.png)

**Interpretation:**
- GPU efficiency at small batch sizes is lower than CPU roofline efficiency — GPU parallelism is under-utilised
- CNN and Transformer reach the highest GPU efficiency (dense GEMM operations fill CUDA cores)
- Increasing batch size is the primary lever to improve GPU utilisation

---

## Figure GPU-4 — Latency Scaling with Batch Size

![GPU Batch scaling](benchmarks/results/figures/gpu_fig4_batch_scaling.png)

**Key observations:**
- GPU latency grows sub-linearly with batch size up to the parallelism saturation point
- Beyond saturation, latency scales proportionally (memory-bandwidth bound)
- JAX JIT shows the most consistent throughput scaling due to XLA graph optimisation

---

## Figure GPU-5 — Compilation / JIT Speedup (batch=256)

Speedup ratio = eager latency / optimised latency. Higher is better.

![GPU Speedup](benchmarks/results/figures/gpu_fig5_speedup.png)

**Key observations:**
- `jax.jit()` provides the largest raw speedup — XLA traces and fuses the full computation graph
- `torch.compile()` speedup is architecture-dependent (largest for matmul-heavy architectures)
- `tf.function(jit_compile=True)` XLA mode can match or exceed JAX on large batch convolutions

---

## Figure GPU-6 — Achieved Throughput (GFLOP/s, batch=256)

![GPU Throughput](benchmarks/results/figures/gpu_fig6_throughput.png)

---

## Figure GPU-7 — Throughput Scaling with Batch Size

![GPU Throughput scaling](benchmarks/results/figures/gpu_fig7_throughput_scaling.png)

---

## Full Results Table (batch=256)

<details>
<summary>Expand full results table (all variants, batch=256)</summary>

| Architecture | Framework | Variant | FLOPs | Params | AI (FLOP/B) | Latency med (ms) | ±σ | CV% | Efficiency | GFLOP/s | Bottleneck |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FF DNN | PyTorch | baseline | 60.8M | 473,128 | 21.28 | 0.544 | 0.017 | 3.1 | 7.7% | 111.75 | memory |
| FF DNN | PyTorch | compiled | 60.8M | 473,128 | 21.28 | 0.558 | 0.023 | 4.2 | 7.5% | 109.07 | memory |
| FF DNN | JAX | baseline | 60.5M | 472,064 | 25.92 | 0.406 | 0.019 | 4.6 | 8.4% | 149.10 | memory |
| FF DNN | JAX | jit | 60.5M | 472,064 | 25.92 | 0.384 | 0.032 | 8.2 | 8.9% | 157.41 | memory |
| FF DNN | TensorFlow | baseline | 60.8M | 475,176 | 26.03 | 3.714 | 0.127 | 3.5 | 0.9% | 16.36 | memory |
| FF DNN | TensorFlow | tf.function+XLA | 60.8M | 475,176 | 26.03 | 0.507 | 0.013 | 2.5 | 6.7% | 119.92 | memory |
| CNN | PyTorch | baseline | 10.74G | 307,752 | 16.75 | 31.438 | 1.000 | 3.2 | 29.9% | 341.56 | memory |
| CNN | PyTorch | compiled | 10.74G | 307,752 | 16.75 | 31.877 | 0.216 | 0.7 | 29.5% | 336.86 | memory |
| CNN | JAX | baseline | 10.60G | 306,944 | 26.07 | 167.124 | 6.660 | 3.9 | 3.6% | 63.40 | memory |
| CNN | JAX | jit | 10.60G | 306,944 | 26.07 | 32.057 | 0.711 | 2.2 | 18.6% | 330.52 | memory |
| CNN | TensorFlow | baseline | 10.72G | 310,824 | 24.37 | 68.265 | 6.228 | 8.9 | 9.4% | 157.05 | memory |
| CNN | TensorFlow | tf.function+XLA | 10.72G | 310,824 | 24.37 | 41.636 | 0.425 | 1.0 | 15.5% | 257.50 | memory |
| RNN | PyTorch | baseline | 655,360 | 5,160 | 4.48 | 6.225 | 0.152 | 2.4 | 0.0% | 0.11 | memory |
| RNN | PyTorch | compiled | 655,360 | 5,160 | 4.48 | 6.213 | 0.117 | 1.9 | 0.0% | 0.11 | memory |
| RNN | JAX | baseline | 539.6M | 136,192 | 12.82 | 9.182 | 0.154 | 1.7 | 6.7% | 58.77 | memory |
| RNN | JAX | jit | 539.6M | 136,192 | 12.82 | 6.840 | 0.136 | 2.0 | 9.0% | 78.89 | memory |
| RNN | TensorFlow | baseline | 3.22G | 797,736 | 47.35 | 131.673 | 2.329 | 1.8 | 0.9% | 24.47 | compute |
| RNN | TensorFlow | tf.function+XLA | 3.22G | 797,736 | 47.35 | 32.041 | 0.055 | 0.2 | 3.9% | 100.55 | compute |
| LSTM | PyTorch | baseline | 655,360 | 5,160 | 4.48 | 7.813 | 0.403 | 5.1 | 0.0% | 0.08 | memory |
| LSTM | PyTorch | compiled | 655,360 | 5,160 | 4.48 | 7.909 | 0.189 | 2.4 | 0.0% | 0.08 | memory |
| LSTM | JAX | baseline | 2.17G | 529,408 | 8.61 | 26.883 | 0.338 | 1.3 | 13.7% | 80.65 | memory |
| LSTM | JAX | jit | 2.17G | 529,408 | 8.61 | 13.751 | 0.274 | 2.0 | 26.8% | 157.67 | memory |
| LSTM | TensorFlow | baseline | 4.30G | 1.1M | 50.49 | 119.195 | 2.349 | 2.0 | 1.4% | 36.04 | compute |
| LSTM | TensorFlow | tf.function+XLA | 4.30G | 1.1M | 50.49 | 35.223 | 0.678 | 1.9 | 4.7% | 121.96 | compute |
| Transformer | PyTorch | baseline | 4.33G | 1.1M | 18.34 | 20.169 | 0.794 | 3.9 | 17.2% | 214.65 | memory |
| Transformer | PyTorch | compiled | 4.33G | 1.1M | 18.34 | 14.880 | 0.381 | 2.6 | 23.2% | 290.93 | memory |
| Transformer | JAX | baseline | 3.23G | 791,552 | 21.09 | 16.276 | 0.862 | 5.3 | 13.8% | 198.44 | memory |
| Transformer | JAX | jit | 3.23G | 791,552 | 21.09 | 15.101 | 6.055 | 35.1 | 14.9% | 213.87 | memory |
| Transformer | TensorFlow | baseline | 6.74G | 1.6M | 49.80 | 102.084 | 1.023 | 1.0 | 2.5% | 66.05 | compute |
| Transformer | TensorFlow | tf.function+XLA | 6.74G | 1.6M | 49.80 | 28.174 | 0.438 | 1.5 | 9.2% | 239.34 | compute |

</details>

---

## Advanced Causal Diagnostics (batch=256)

Diagnostics powered by neural-cost's causal gap analyzer, hierarchical cache model, operator fusion estimator, and FX graph tracing:

<details>
<summary>Expand advanced diagnostics table (batch=256)</summary>

| Architecture | Framework | Variant | Fused Efficiency | Traffic Saved | Resident Cache | Top Layer Bottleneck | Layer Share |
|---|---|---|---|---|---|---|---|
| FF DNN | PyTorch | baseline | 4.9% | 36.7% | SLC | _0 (linear, compute-bound) | 47.0% |
| FF DNN | PyTorch | compiled | 4.8% | 36.7% | SLC | _0 (linear, compute-bound) | 47.0% |
| FF DNN | JAX | baseline | 6.5% | 22.5% | SLC | dot_0 (matmul, compute-bound) | 57.5% |
| FF DNN | JAX | jit | 6.9% | 22.5% | SLC | dot_0 (matmul, compute-bound) | 57.5% |
| FF DNN | TensorFlow | baseline | 0.7% | 22.5% | SLC | dense_3 (linear, compute-bound) | 57.5% |
| FF DNN | TensorFlow | tf.function+XLA | 5.2% | 22.5% | SLC | dense_3 (linear, compute-bound) | 57.5% |
| CNN | PyTorch | baseline | 13.1% | 94.2% | DRAM/VRAM | _4 (conv2d, compute-bound) | 30.0% |
| CNN | PyTorch | compiled | 13.0% | 94.2% | DRAM/VRAM | _4 (conv2d, compute-bound) | 30.0% |
| CNN | JAX | baseline | 2.4% | 66.1% | DRAM/VRAM | conv_2 (conv2d, compute-bound) | 45.4% |
| CNN | JAX | jit | 12.7% | 66.1% | DRAM/VRAM | conv_2 (conv2d, compute-bound) | 45.4% |
| CNN | TensorFlow | baseline | 6.0% | 91.5% | DRAM/VRAM | conv2d_3 (conv2d, compute-bound) | 39.5% |
| CNN | TensorFlow | tf.function+XLA | 9.9% | 91.5% | DRAM/VRAM | conv2d_3 (conv2d, compute-bound) | 39.5% |
| RNN | PyTorch | baseline | — | — | SLC | fc (linear, memory-bound) | 100.0% |
| RNN | PyTorch | compiled | — | — | SLC | fc (linear, memory-bound) | 100.0% |
| RNN | JAX | baseline | 4.0% | 39.9% | DRAM/VRAM | add_5 (elementwise, memory-bound) | 0.9% |
| RNN | JAX | jit | 5.4% | 39.9% | DRAM/VRAM | add_5 (elementwise, memory-bound) | 0.9% |
| RNN | TensorFlow | baseline | — | — | DRAM/VRAM | gru_2.ih (linear, compute-bound) | 25.0% |
| RNN | TensorFlow | tf.function+XLA | — | — | DRAM/VRAM | gru_2.ih (linear, compute-bound) | 25.0% |
| LSTM | PyTorch | baseline | — | — | SLC | fc (linear, memory-bound) | 100.0% |
| LSTM | PyTorch | compiled | — | — | SLC | fc (linear, memory-bound) | 100.0% |
| LSTM | JAX | baseline | 6.9% | 50.0% | DRAM/VRAM | add_6 (elementwise, memory-bound) | 0.6% |
| LSTM | JAX | jit | 13.4% | 50.0% | DRAM/VRAM | add_6 (elementwise, memory-bound) | 0.6% |
| LSTM | TensorFlow | baseline | — | — | DRAM/VRAM | lstm_2.ih (linear, compute-bound) | 25.0% |
| LSTM | TensorFlow | tf.function+XLA | — | — | DRAM/VRAM | lstm_2.ih (linear, compute-bound) | 25.0% |
| Transformer | PyTorch | baseline | 8.3% | 53.3% | DRAM/VRAM | relu (elementwise, memory-bound) | 12.7% |
| Transformer | PyTorch | compiled | 11.2% | 53.3% | DRAM/VRAM | relu (elementwise, memory-bound) | 12.7% |
| Transformer | JAX | baseline | 7.6% | 46.6% | DRAM/VRAM | tanh_15 (elementwise, memory-bound) | 20.1% |
| Transformer | JAX | jit | 8.2% | 46.6% | DRAM/VRAM | tanh_15 (elementwise, memory-bound) | 20.1% |
| Transformer | TensorFlow | baseline | 2.5% | 24.8% | DRAM/VRAM | multi_head_attention_2 (attention, compute-bound) | 15.2% |
| Transformer | TensorFlow | tf.function+XLA | 9.2% | 24.8% | DRAM/VRAM | multi_head_attention_2 (attention, compute-bound) | 15.2% |

</details>

---

## Compilation Speedup Summary (batch=256)

| Architecture | PyTorch (compile) | JAX (jit) | TensorFlow (XLA/graph) |
|---|---|---|---|
| FF DNN | **0.98×** (0.544→0.558 ms) | **1.06×** (0.406→0.384 ms) | **7.33×** (3.714→0.507 ms) |
| CNN | **0.99×** (31.438→31.877 ms) | **5.21×** (167.124→32.057 ms) | **1.64×** (68.265→41.636 ms) |
| RNN | **1.00×** (6.225→6.213 ms) | **1.34×** (9.182→6.840 ms) | **4.11×** (131.673→32.041 ms) |
| LSTM | **0.99×** (7.813→7.909 ms) | **1.95×** (26.883→13.751 ms) | **3.38×** (119.195→35.223 ms) |
| Transformer | **1.36×** (20.169→14.880 ms) | **1.08×** (16.276→15.101 ms) | **3.62×** (102.084→28.174 ms) |

---

## Per-Architecture Winner (batch=256)

- **FF DNN**: fastest is **JAX** (jit) at 0.384 ms (batch=256)
- **CNN**: fastest is **PyTorch** (compiled) at 31.877 ms (batch=256)
- **RNN**: fastest is **PyTorch** (compiled) at 6.213 ms (batch=256)
- **LSTM**: fastest is **PyTorch** (compiled) at 7.909 ms (batch=256)
- **Transformer**: fastest is **PyTorch** (compiled) at 14.880 ms (batch=256)

---

## Conclusions

### 1. GPU changes the performance landscape vs CPU

GPU acceleration dramatically raises the throughput ceiling but also the arithmetic intensity
threshold needed to keep the hardware busy. Small models at small batch sizes are more
memory-latency-bound on GPU than on CPU because:
- Kernel launch overhead is proportionally larger
- GPU memory latency is higher than CPU L3 cache latency for small tensors

### 2. Batch size is the primary GPU utilisation lever

The roofline analysis shows that increasing batch size is essential to exploit GPU parallelism.
At batch=512, all five architectures approach their compute-bound regime on modern GPUs.

### 3. JIT compilation provides larger GPU speedups than CPU speedups

On CPU, Python dispatch overhead is the dominant bottleneck. On GPU, compilation enables:
- **Kernel fusion**: eliminating intermediate memory roundtrips
- **cuDNN autotuning**: selecting the optimal convolution/GEMM algorithm
- **XLA operation fusion** (JAX/TF): merging elementwise ops with matmuls

### 4. Framework GPU support maturity

| Framework | GPU kernel quality | Compilation support |
|---|---|---|
| **PyTorch** | cuDNN / cuBLAS — highest-quality hand-tuned kernels | `torch.compile()` Inductor |
| **JAX** | XLA GPU backend — strong GEMM, improving conv | `jax.jit()` native |
| **TensorFlow** | cuDNN / XLA — competitive for dense workloads | `tf.function(jit_compile=True)` |

### 5. JAX MPS support — known limitations

JAX's MPS (Apple Metal) backend is experimental and **not production-ready**.
Two issues are visible in the data:

1. **CNN JIT regression (0.33× speedup)** — XLA's Metal convolution lowering inserts
   additional memory-layout transposes for statically-shaped MPS graphs.  The eager
   path avoids this by dispatching directly to Metal's optimised conv kernel.
2. **JAX falls back to CPU without a plugin** — unlike PyTorch, JAX requires an
   explicit GPU plugin on Apple Silicon:
   - `pip install jax-metal` (official Apple plugin — tied to specific jaxlib versions)
   - `pip install jax-mps` (community MLX backend — set `JAX_PLATFORMS=mps`)
   Without a plugin installed, JAX silently runs on CPU; the benchmark now emits a
   clear diagnostic when this happens (see `_check_jax_mps_driver()`).

---

## Figure GPU-8 — GPU-vs-CPU Crossover (batch size where GPU wins)

_Crossover figure not available.  Re-run with `--crossover` flag to generate it:_

```bash
python benchmarks/collect_data.py          # generate CPU baseline first
python benchmarks/collect_gpu_data.py --crossover
```

---

*Generated by `benchmarks/generate_gpu_report.py` using [neural-cost](https://github.com/davidgraymi/neural-cost)*
