# neural-cost

`neural-cost` estimates a neural network's useful compute and compulsory tensor
traffic, measures its runtime, and uses a roofline lower bound to highlight
likely optimization opportunities. It is intentionally framework-neutral at
its core: PyTorch, TensorFlow, and JAX are optional adapters rather than base
dependencies.

## Install

```bash
pip install -e '.[dev]'
# Choose a framework adapter when needed:
pip install -e '.[torch]'
```

## Analyze portable operations

```python
from neural_cost import HardwareSpec, Operation, analyze_gap, benchmark, estimate_operations

ops = [Operation("classifier", "linear", ((32, 768), (768, 1000)), (32, 1000), 2)]
estimate = estimate_operations(ops)
measurement = benchmark(lambda: run_inference(), warmup=5, repeats=20)
hardware = HardwareSpec("GPU", peak_flops=312e12, memory_bandwidth=1.6e12)
report = analyze_gap(estimate, measurement, hardware)

print(report.render())
```

The theoretical model reports FLOPs, tensor reads/writes, arithmetic intensity,
and a compute/bandwidth lower bound. The measured gap is expected: it captures
launch overhead, synchronization, framework behavior, unfused intermediates,
workspaces, caches, and imperfect kernel utilization.

## Framework adapters

```python
import torch
from neural_cost import estimate_model
from neural_cost.adapters import TorchAdapter

model = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10))
inputs = (torch.randn(16, 128),)
estimate = estimate_model(model, inputs, TorchAdapter())
measurement = TorchAdapter().benchmark(model, *inputs)
```

`TorchAdapter` captures `Linear` and `Conv2d` modules and synchronizes CUDA
benchmarks. `TensorFlowAdapter` captures Keras `Dense` and `Conv2D` calls.
`JaxAdapter` traces conventional `dot_general` and common elementwise jaxpr
primitives. All adapters are optional imports:

```python
from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchAdapter
```

## Custom frameworks

Subclass `FrameworkAdapter` and implement `operations(model, example_inputs)`
to return portable `Operation` records. The adapter can also override
`benchmark` to synchronize an accelerator or collect framework-specific memory
statistics. This contract keeps model extraction separate from the framework-
independent estimator and analyzer.

## Current scope

This first release models inference with concrete shapes and common dense,
matrix-multiply, convolution, and elementwise operations. Training-mode
backpropagation, activation checkpointing, distributed communication, dynamic
shapes, operator fusion details, and complete graph coverage are deliberately
next increments rather than silently approximated.
