"""End-to-end adapter tests against each supported framework.

The frameworks remain optional dependencies, so each test skips unless its
corresponding extra is installed.  CI can exercise all of them with
``pip install -e '.[torch,jax,tensorflow,dev]'``.
"""

import importlib.util
import unittest

from neural_cost import estimate_model
from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchAdapter


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchAdapterE2ETest(unittest.TestCase):
    def test_linear_model_is_estimated_and_measured(self) -> None:
        import torch

        model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias=False), torch.nn.ReLU())
        inputs = (torch.ones((2, 8)),)
        adapter = TorchAdapter()

        estimate = estimate_model(model, inputs, adapter)
        measurement = adapter.benchmark(model, *inputs, warmup=1, repeats=2)

        self.assertEqual(estimate.operations, 1)
        self.assertEqual(estimate.flops, 128)
        self.assertGreater(measurement.median_seconds, 0)


@unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is not installed")
class JaxAdapterE2ETest(unittest.TestCase):
    def test_matmul_function_is_estimated_and_measured(self) -> None:
        import jax.numpy as jnp

        def model(left, right):
            return jnp.matmul(left, right)

        inputs = (jnp.ones((2, 8)), jnp.ones((8, 4)))
        adapter = JaxAdapter()
        estimate = estimate_model(model, inputs, adapter)
        measurement = adapter.benchmark(model, *inputs, warmup=1, repeats=2)

        self.assertEqual(estimate.operations, 1)
        self.assertEqual(estimate.flops, 128)
        self.assertGreater(measurement.median_seconds, 0)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
class TensorFlowAdapterE2ETest(unittest.TestCase):
    def test_dense_model_is_estimated_and_measured(self) -> None:
        import tensorflow as tf

        model = tf.keras.Sequential([tf.keras.layers.Dense(4, use_bias=False)])
        inputs = (tf.ones((2, 8)),)
        adapter = TensorFlowAdapter()

        estimate = estimate_model(model, inputs, adapter)
        measurement = adapter.benchmark(model, *inputs, warmup=1, repeats=2)

        self.assertEqual(estimate.operations, 1)
        self.assertEqual(estimate.flops, 128)
        self.assertGreater(measurement.median_seconds, 0)
