"""End-to-end adapter tests against each supported framework.

The frameworks remain optional dependencies, so each test skips unless its
corresponding extra is installed.  CI can exercise all of them with
``pip install -e '.[torch,jax,tensorflow,dev]'``.
"""

import importlib.util
import unittest

from neural_cost import estimate_model, profile_model
from neural_cost.adapters import JaxAdapter, TensorFlowAdapter, TorchAdapter


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchAdapterE2ETest(unittest.TestCase):
    def test_linear_model_is_estimated_and_measured(self) -> None:
        import torch

        model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias=False), torch.nn.ReLU())
        inputs = (torch.ones((2, 8)),)
        adapter = TorchAdapter()

        estimate = estimate_model(model, inputs, adapter)
        profile = profile_model(model, inputs, adapter)
        measurement = adapter.benchmark(model, *inputs, warmup=1, repeats=2)

        self.assertEqual(estimate.operations, 1)
        self.assertEqual(estimate.flops, 128)
        self.assertEqual(profile.memory.parameter_bytes, 128)
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
        profile = profile_model(model, inputs, adapter)
        measurement = adapter.benchmark(model, *inputs, warmup=1, repeats=2)

        self.assertEqual(estimate.operations, 1)
        self.assertEqual(estimate.flops, 128)
        self.assertEqual(profile.memory.parameter_bytes, 128)
        self.assertGreater(measurement.median_seconds, 0)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
class TensorFlowAdapterE2ETest(unittest.TestCase):
    def test_dense_model_is_estimated_and_measured(self) -> None:
        import tensorflow as tf

        model = tf.keras.Sequential([tf.keras.layers.Dense(4, use_bias=False)])
        inputs = (tf.ones((2, 8)),)
        adapter = TensorFlowAdapter()

        estimate = estimate_model(model, inputs, adapter)
        profile = profile_model(model, inputs, adapter)
        measurement = adapter.benchmark(model, *inputs, warmup=1, repeats=2)

        self.assertEqual(estimate.operations, 1)
        self.assertEqual(estimate.flops, 128)
        self.assertEqual(profile.memory.parameter_bytes, 128)
        self.assertGreater(measurement.median_seconds, 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class TorchAdapterExpandedE2ETest(unittest.TestCase):
    def test_conv2d_model(self):
        import torch
        model = torch.nn.Conv2d(3, 16, kernel_size=3, bias=False)
        inputs = (torch.randn(1, 3, 8, 8),)
        adapter = TorchAdapter()
        estimate = estimate_model(model, inputs, adapter)
        self.assertGreater(estimate.flops, 0)
        self.assertGreater(estimate.total_bytes, 0)
        
    def test_multi_layer_model(self):
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10, bias=False),
        )
        inputs = (torch.randn(8, 128),)
        adapter = TorchAdapter()
        estimate = estimate_model(model, inputs, adapter)
        # Should capture 3 linear operations
        self.assertEqual(estimate.operations, 3)
        self.assertGreater(estimate.flops, 0)
        
    def test_profile_with_training(self):
        import torch
        model = torch.nn.Linear(64, 32, bias=False)
        inputs = (torch.randn(4, 64),)
        from neural_cost import profile_model
        profile = profile_model(model, inputs, TorchAdapter(), training=True, optimizer_state_multiplier=2)
        self.assertGreater(profile.memory.gradient_bytes, 0)
        self.assertGreater(profile.memory.optimizer_state_bytes, 0)
        self.assertGreater(profile.memory.training_minimum_bytes, profile.memory.inference_minimum_bytes)


@unittest.skipUnless(importlib.util.find_spec("jax"), "jax not installed")
class JaxAdapterExpandedE2ETest(unittest.TestCase):
    def test_elementwise_ops_captured(self):
        import jax.numpy as jnp
        def model(x):
            return jnp.exp(x) + jnp.tanh(x)
        inputs = (jnp.ones((4, 8)),)
        adapter = JaxAdapter()
        estimate = estimate_model(model, inputs, adapter)
        # Should capture elementwise operations
        self.assertGreater(estimate.operations, 0)
        self.assertGreater(estimate.flops, 0)
        
    def test_multi_matmul(self):
        import jax.numpy as jnp
        def model(x, w1, w2):
            return jnp.matmul(jnp.matmul(x, w1), w2)
        inputs = (jnp.ones((4, 8)), jnp.ones((8, 16)), jnp.ones((16, 4)))
        adapter = JaxAdapter()
        estimate = estimate_model(model, inputs, adapter)
        self.assertEqual(estimate.operations, 2)  # two matmuls


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "tensorflow not installed")
class TensorFlowAdapterExpandedE2ETest(unittest.TestCase):
    def test_multi_layer_dense(self):
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.Dense(32, use_bias=False),
            tf.keras.layers.Dense(10, use_bias=False),
        ])
        inputs = (tf.ones((8, 128)),)
        adapter = TensorFlowAdapter()
        estimate = estimate_model(model, inputs, adapter)
        self.assertEqual(estimate.operations, 3)
        self.assertGreater(estimate.flops, 0)

    def test_conv2d_model(self):
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, use_bias=False, padding='valid'),
        ])
        inputs = (tf.ones((1, 8, 8, 3)),)
        adapter = TensorFlowAdapter()
        estimate = estimate_model(model, inputs, adapter)
        self.assertGreater(estimate.flops, 0)
        self.assertGreater(estimate.total_bytes, 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class TorchFxAdapterE2ETest(unittest.TestCase):
    def test_captures_inline_functional_and_residual_add(self):
        import torch
        from neural_cost.adapters import TorchFxAdapter

        class ResidualBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(64, 64, bias=False)

            def forward(self, x):
                res = x
                x = self.fc(x)
                x = torch.relu(x)
                return x + res  # Inline functional addition

        model = ResidualBlock()
        inputs = (torch.ones((4, 64)),)
        adapter = TorchFxAdapter()
        estimate = estimate_model(model, inputs, adapter)

        # Should capture Linear, ReLU, and addition!
        self.assertGreaterEqual(estimate.operations, 3)
        self.assertGreater(estimate.flops, 0)

    def test_fallback_on_untraceable_control_flow(self):
        import torch
        from neural_cost.adapters import TorchFxAdapter

        class DynamicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(32, 16, bias=False)

            def forward(self, x):
                # Dynamic python control flow prevents FX symbolic tracing
                if x.sum() > 0:
                    return self.fc(x)
                return self.fc(x)

        model = DynamicModel()
        inputs = (torch.ones((2, 32)),)
        adapter = TorchFxAdapter()
        estimate = estimate_model(model, inputs, adapter)
        self.assertGreaterEqual(estimate.operations, 1)

