import unittest

from neural_cost.hardware import HardwareSpec


class HardwareTests(unittest.TestCase):
    def test_valid_construction(self):
        spec = HardwareSpec(name="TestGPU", peak_flops=1e12, memory_bandwidth=1e9)
        self.assertEqual(spec.name, "TestGPU")
        self.assertEqual(spec.peak_flops, 1e12)
        self.assertEqual(spec.memory_bandwidth, 1e9)
        self.assertIsNone(spec.memory_capacity)

    def test_ridge_point(self):
        spec = HardwareSpec(name="TestGPU", peak_flops=2e12, memory_bandwidth=1e9)
        self.assertEqual(spec.ridge_point, 2000.0)

    def test_invalid_peak_flops(self):
        with self.assertRaises(ValueError):
            HardwareSpec(name="TestGPU", peak_flops=0, memory_bandwidth=1e9)
        with self.assertRaises(ValueError):
            HardwareSpec(name="TestGPU", peak_flops=-1, memory_bandwidth=1e9)

    def test_invalid_memory_bandwidth(self):
        with self.assertRaises(ValueError):
            HardwareSpec(name="TestGPU", peak_flops=1e12, memory_bandwidth=0)
        with self.assertRaises(ValueError):
            HardwareSpec(name="TestGPU", peak_flops=1e12, memory_bandwidth=-1)

    def test_optional_memory_capacity(self):
        spec = HardwareSpec(name="TestGPU", peak_flops=1e12, memory_bandwidth=1e9, memory_capacity=1024)
        self.assertEqual(spec.memory_capacity, 1024)

if __name__ == "__main__":
    unittest.main()
