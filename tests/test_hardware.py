import unittest

from neural_cost.hardware import CacheSpec, HardwareSpec


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

    def test_cache_spec_valid(self):
        cache = CacheSpec("L2", bandwidth=3e12, capacity=40 * 1024 * 1024)
        self.assertEqual(cache.name, "L2")
        self.assertEqual(cache.bandwidth, 3e12)
        self.assertEqual(cache.capacity, 40 * 1024 * 1024)

    def test_cache_spec_invalid(self):
        with self.assertRaises(ValueError):
            CacheSpec("L2", bandwidth=0, capacity=1024)
        with self.assertRaises(ValueError):
            CacheSpec("L2", bandwidth=1e9, capacity=0)

    def test_hardware_spec_caches(self):
        l1 = CacheSpec("L1", bandwidth=10e12, capacity=128 * 1024)
        l2 = CacheSpec("L2", bandwidth=3e12, capacity=40 * 1024 * 1024)
        spec = HardwareSpec("TestGPU", peak_flops=1e12, memory_bandwidth=1e9, caches=(l1, l2))
        self.assertEqual(len(spec.caches), 2)
        self.assertEqual(spec.get_cache("L1"), l1)
        self.assertEqual(spec.get_cache("l2"), l2)
        self.assertIsNone(spec.get_cache("L3"))

        # Working set fitting tests
        self.assertEqual(spec.find_resident_cache(64 * 1024), l1)
        self.assertEqual(spec.find_resident_cache(10 * 1024 * 1024), l2)
        self.assertIsNone(spec.find_resident_cache(100 * 1024 * 1024))


if __name__ == "__main__":
    unittest.main()

