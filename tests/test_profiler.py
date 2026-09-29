import unittest

from neural_cost.profiler import benchmark


class ProfilerTests(unittest.TestCase):
    def test_benchmark_returns_samples(self) -> None:
        measurement = benchmark(lambda value: value + 1, 1, warmup=1, repeats=3)
        self.assertEqual(len(measurement.samples_seconds), 3)
        self.assertGreater(measurement.median_seconds, 0)
