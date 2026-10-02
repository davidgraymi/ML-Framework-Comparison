import unittest

from neural_cost.profiler import benchmark


class ProfilerTests(unittest.TestCase):
    def test_benchmark_returns_samples(self) -> None:
        measurement = benchmark(lambda value: value + 1, 1, warmup=1, repeats=3)
        self.assertEqual(len(measurement.samples_seconds), 3)
        self.assertGreater(measurement.median_seconds, 0)

    def test_warmup_zero_is_valid(self) -> None:
        measurement = benchmark(lambda value: value + 1, 1, warmup=0, repeats=2)
        self.assertEqual(len(measurement.samples_seconds), 2)

    def test_repeats_one_is_valid(self) -> None:
        measurement = benchmark(lambda value: value + 1, 1, warmup=1, repeats=1)
        self.assertEqual(len(measurement.samples_seconds), 1)

    def test_negative_warmup_raises(self) -> None:
        with self.assertRaises(ValueError):
            benchmark(lambda value: value + 1, 1, warmup=-1, repeats=3)

    def test_zero_repeats_raises(self) -> None:
        with self.assertRaises(ValueError):
            benchmark(lambda value: value + 1, 1, warmup=1, repeats=0)

    def test_measurement_negative_median_raises(self) -> None:
        from neural_cost.profiler import Measurement

        with self.assertRaises(ValueError):
            Measurement(median_seconds=-0.1, samples_seconds=(-0.1,))

    def test_measurement_negative_memory_raises(self) -> None:
        from neural_cost.profiler import Measurement

        with self.assertRaises(ValueError):
            Measurement(median_seconds=0.1, samples_seconds=(0.1,), peak_memory_bytes=-100)
