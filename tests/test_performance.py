import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig
from performance_analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzer(unittest.TestCase):

    def setUp(self):
        self.config = VectorProcessorConfig(vector_length=8, num_lanes=4, memory_bandwidth=4)
        self.analyzer = PerformanceAnalyzer(self.config)

    def test_speedup(self):
        self.assertEqual(self.analyzer.calculate_speedup(100, 20), 5.0)
        self.assertEqual(self.analyzer.calculate_speedup(100, 0), 0)

    def test_throughput(self):
        self.assertAlmostEqual(self.analyzer.calculate_throughput(8, 20), 0.4)

    def test_lane_utilization(self):
        util = self.analyzer.calculate_lane_utilization(8)
        self.assertAlmostEqual(util, 1.0)

    def test_memory_bandwidth_utilization_is_bounded(self):
        self.assertEqual(
            self.analyzer.calculate_memory_bandwidth_utilization(100, 10),
            1.0,
        )
        self.assertEqual(
            self.analyzer.calculate_memory_bandwidth_utilization(-1, 10),
            0.0,
        )

    def test_efficiency_is_bounded(self):
        self.assertEqual(self.analyzer.calculate_efficiency(8.0, 4), 1.0)
        self.assertEqual(self.analyzer.calculate_efficiency(-1.0, 4), 0.0)

    def test_execution_time_ns(self):
        ns = self.analyzer.execution_time_ns(1000)
        self.assertAlmostEqual(ns, 1000.0)

    def test_summarize_row(self):
        row = self.analyzer.summarize("pipeline", 88, 11, 8)
        self.assertEqual(row["mode"], "pipeline")
        self.assertAlmostEqual(row["speedup"], 88 / 11)
        self.assertIn("throughput_elem_per_cycle", row)


if __name__ == "__main__":
    unittest.main()
