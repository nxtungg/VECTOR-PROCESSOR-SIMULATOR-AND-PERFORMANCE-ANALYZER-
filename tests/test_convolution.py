import unittest
import sys
import os
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig
from convolution_simulator import ConvolutionSimulator as CS


class TestConvolutionStrategies(unittest.TestCase):

    def setUp(self):
        self.image: list[list[float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
        self.kernel: list[list[float]] = [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
        self.config = VectorProcessorConfig(vector_length=64, num_lanes=4,
                                            memory_bandwidth=8)

    def _eq(
        self,
        a: Sequence[Sequence[float]],
        b: Sequence[Sequence[float]],
    ) -> None:
        self.assertEqual(len(a), len(b))
        for ra, rb in zip(a, b):
            self.assertEqual([float(x) for x in ra], [float(x) for x in rb])

    def test_im2col_matches_naive(self):
        naive = CS.conv2d_vectorized(self.image, self.kernel)
        im2col = CS.conv2d_im2col(self.image, self.kernel)
        self._eq(naive, im2col)

    def test_tiled_matches_naive(self):
        naive = CS.conv2d_vectorized(self.image, self.kernel)
        for t in (1, 2, 3):
            tiled = CS.conv2d_tiled(self.image, self.kernel, tile_size=t)
            self._eq(naive, tiled)

    def test_im2col_shape(self):
        cols, oh, ow = CS.im2col(self.image, 3, 3)
        self.assertEqual(oh, 2)
        self.assertEqual(ow, 2)
        self.assertEqual(len(cols), 4)
        self.assertTrue(cols)
        self.assertEqual(len(cols[0]), 9)

    def test_compare_strategies_keys(self):
        cmp = CS.compare_conv_strategies(16, 3, self.config,
                                         tile_size=4, cache_lines=8, line_size=8)
        for key in ("naive", "im2col", "tiled"):
            self.assertIn(key, cmp)
            s = cmp[key]
            for field in ("cache_hit_rate", "cache_penalty_cycles",
                          "total_cycles", "speedup_vs_naive", "memory_accesses"):
                self.assertIn(field, s)
            self.assertGreaterEqual(s["cache_hit_rate"], 0.0)
            self.assertLessEqual(s["cache_hit_rate"], 1.0)

    def test_im2col_has_more_traffic(self):
        cmp = CS.compare_conv_strategies(16, 3, self.config, tile_size=4)
        self.assertGreater(cmp["im2col"]["memory_accesses"],
                           cmp["naive"]["memory_accesses"])

    def test_vectorized_kernel_program_parses(self):
        from parser import ProgramParser
        prog = CS.generate_vectorized_kernel_program(self.image, self.kernel)
        data, scalars, ins = ProgramParser().parse_text(prog)
        self.assertGreater(len(ins), 0)
        self.assertIn("KERNEL", data)


if __name__ == "__main__":
    unittest.main()
