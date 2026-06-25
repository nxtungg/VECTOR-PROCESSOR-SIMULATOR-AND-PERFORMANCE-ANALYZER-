import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import Cache, CachePolicy
from memory import Memory


class TestCache(unittest.TestCase):

    def test_cache_hit_miss_stats(self):
        cache = Cache(size_bytes=256, line_size_bytes=64, associativity=2)
        self.assertFalse(cache.access(0))
        self.assertTrue(cache.access(0))
        stats = cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_memory_cache_simulation_increases_cycles(self):
        mem = Memory({"A": [1, 2, 3, 4, 5, 6, 7, 8]}, bandwidth=4, latency=2)
        _, cycles_no_cache = mem.load_vector("A")
        mem.enable_cache_simulation(cache_size=2, line_size=2)
        _, cycles_with_cache = mem.load_vector("A")
        stats = mem.get_stats()
        self.assertTrue(stats["cache_enabled"])
        self.assertGreater(stats["cache_misses"], 0)
        self.assertGreaterEqual(cycles_with_cache, cycles_no_cache)


if __name__ == "__main__":
    unittest.main()
