import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import CachePolicy
from memory import Memory
from memory_hierarchy import MemoryHierarchy, LevelConfig


class TestMemoryHierarchy(unittest.TestCase):

    def test_single_level_hit_miss(self):
        l1 = LevelConfig(size_bytes=4 * 64, line_size_bytes=64,
                         associativity=2, penalty=0)
        h = MemoryHierarchy(l1=l1, mem_penalty=10)
        lvl0, pen0 = h.access(0)            # compulsory miss
        lvl1, pen1 = h.access(0)            # hit
        # Bất biến hành vi (không phụ thuộc giá trị penalty tuyệt đối):
        self.assertEqual(lvl0, 0)           # miss phải xuống main memory
        self.assertEqual(lvl1, 1)           # lần sau hit ở L1
        self.assertEqual(pen1, 0)           # L1 hit không bị phạt
        self.assertGreater(pen0, pen1)      # miss tốn hơn hit
        stats = h.get_stats()
        self.assertEqual(stats["l1_hits"], 1)
        self.assertEqual(stats["l1_misses"], 1)

    def test_two_level_hierarchy(self):
        l1 = LevelConfig(size_bytes=64, line_size_bytes=64,
                         associativity=1, penalty=2)
        l2 = LevelConfig(size_bytes=4 * 64, line_size_bytes=64,
                         associativity=2, penalty=6)
        h = MemoryHierarchy(l1=l1, l2=l2, mem_penalty=20)
        # lần đầu: miss cả L1 và L2 -> xuống mem
        lvl_full, pen_full = h.access(0)
        self.assertEqual(lvl_full, 0)
        # đẩy line khác vào L1 (cùng set vì L1 1 dòng) để evict line 0
        h.access(1000)
        # truy xuất lại 0: L1 miss nhưng L2 còn -> hit L2
        lvl_l2, pen_l2 = h.access(0)
        self.assertEqual(lvl_l2, 2)
        # Bất biến phân cấp: càng xuống sâu càng tốn (full miss > L2 hit > 0).
        self.assertGreater(pen_l2, 0)
        self.assertGreater(pen_full, pen_l2)
        stats = h.get_stats()
        self.assertEqual(stats["levels"], 2)
        self.assertGreaterEqual(stats["l2_hits"], 1)

    def test_memory_routes_all_paths(self):
        mem = Memory({"A": [float(i) for i in range(16)],
                      "IDX": [0, 2, 4, 6]}, bandwidth=4, latency=2)
        mem.enable_cache_simulation(cache_size=4, line_size=2, levels=2)
        # mọi đường truy xuất phải đi qua hierarchy và cộng penalty
        _, c_load = mem.load_vector("A")
        _, c_stride = mem.load_stride("A", 2, 4)
        _, c_gather = mem.gather("A", [0, 2, 4, 6])
        c_store = mem.store_vector("B", [1.0, 2.0, 3.0, 4.0])
        stats = mem.get_stats()
        self.assertTrue(stats["cache_enabled"])
        self.assertIn("hierarchy", stats)
        self.assertGreater(stats["cache_misses"], 0)
        for c in (c_load, c_stride, c_gather, c_store):
            self.assertGreater(c, 0)

    def test_policy_selection(self):
        mem = Memory({"A": [1.0, 2.0]}, bandwidth=2, latency=2)
        mem.enable_cache_simulation(cache_size=8, line_size=4, policy="fifo")
        hierarchy = mem.hierarchy
        self.assertIsNotNone(hierarchy)
        if hierarchy is None:
            self.fail("Memory hierarchy was not initialized")
        self.assertEqual(hierarchy.l1.policy, CachePolicy.FIFO)


if __name__ == "__main__":
    unittest.main()
