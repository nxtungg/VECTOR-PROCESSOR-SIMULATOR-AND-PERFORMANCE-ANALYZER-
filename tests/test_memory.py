import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import Memory, MemoryAccessType


class TestMemory(unittest.TestCase):
    
    def setUp(self):
        self.data = {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "B": [10.0, 20.0, 30.0, 40.0]
        }
        self.memory = Memory(self.data, bandwidth=4, latency=3)
        
    def assertListAlmostEqual(self, list1, list2, places=5):
        """Helper so sánh mảng số thực để loại bỏ sai số dấu phẩy động"""
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(float(a), float(b), places=places)
    
    def test_load_vector(self):
        """Test load vector tuần tự"""
        data, cycles = self.memory.load_vector("A")
        self.assertListAlmostEqual(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertGreater(cycles, 0)
    
    def test_store_vector(self):
        """Test store vector"""
        new_data = [100.0, 200.0, 300.0, 400.0]
        cycles = self.memory.store_vector("C", new_data)
        self.assertIn("C", self.memory.data)
        self.assertListAlmostEqual(self.memory.data["C"], new_data)
    
    def test_load_stride(self):
        """Test load với stride"""
        data, cycles = self.memory.load_stride("A", stride=2, length=4)
        expected = [1.0, 3.0, 5.0, 7.0]
        self.assertListAlmostEqual(data, expected)
    
    def test_gather(self):
        """Test gather operation"""
        indices = [0, 2, 4, 6]
        data, cycles = self.memory.gather("A", indices)
        expected = [1.0, 3.0, 5.0, 7.0]
        self.assertListAlmostEqual(data, expected)
    
    def test_scatter(self):
        """Test scatter operation"""
        indices = [1, 3, 5, 7]
        values = [99.0, 88.0, 77.0, 66.0]
        cycles = self.memory.scatter("A", indices, values)
        
        self.assertAlmostEqual(self.memory.data["A"][1], 99.0)
        self.assertAlmostEqual(self.memory.data["A"][3], 88.0)
        self.assertAlmostEqual(self.memory.data["A"][5], 77.0)
        self.assertAlmostEqual(self.memory.data["A"][7], 66.0)
    
    def test_load_range(self):
        """Test load một đoạn mảng"""
        data, cycles = self.memory.load_range("A", 2, 6)
        expected = [3.0, 4.0, 5.0, 6.0]
        self.assertListAlmostEqual(data, expected)
    
    def test_get_stats(self):
        """Test thống kê memory"""
        self.memory.load_vector("A")
        self.memory.store_vector("C", [1, 2, 3])
        
        stats = self.memory.get_stats()
        self.assertEqual(stats["total_loads"], 1)
        self.assertEqual(stats["total_stores"], 1)
    
    def test_get_bandwidth_utilization(self):
        """Test tính bandwidth utilization"""
        self.memory.load_vector("A")
        util = self.memory.get_bandwidth_utilization(total_cycles=100)
        self.assertBetween(util, 0, 1)

        self.assertEqual(self.memory.get_bandwidth_utilization(total_cycles=1), 1.0)
        self.assertEqual(self.memory.get_bandwidth_utilization(total_cycles=-1), 0.0)
    
    def assertBetween(self, value, low, high):
        """Helper để kiểm tra value nằm trong khoảng"""
        self.assertGreaterEqual(value, low)
        self.assertLessEqual(value, high)
    
    def test_access_history(self):
        """Test lịch sử truy xuất"""
        self.memory.load_vector("A")
        self.memory.print_access_history(limit=5)
        self.assertEqual(len(self.memory.access_history), 1)
    
    def test_has_array(self):
        """Test kiểm tra mảng tồn tại"""
        self.assertTrue(self.memory.has_array("A"))
        self.assertFalse(self.memory.has_array("Z"))
    
    def test_dump(self):
        """Test xuất memory"""
        dumped = self.memory.dump()
        self.assertIn("A", dumped)
        self.assertIn("B", dumped)


if __name__ == "__main__":
    unittest.main()
