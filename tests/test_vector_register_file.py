import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_register_file import VectorRegisterFile, RegisterStatus


class TestVectorRegisterFile(unittest.TestCase):
    
    def setUp(self):
        self.vrf = VectorRegisterFile(num_registers=8, vector_length=8, enable_chaining=True)
        
    def assertListAlmostEqual(self, list1, list2, places=4):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(float(a), float(b), places=places)
    
    def test_write_and_read(self):
        """Test ghi và đọc"""
        data = [1.0, 2.0, 3.0, 4.0]
        self.vrf.write("V0", data)
        read_data = self.vrf.read("V0")
        self.assertListAlmostEqual(read_data, data)
    
    def test_read_invalid_register(self):
        """Test đọc thanh ghi không tồn tại"""
        with self.assertRaises(KeyError):
            self.vrf.read("V99")
    
    def test_ready_status(self):
        """Test trạng thái ready"""
        self.vrf.write("V0", [1, 2, 3], current_cycle=0, latency=5)
        
        self.assertFalse(self.vrf.is_ready("V0", 3))
        self.assertTrue(self.vrf.is_ready("V0", 5))
    
    def test_get_ready_cycle(self):
        """Test lấy ready cycle"""
        self.vrf.write("V0", [1, 2, 3], current_cycle=10, latency=4)
        self.assertEqual(self.vrf.get_ready_cycle("V0"), 14)
        
    def test_write_hazard_overwrite(self):
        """Test xung đột dữ liệu WAW hazard khi ghi đè lên thanh ghi chưa kịp ready"""
        # Lệnh 1: ghi V0 tại cycle 0, chiếm dụng đến cycle 5
        self.vrf.write("V0", [1, 2, 3], current_cycle=0, latency=5)
        # Lệnh 2: ghi đè V0 tại cycle 2, latency 4 -> Phải ready tại cycle 6
        self.vrf.write("V0", [4, 5, 6], current_cycle=2, latency=4)
        
        self.assertEqual(self.vrf.get_ready_cycle("V0"), 6)
        self.assertFalse(self.vrf.is_ready("V0", 5))
        self.assertTrue(self.vrf.is_ready("V0", 6))
    
    def test_partial_write(self):
        """Test ghi một phần"""
        self.vrf.write("V0", [0, 0, 0, 0, 0, 0, 0, 0])
        self.vrf.write_partial("V0", 2, [99, 100])
        
        read_data = self.vrf.read("V0")
        self.assertAlmostEqual(read_data[2], 99)
        self.assertAlmostEqual(read_data[3], 100)
    
    def test_forwarding(self):
        """Test chaining/forwarding"""
        self.vrf.add_forwarding("V1", 10, [1.0, 2.0, 3.0])
        
        # Nên lấy được dữ liệu forwarding chuẩn xác
        data = self.vrf.read("V1", current_cycle=10)
        self.assertListAlmostEqual(data, [1.0, 2.0, 3.0])
        data = self.vrf.read("V1", current_cycle=15)
        self.assertListAlmostEqual(data, [1.0, 2.0, 3.0])
    
    def test_get_stats(self):
        """Test thống kê"""
        self.vrf.write("V0", [1, 2, 3])
        self.vrf.read("V0")
        
        stats = self.vrf.get_stats()
        self.assertEqual(stats["total_reads"], 1)
        self.assertEqual(stats["total_writes"], 1)
    
    def test_clear(self):
        """Test xóa thanh ghi"""
        self.vrf.write("V0", [1, 2, 3])
        self.vrf.clear("V0")
        
        read_data = self.vrf.read("V0")
        self.assertEqual(list(read_data), [])
    
    def test_dump(self):
        """Test dump"""
        self.vrf.write("V0", [1, 2, 3])
        dumped = self.vrf.dump()
        self.assertIn("V0", dumped)


if __name__ == "__main__":
    unittest.main()