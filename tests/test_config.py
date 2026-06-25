import unittest
import sys
import os
import json
import tempfile

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig, ExperimentConfigs


class TestConfig(unittest.TestCase):
    
    def setUp(self):
        """Chạy trước mỗi test"""
        self.config = VectorProcessorConfig()
    
    def test_default_values(self):
        """Test giá trị mặc định"""
        self.assertEqual(self.config.vector_length, 8)
        self.assertEqual(self.config.num_lanes, 4)
        self.assertEqual(self.config.memory_bandwidth, 4)
        self.assertEqual(self.config.add_latency, 2)
    
    def test_validate_valid_config(self):
        """Test config hợp lệ"""
        # Không nên raise exception
        self.config.validate()
    
    def test_validate_invalid_vector_length(self):
        """Test vector_length âm"""
        self.config.vector_length = -1
        with self.assertRaises(ValueError):
            self.config.validate()
    
    def test_validate_invalid_lanes(self):
        """Test num_lanes = 0"""
        self.config.num_lanes = 0
        with self.assertRaises(ValueError):
            self.config.validate()
    
    def test_get_latency(self):
        """Test lấy latency theo opcode"""
        self.assertEqual(self.config.get_latency("VADD"), self.config.add_latency)
        self.assertEqual(self.config.get_latency("VMUL"), self.config.mul_latency)
        self.assertEqual(self.config.get_latency("UNKNOWN"), self.config.add_latency)  # default
    
    def test_compute_cycles(self):
        """Test tính số cycles"""
        # VL=8, lanes=4 → groups=2
        cycles = self.config.compute_cycles(8, self.config.add_latency)
        expected = self.config.startup_latency + self.config.add_latency + 2
        self.assertEqual(cycles, expected)
    
    def test_compute_memory_cycles(self):
        """Test tính cycles cho memory"""
        cycles = self.config.compute_memory_cycles(8, is_load=True)
        transfers = 8 // self.config.memory_bandwidth + (1 if 8 % self.config.memory_bandwidth else 0)
        expected = self.config.load_latency + transfers
        self.assertEqual(cycles, expected)
    
    def test_to_dict_and_from_dict(self):
        """Test chuyển đổi dict"""
        config_dict = self.config.to_dict()
        new_config = VectorProcessorConfig.from_dict(config_dict)
        self.assertEqual(self.config.vector_length, new_config.vector_length)
        self.assertEqual(self.config.num_lanes, new_config.num_lanes)
    
    def test_save_and_load_json(self):
        """Test lưu và đọc JSON độc lập để tránh crash luồng file trên Windows"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.config.save_to_file(temp_file)
            loaded_config = VectorProcessorConfig.load_from_file(temp_file)
            self.assertEqual(self.config.vector_length, loaded_config.vector_length)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_experiment_configs(self):
        """Test các config mẫu cho thí nghiệm"""
        exp1_configs = ExperimentConfigs.exp1_vector_length()
        self.assertEqual(len(exp1_configs), 5)  # 4,8,16,32,64
        
        exp2_configs = ExperimentConfigs.exp2_num_lanes()
        self.assertEqual(len(exp2_configs), 5)  # 1,2,4,8,16
    
    def test_peak_performance(self):
        """Test tính peak performance"""
        peak = self.config.get_peak_performance()
        expected = self.config.num_lanes * self.config.clock_frequency_mhz * 1e6
        self.assertEqual(peak, expected)
    
    def test_string_representation(self):
        """Test hiển thị config"""
        config_str = str(self.config)
        self.assertIn("Vector Length", config_str)
        self.assertIn("Number of Lanes", config_str)


if __name__ == "__main__":
    unittest.main() 