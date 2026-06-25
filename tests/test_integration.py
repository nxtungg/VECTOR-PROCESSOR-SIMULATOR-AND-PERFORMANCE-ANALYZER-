import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig
from parser import ProgramParser
from scalar_simulator import ScalarSimulator
from vector_simulator import VectorSimulator
from pipeline_simulator import PipelineSimulator
from performance_analyzer import PerformanceAnalyzer


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.config = VectorProcessorConfig(
            vector_length=8,
            num_lanes=4,
            memory_bandwidth=4
        )
        
    def assertListAlmostEqual(self, list1, list2, places=4):
        """Hàm helper so sánh hai danh sách số thực có sai số số thập phân"""
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(float(a), float(b), places=places)
    
    def test_vector_add_integration(self):
        """Test tích hợp vector addition"""
        parser = ProgramParser()
        
        program = """
DATA A = [1, 2, 3, 4, 5, 6, 7, 8]
DATA B = [10, 20, 30, 40, 50, 60, 70, 80]

VLOAD V1, A
VLOAD V2, B
VADD V3, V1, V2
VSTORE C, V3
"""
        
        data, scalars, instructions = parser.parse_text(program)
        
        scalar_sim = ScalarSimulator(self.config)
        scalar_cycles = scalar_sim.estimate_cycles(instructions, data)
        
        vector_sim = VectorSimulator(self.config, data, scalars)
        memory, timeline, vector_cycles = vector_sim.run(instructions)
        
        pipeline_sim = PipelineSimulator(self.config, data, scalars)
        memory2, timeline2, pipeline_cycles = pipeline_sim.run(instructions)
        
        self.assertIn("C", memory)
        self.assertListAlmostEqual(memory["C"], [11, 22, 33, 44, 55, 66, 77, 88])
        
        self.assertGreater(scalar_cycles, vector_cycles)
        self.assertGreaterEqual(vector_cycles, pipeline_cycles)
        
        analyzer = PerformanceAnalyzer(self.config)
        speedup = analyzer.calculate_speedup(scalar_cycles, pipeline_cycles)
        self.assertGreater(speedup, 1)
    
    def test_dot_product_integration(self):
        """Test tích hợp dot product"""
        parser = ProgramParser()
        
        program = """
DATA A = [1, 2, 3, 4]
DATA B = [10, 20, 30, 40]

VLOAD V1, A
VLOAD V2, B
VMUL V3, V1, V2
VREDUCE_SUM s, V3
"""
        
        data, scalars, instructions = parser.parse_text(program)
        vector_sim = VectorSimulator(self.config, data, scalars)
        memory, timeline, cycles = vector_sim.run(instructions)
        
        self.assertIn('s', vector_sim.scalars)
        result = vector_sim.scalars['s']
        expected = 1*10 + 2*20 + 3*30 + 4*40  # = 300
        self.assertAlmostEqual(float(result), float(expected), places=4)
    
    def test_saxpy_integration(self):
        """Test tích hợp SAXPY dùng cấu trúc kiểm tra số thực chuẩn xác"""
        parser = ProgramParser()
        
        program = """
DATA X = [1, 2, 3, 4, 5, 6, 7, 8]
DATA Y = [10, 10, 10, 10, 10, 10, 10, 10]
SCALAR a = 2

VLOAD V1, X
VLOAD V2, Y
VMULS V3, V1, a
VADD V4, V3, V2
VSTORE Y, V4
"""
        
        data, scalars, instructions = parser.parse_text(program)
        vector_sim = VectorSimulator(self.config, data, scalars)
        memory, timeline, cycles = vector_sim.run(instructions)
        
        expected = [2*x + 10 for x in range(1, 9)]
        self.assertIn("Y", memory)
        self.assertListAlmostEqual(memory["Y"], expected)


if __name__ == "__main__":
    unittest.main()