import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig
from parser import ProgramParser
from vector_simulator import VectorSimulator
from pipeline_simulator import PipelineSimulator

class TestMaskOperations(unittest.TestCase):
    
    def setUp(self):
        self.config = VectorProcessorConfig(vector_length=8, num_lanes=4, num_vector_registers=16, enable_chaining=True)
    
    def test_vmask_creation(self):
        """Test tạo mask từ điều kiện - so sánh dựa trên logic bộ xử lý vector toán hạng.
        Data [1..8], mean=4.5, gt → phần tử > 4.5 → index 4,5,6,7 (values 5,6,7,8)
        """
        program = """
DATA A = [1, 2, 3, 4, 5, 6, 7, 8]

VLOAD V1, A
VMASK V9, V1, gt
"""
        parser = ProgramParser()
        data, scalars, instructions = parser.parse_text(program)
        
        sim = VectorSimulator(self.config, data, scalars)
        memory, timeline, cycles = sim.run(instructions)
        
        # Kiem tra V9 thuc su ton tai trong mask_registers (khong dung default [])
        # Bug truoc: .get("V9", []) nen assertIsNotNone luon pass du V9 khong ton tai
        self.assertIn("V9", sim.mask_registers,
                    "Thanh ghi mat na V9 khong ton tai hoac chua duoc khoi tao!")
        mask = sim.mask_registers["V9"]  # Lay truc tiep sau khi xac nhan ton tai
        
        expected = [False, False, False, False, True, True, True, True]
        self.assertEqual(list(mask), expected)
    
    def test_masked_addition(self):
        """Test cộng có điều kiện với mask
        Data A=[1..8], B=[10,20..80], mask gt (>4.5)
        Index 0-3: giữ nguyên A, Index 4-7: A+B
        """
        program = """
DATA A = [1, 2, 3, 4, 5, 6, 7, 8]
DATA B = [10, 20, 30, 40, 50, 60, 70, 80]

VLOAD V1, A
VLOAD V2, B
VMASK V9, V1, gt
VADD_MASKED V3, V1, V2, V9
VSTORE C, V3
"""
        parser = ProgramParser()
        data, scalars, instructions = parser.parse_text(program)
        
        sim = VectorSimulator(self.config, data, scalars)
        memory, timeline, cycles = sim.run(instructions)
        
        result = memory.get('C', [])
        expected = [1, 2, 3, 4, 55, 66, 77, 88]
        self.assertEqual(list(result), expected)

    def test_pipeline_vmask_cond_ne(self):
        """Pipeline phải hỗ trợ điều kiện ne giống VectorSimulator."""
        program = """
DATA A = [1, 2, 3, 4]
DATA B = [1, 0, 3, 0]

VLOAD V1, A
VLOAD V2, B
VMASK_COND V9, V1, V2, ne
VADD_MASKED V3, V1, V2, V9
VSTORE C, V3
"""
        parser = ProgramParser()
        data, scalars, instructions = parser.parse_text(program)

        sim = PipelineSimulator(self.config, data, scalars)
        memory, _, _ = sim.run(instructions)

        self.assertEqual(memory["C"], [1, 2, 3, 4])

if __name__ == "__main__":
    unittest.main()
