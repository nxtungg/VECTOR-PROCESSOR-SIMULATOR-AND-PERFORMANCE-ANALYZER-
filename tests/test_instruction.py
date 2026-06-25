import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instruction import Instruction, InstructionType, FunctionalUnit, InstructionTemplates


class TestInstruction(unittest.TestCase):
    
    def test_create_instruction(self):
        """Test tạo instruction"""
        inst = Instruction("VADD", "V3", ["V1", "V2"], "VADD V3, V1, V2", line_number=1)
        self.assertEqual(inst.opcode, "VADD")
        self.assertEqual(inst.dst, "V3")
        self.assertEqual(inst.src, ["V1", "V2"])
    
    def test_instruction_type(self):
        """Test phân loại instruction"""
        # Memory
        inst = Instruction("VLOAD", "V1", ["A"], "")
        self.assertEqual(inst.inst_type, InstructionType.MEMORY_LOAD)
        
        # Arithmetic binary
        inst = Instruction("VADD", "V3", ["V1", "V2"], "")
        self.assertEqual(inst.inst_type, InstructionType.ARITHMETIC_BINARY)
        
        # Arithmetic scalar
        inst = Instruction("VADDS", "V3", ["V1", "a"], "")
        self.assertEqual(inst.inst_type, InstructionType.ARITHMETIC_SCALAR)
        
        # Reduction
        inst = Instruction("VREDUCE_SUM", "S", ["V1"], "")
        self.assertEqual(inst.inst_type, InstructionType.ARITHMETIC_REDUCTION)
    
    def test_functional_unit(self):
        """Test xác định functional unit"""
        inst = Instruction("VLOAD", "V1", ["A"], "")
        self.assertEqual(inst.functional_unit, FunctionalUnit.MEMORY)
        
        inst = Instruction("VADD", "V3", ["V1", "V2"], "")
        self.assertEqual(inst.functional_unit, FunctionalUnit.ALU)
        
        inst = Instruction("VREDUCE_SUM", "S", ["V1"], "")
        self.assertEqual(inst.functional_unit, FunctionalUnit.REDUCTION)
    
    def test_register_extraction(self):
        """Test trích xuất thanh ghi"""
        inst = Instruction("VADD", "V3", ["V1", "V2"], "")
        self.assertEqual(inst.src_registers, ["V1", "V2"])
        self.assertEqual(inst.dst_register, "V3")
    
    def test_memory_operands(self):
        """Test trích xuất memory operands"""
        inst = Instruction("VLOAD", "V1", ["A"], "")
        self.assertEqual(inst.memory_operands, ["A"])
        
        inst = Instruction("VSTORE", "C", ["V3"], "")
        self.assertEqual(inst.memory_operands, ["C"])
    
    def test_has_dependency(self):
        """Test kiểm tra dependency"""
        inst = Instruction("VADD", "V3", ["V1", "V2"], "")
        self.assertTrue(inst.has_dependency)
        
        inst = Instruction("VLOAD", "V1", ["A"], "")
        self.assertFalse(inst.has_dependency)  # không có src register
    
    def test_is_memory_instruction(self):
        """Test kiểm tra memory instruction"""
        inst = Instruction("VLOAD", "V1", ["A"], "")
        self.assertTrue(inst.is_memory_instruction)
        
        inst = Instruction("VADD", "V3", ["V1", "V2"], "")
        self.assertFalse(inst.is_memory_instruction)
    
    def test_instruction_templates(self):
        """Test các template instruction"""
        inst = InstructionTemplates.vload("V1", "A")
        self.assertEqual(inst.opcode, "VLOAD")
        self.assertEqual(inst.dst, "V1")
        self.assertEqual(inst.src, ["A"])
        
        inst = InstructionTemplates.vadd("V3", "V1", "V2")
        self.assertEqual(inst.opcode, "VADD")
        
        inst = InstructionTemplates.vload_stride("V1", "A", 2)
        self.assertEqual(inst.opcode, "VLOAD_STRIDE")
        # stride duoc luu dang string trong src, kiem tra gia tri nguyen
        self.assertEqual(inst.src[0], "A")   # array name
        self.assertEqual(int(inst.src[1]), 2)  # stride = 2
    
    def test_to_dict(self):
        """Test chuyển đổi dict"""
        inst = Instruction("VADD", "V3", ["V1", "V2"], "VADD V3, V1, V2", line_number=5)
        inst_dict = inst.to_dict()
        
        self.assertEqual(inst_dict["opcode"], "VADD")
        self.assertEqual(inst_dict["dst"], "V3")
        self.assertEqual(inst_dict["line_number"], 5)


if __name__ == "__main__":
    unittest.main()