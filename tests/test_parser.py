import unittest
import sys
import os
import tempfile
import io
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import ProgramParser, ParseError


class TestParser(unittest.TestCase):
    
    def setUp(self):
        self.parser = ProgramParser()
    
    def create_temp_file(self, content):
        """Tạo file tạm thời với nội dung"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_parse_simple_program(self):
        """Test parse chương trình đơn giản"""
        content = """
DATA A = [1, 2, 3, 4]
DATA B = [10, 20, 30, 40]

VLOAD V1, A
VLOAD V2, B
VADD V3, V1, V2
VSTORE C, V3
"""
        filename = self.create_temp_file(content)
        
        try:
            data, scalars, instructions = self.parser.parse_file(filename)
            
            self.assertIn("A", data)
            self.assertEqual(data["A"], [1, 2, 3, 4])
            self.assertEqual(len(instructions), 4)
            self.assertEqual(instructions[0].opcode, "VLOAD")
            self.assertEqual(instructions[0].dst, "V1")
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_parse_with_scalar(self):
        """Test parse với scalar"""
        content = """
DATA X = [1, 2, 3, 4]
SCALAR a = 2

VLOAD V1, X
VMULS V2, V1, a
"""
        filename = self.create_temp_file(content)
        
        try:
            data, scalars, instructions = self.parser.parse_file(filename)
            
            self.assertIn("a", scalars)
            self.assertEqual(scalars["a"], 2)
            self.assertEqual(instructions[1].opcode, "VMULS")
            self.assertEqual(instructions[1].src[1], "a")
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_parse_with_comments(self):
        """Test parse với comments"""
        content = """
# This is a comment
DATA A = [1, 2, 3]  # inline comment

VLOAD V1, A  # load array A
"""
        filename = self.create_temp_file(content)
        
        try:
            data, scalars, instructions = self.parser.parse_file(filename)
            
            self.assertIn("A", data)
            self.assertEqual(len(instructions), 1)
            self.assertEqual(instructions[0].comment, "load array A")
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_parse_stride_access(self):
        """Test parse stride access"""
        content = """
DATA A = [1, 2, 3, 4, 5, 6, 7, 8]

VLOAD_STRIDE V1, A, 2
"""
        filename = self.create_temp_file(content)
        
        try:
            data, scalars, instructions = self.parser.parse_file(filename)
            
            self.assertEqual(instructions[0].opcode, "VLOAD_STRIDE")
            self.assertEqual(instructions[0].src[1], "2")
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_parse_multiline_data(self):
        """Test parse mảng DATA trải nhiều dòng"""
        content = """
DATA IMAGE = [
    10, 20,
    30, 40
]
DATA KERNEL = [1, 2, 3]

VLOAD V1, IMAGE
"""
        filename = self.create_temp_file(content)

        try:
            data, scalars, instructions = self.parser.parse_file(filename)
            self.assertEqual(data["IMAGE"], [10, 20, 30, 40])
            self.assertEqual(data["KERNEL"], [1, 2, 3])
            self.assertEqual(len(instructions), 1)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_parse_convolution_2d_example(self):
        """Test parse file example convolution_2d.txt"""
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
            "convolution_2d.txt",
        )
        if not os.path.exists(example_path):
            self.skipTest("convolution_2d.txt not found")

        data, scalars, instructions = self.parser.parse_file(example_path)
        self.assertEqual(len(data["IMAGE"]), 9)
        self.assertEqual(len(data["KERNEL"]), 9)
        self.assertEqual(scalars["norm"], 16.0)
        self.assertGreaterEqual(len(instructions), 4)

    def test_parse_gather_scatter(self):
        """Test parse gather/scatter"""
        content = """
DATA A = [100, 200, 300, 400]
DATA IDX = [0, 2]

VGATHER V1, A, IDX
VSCATTER C, IDX, V1
"""
        filename = self.create_temp_file(content)
        
        try:
            data, scalars, instructions = self.parser.parse_file(filename)
            
            self.assertEqual(instructions[0].opcode, "VGATHER")
            self.assertEqual(instructions[1].opcode, "VSCATTER")
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_parse_mask_instruction_family(self):
        """Parser phải nhận đủ các opcode masked đã khai báo hợp lệ."""
        content = """
DATA A = [1, 2, 3, 4]
DATA B = [1, 0, 3, 0]

VLOAD V1, A
VLOAD V2, B
VMASK_COND V9, V1, V2, ne
VSUB_MASKED V3, V1, V2, V9
VMOV_MASKED V4, V3, V9
"""
        filename = self.create_temp_file(content)

        try:
            _, _, instructions = self.parser.parse_file(filename)
            self.assertEqual(
                [inst.opcode for inst in instructions[-3:]],
                ["VMASK_COND", "VSUB_MASKED", "VMOV_MASKED"],
            )
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_parse_error_invalid_opcode(self):
        """Test lỗi với opcode không hợp lệ"""
        content = "INVALID V1, A"
        filename = self.create_temp_file(content)
        
        try:
            with self.assertRaises(ParseError):
                self.parser.parse_file(filename)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_parse_error_missing_equals(self):
        """Test lỗi DATA thiếu dấu ="""
        content = "DATA A [1, 2, 3]"
        filename = self.create_temp_file(content)
        
        try:
            with self.assertRaises(ParseError):
                self.parser.parse_file(filename)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_get_program_analyzer(self):
        """Test program analyzer"""
        content = """
DATA A = [1, 2, 3]
VLOAD V1, A
VADD V2, V1, V1
"""
        filename = self.create_temp_file(content)
        
        try:
            data, scalars, instructions = self.parser.parse_file(filename)
            analyzer = self.parser.get_program_analyzer()
            
            counts = analyzer.get_instruction_counts()
            self.assertEqual(counts.get("VLOAD", 0), 1)
            self.assertEqual(counts.get("VADD", 0), 1)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_print_summary(self):
        """Test in và chụp lại luồng stdout nhằm đối chiếu chuỗi kết quả kết xuất"""
        content = "DATA A = [1, 2, 3]\nVLOAD V1, A"
        filename = self.create_temp_file(content)
        
        try:
            self.parser.parse_file(filename)
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                self.parser.print_summary()
                output = mock_stdout.getvalue()
                # Xác thực log in ra không rỗng và chứa thông tin cần thiết
                self.assertTrue(len(output) > 0)
                self.assertIn("VLOAD", output)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


if __name__ == "__main__":
    unittest.main()
