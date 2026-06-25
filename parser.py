import re
import ast
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from instruction import Instruction, InstructionTemplates, ProgramAnalyzer

class ParseError(Exception):
    """Lỗi parse chương trình"""
    pass

class ProgramParser:
    """
    Parser cho chương trình vector processor
    
    Định dạng file:
        DATA <tên_mảng> = [giá_trị1, giá_trị2, ...]
        SCALAR <tên_biến> = <giá_trị>
        <opcode> <dst>, <src1>, <src2> ...
    
    Ví dụ:
        DATA A = [1, 2, 3, 4, 5, 6, 7, 8]
        DATA B = [10, 20, 30, 40, 50, 60, 70, 80]
        SCALAR a = 2
        
        VLOAD V1, A
        VLOAD V2, B
        VMULS V3, V1, a
        VADD V4, V3, V2
        VSTORE C, V4
    """
    
    def __init__(self):
        # Các thuộc tính của class
        self.data: Dict[str, List[float]] = {}
        self.scalars: Dict[str, Any] = {}
        self.instructions: List[Instruction] = []
        self.line_number: int = 0
        self.warnings: List[str] = []           
    
        # Định nghĩa các opcode hợp lệ
        self.valid_opcodes = {
            # Memory instructions
            "VLOAD", "VSTORE", "VLOAD_STRIDE", "VGATHER", "VSCATTER",
            # Arithmetic binary
            "VADD", "VSUB", "VMUL", "VDIV", "VMAX", "VMIN",
            # Arithmetic scalar
            "VADDS", "VSUBS", "VMULS", "VDIVS",
            # Reduction
            "VREDUCE_SUM", "VREDUCE_MAX", "VREDUCE_MIN",
            # Dot product
            "VDOT",
            # Logical
            "VAND", "VOR", "VXOR",
            # Mask
            "VMASK",
            "VADD_MASKED",
            "VMASK_COND", "VSUB_MASKED", "VMOV_MASKED",
        }
        
        # Số lượng toán hạng cho mỗi opcode
        self.operand_count = {
            "VLOAD": 2, "VSTORE": 2, "VLOAD_STRIDE": 3, "VGATHER": 3, "VSCATTER": 3,
            "VADD": 3, "VSUB": 3, "VMUL": 3, "VDIV": 3, "VMAX": 3, "VMIN": 3,
            "VADDS": 3, "VSUBS": 3, "VMULS": 3, "VDIVS": 3,
            "VREDUCE_SUM": 2, "VREDUCE_MAX": 2, "VREDUCE_MIN": 2,
            "VDOT": 3,
            "VAND": 3, "VOR": 3, "VXOR": 3,
            "VMASK": 3,
            "VMASK_COND": 4,
            "VADD_MASKED": 4,
            "VSUB_MASKED": 4,
            "VMOV_MASKED": 3,
        }
    
    def parse_file(self, filepath: str) -> Tuple[Dict, Dict, List[Instruction]]:
        """Parse file chương trình"""
        path = Path(filepath)
        if not path.exists():
            raise ParseError(f"File not found: {filepath}")
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return self.parse_text(content, str(path))
    
    def parse_text(self, text: str, source_name: str = "<string>") -> Tuple[Dict, Dict, List[Instruction]]:
        """Parse text content"""
        self.data = {}
        self.scalars = {}
        self.instructions = []
        self.warnings = []
        
        lines = text.strip().split('\n')
        line_idx = 0
        
        while line_idx < len(lines):
            self.line_number = line_idx + 1
            line = lines[line_idx].strip()
            
            # Bỏ qua dòng trống và comment
            if not line or line.startswith('#'):
                line_idx += 1
                continue
            
            # Xóa comment ở cuối dòng (nếu có)
            if '#' in line:
                comment_start = line.index('#')
                comment = line[comment_start+1:].strip()
                line = line[:comment_start].strip()
            else:
                comment = ""
            
            try:
                if line.startswith('DATA'):
                    line, line_idx = self._collect_data_declaration(lines, line_idx, line)
                    self._parse_data_line(line, comment)
                    line_idx += 1
                elif line.startswith('SCALAR'):
                    self._parse_scalar_line(line, comment)
                    line_idx += 1
                else:
                    self._parse_instruction_line(line, comment)
                    line_idx += 1
            except ParseError as e:
                raise ParseError(f"Error at line {self.line_number}: {e}\nLine: {line}")
        
        # Kiểm tra tính hợp lệ của chương trình
        self._validate_program()
        
        return self.data, self.scalars, self.instructions
    
    def _strip_line_comment(self, raw_line: str) -> str:
        """Loại bỏ comment inline, giữ nguyên nội dung trước #."""
        if '#' not in raw_line:
            return raw_line.strip()
        return raw_line[:raw_line.index('#')].strip()

    def _collect_data_declaration(self, lines: List[str], start_idx: int,
                                first_line: str) -> Tuple[str, int]:
        """
        Gom các dòng DATA khi mảng trải nhiều dòng:
            DATA IMAGE = [
                1, 2, 3,
                4, 5, 6
            ]
        """
        parts = [first_line]
        bracket_depth = first_line.count('[') - first_line.count(']')
        idx = start_idx

        while bracket_depth > 0 and idx + 1 < len(lines):
            idx += 1
            self.line_number = idx + 1
            next_line = self._strip_line_comment(lines[idx])
            if not next_line:
                continue
            parts.append(next_line)
            bracket_depth += next_line.count('[') - next_line.count(']')

        if bracket_depth != 0:
            raise ParseError("Unclosed array brackets '[]' in DATA declaration")

        combined = ' '.join(parts)
        combined = re.sub(r'\s+', ' ', combined)
        return combined, idx

    def _parse_data_line(self, line: str, comment: str):
        """Parse dòng DATA A = [1, 2, 3] (hỗ trợ một hoặc nhiều dòng)"""
        # Bỏ prefix "DATA"
        content = line[4:].strip()
        
        # Tìm dấu =
        if '=' not in content:
            raise ParseError("DATA line must contain '='")
        
        name_part, array_part = content.split('=', 1)
        name = name_part.strip()
        
        # Kiểm tra tên hợp lệ
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            raise ParseError(f"Invalid array name: {name}")
        
        array_part = array_part.strip()
        
        # Parse mảng
        if not (array_part.startswith('[') and array_part.endswith(']')):
            raise ParseError("Array must be enclosed in brackets []")
        
        try:
            parsed = ast.literal_eval(array_part)
        except (ValueError, SyntaxError) as e:
            raise ParseError(f"Invalid array literal: {e}")

        if not isinstance(parsed, list):
            raise ParseError("DATA value must be a list")

        values = []
        for item in parsed:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ParseError(f"Invalid array element: {item}")
            values.append(item)
        self.data[name] = values
    
    def _parse_scalar_line(self, line: str, comment: str):
        """Parse dòng SCALAR a = 2 hoặc SCALAR result = S"""
        # Bỏ prefix "SCALAR"
        content = line[6:].strip()
        
        # Tìm dấu =
        if '=' not in content:
            raise ParseError("SCALAR line must contain '='")
        
        name_part, value_part = content.split('=', 1)
        name = name_part.strip()
        
        # Kiểm tra tên hợp lệ
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            raise ParseError(f"Invalid scalar name: {name}")
        
        value_part = value_part.strip()
        if value_part.isalpha():
            # Lưu dưới dạng string, sẽ xử lý sau
            self.scalars[name] = value_part
        else:
            try:
                self.scalars[name] = float(value_part)
            except ValueError:
                raise ParseError(f"Invalid scalar value: {value_part}")
    
    def _parse_instruction_line(self, line: str, comment: str):
        """Parse dòng lệnh: VADD V3, V1, V2"""
        # Tách opcode và phần còn lại
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            raise ParseError("Instruction must have opcode and operands")
        
        opcode = parts[0].upper()
        rest = parts[1].strip()

        # Kiểm tra opcode hợp lệ
        if opcode not in self.valid_opcodes:
            raise ParseError(f"Invalid opcode: {opcode}. Valid opcodes: {sorted(self.valid_opcodes)}")

        # ===== MỞ RỘNG: predicate per-lane tùy chọn dạng '@Vx' =====
        # Cú pháp THÊM (không phá format cũ): bất kỳ lệnh vector nào cũng có thể
        # gắn một mask register ở cuối bằng '@Vx', vd:
        #     VADD V3, V1, V2 @V9
        # Chương trình cũ không dùng '@' nên hoàn toàn không bị ảnh hưởng.
        mask_reg: Optional[str] = None
        mask_match = re.search(r'@\s*(V\d+)\b', rest)
        if mask_match:
            mask_reg = mask_match.group(1)
            rest = (rest[:mask_match.start()] + rest[mask_match.end():]).strip()
        elif '@' in rest:
            raise ParseError("Predicate mask phải có dạng '@Vx' (vd '@V9')")

        # Tách toán hạng
        # Xóa dấu phẩy và tách
        rest = rest.replace(',', ' ')
        operands = [op.strip() for op in rest.split() if op.strip()]
        
        # Kiểm tra số lượng toán hạng
        expected = self.operand_count.get(opcode, 2)
        if len(operands) != expected:
            raise ParseError(f"{opcode} expected {expected} operands, got {len(operands)}")
        
        # Parse theo loại lệnh
        if opcode == "VLOAD":
            # VLOAD V1, A
            dst = operands[0]
            src = [operands[1]]
            self._validate_register(dst)
        
        elif opcode == "VGATHER":
            # VGATHER V1, A, IDX
            dst = operands[0]
            src = [operands[1], operands[2]]  # array_name, idx_array
            self._validate_register(dst)
            
        elif opcode == "VSTORE":
            # VSTORE C, V3
            dst = operands[0]  # array name
            src = [operands[1]]  # register
            self._validate_register(operands[1])
            
        elif opcode == "VLOAD_STRIDE":
            # VLOAD_STRIDE V1, A, stride
            dst = operands[0]
            src = [operands[1], operands[2]]
            self._validate_register(dst)
            # Kiểm tra stride là số
            try:
                int(operands[2])
            except ValueError:
                raise ParseError(f"Stride must be an integer: {operands[2]}")
            
        elif opcode == "VSCATTER":
            # VSCATTER C, IDX, V1
            dst = operands[0]  # array name
            src = [operands[1], operands[2]]  # idx_array, src_reg
            self._validate_register(operands[2])
            
        elif opcode in ["VADD", "VSUB", "VMUL", "VDIV", "VMAX", "VMIN", "VDOT", "VAND", "VOR", "VXOR"]:
            # VADD V3, V1, V2
            dst = operands[0]
            src = [operands[1], operands[2]]
            self._validate_register(dst)
            self._validate_register(operands[1])
            self._validate_register(operands[2])
            
        elif opcode in ["VADDS", "VSUBS", "VMULS", "VDIVS"]:
            # VADDS V3, V1, a
            dst = operands[0]
            src = [operands[1], operands[2]]
            self._validate_register(dst)
            self._validate_register(operands[1])
            # Toán hạng thứ 2 có thể là scalar hoặc số
            if not (operands[2] in self.scalars or self._is_number(operands[2])):
                raise ParseError(f"Expected scalar or number: {operands[2]}")
            
        elif opcode in ["VREDUCE_SUM", "VREDUCE_MAX", "VREDUCE_MIN"]:
            # VREDUCE_SUM S, V3
            dst = operands[0]
            src = [operands[1]]
            self._validate_register(operands[1])
            # Đích có thể là scalar register (S0, S1, ...)
            if not (dst.startswith('S') or dst in self.scalars):
                self.warnings.append(f"Reduction destination {dst} is not a scalar variable")
            
        elif opcode == "VMASK":
            # VMASK Vmask, V1, condition
            dst = operands[0]
            src = [operands[1], operands[2]]
            self._validate_register(dst)
            self._validate_register(operands[1])
            self._validate_condition(operands[2])

        elif opcode == "VMASK_COND":
            dst = operands[0]
            src = [operands[1], operands[2], operands[3]]
            self._validate_register(dst)
            self._validate_register(operands[1])
            self._validate_register(operands[2])
            self._validate_condition(operands[3])

        elif opcode in ["VADD_MASKED", "VSUB_MASKED"]:
            dst = operands[0]
            src = [operands[1], operands[2], operands[3]]
            self._validate_register(dst)
            self._validate_register(operands[1])
            self._validate_register(operands[2])
            self._validate_register(operands[3])

        elif opcode == "VMOV_MASKED":
            dst = operands[0]
            src = [operands[1], operands[2]]
            self._validate_register(dst)
            self._validate_register(operands[1])
            self._validate_register(operands[2])
        
        else:
            # Mặc định
            dst = operands[0]
            src = operands[1:] if len(operands) > 1 else []
            if dst.startswith('V'):
                self._validate_register(dst)
        
        # Tạo instruction object (gắn predicate per-lane nếu có '@Vx')
        inst = Instruction(
            opcode=opcode,
            dst=dst,
            src=src,
            raw_text=line,
            line_number=self.line_number,
            comment=comment,
            mask=mask_reg,
        )

        self.instructions.append(inst)
    
    def _validate_register(self, reg: str):
        """Kiểm tra tên thanh ghi hợp lệ"""
        if not re.match(r'^V\d+$', reg):
            raise ParseError(f"Invalid vector register: {reg}. Expected format: V0, V1, ...")

    def _validate_condition(self, cond: str):
        """Kiểm tra mã điều kiện mask."""
        if cond not in {"gt", "lt", "eq", "ne", "ge", "le"}:
            raise ParseError(
                f"Invalid condition: {cond}. Expected one of gt, lt, eq, ne, ge, le"
            )
    
    def _is_number(self, token: str) -> bool:
        """Kiểm tra token có phải số không"""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def _validate_program(self):
        """Kiểm tra tính hợp lệ của toàn bộ chương trình"""
        # Kiểm tra có instruction nào không
        if not self.instructions:
            self.warnings.append("No instructions found in program")
        
        # Kiểm tra các thanh ghi được sử dụng có tồn tại không
        all_registers = set()
        used_registers = set()
        
        for inst in self.instructions:
            if inst.dst_register:
                all_registers.add(inst.dst_register)
            for reg in inst.src_registers:
                used_registers.add(reg)
        
        # Cảnh báo nếu dùng register chưa được ghi
        undefined_regs = used_registers - all_registers
        if undefined_regs:
            for reg in undefined_regs:
                self.warnings.append(f"Register {reg} used before being written")
    
    def get_program_analyzer(self) -> ProgramAnalyzer:
        """Lấy ProgramAnalyzer cho chương trình đã parse"""
        return ProgramAnalyzer(self.instructions)
    
    def print_summary(self):
        """In tóm tắt chương trình"""
        print("=" * 60)
        print("PROGRAM SUMMARY")
        print("=" * 60)
        print(f"Data arrays: {list(self.data.keys())}")
        print(f"Scalars: {list(self.scalars.keys())}")
        print(f"Instructions: {len(self.instructions)}")
        
        # Thống kê lệnh theo opcode
        opcode_counts = {}
        for inst in self.instructions:
            opcode_counts[inst.opcode] = opcode_counts.get(inst.opcode, 0) + 1
        
        print("\nInstruction counts:")
        for opcode, count in sorted(opcode_counts.items()):
            print(f"  {opcode}: {count}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
    
    def export_to_json(self, filepath: str):
        """Export parsed data to JSON"""
        import json
        
        output = {
            "data": {k: v for k, v in self.data.items()},
            "scalars": self.scalars,
            "instructions": [inst.to_dict() for inst in self.instructions],
            "warnings": self.warnings
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


# ===== CÁC FILE MẪU MỚI =====

class ExamplePrograms:
    """Tạo các file chương trình mẫu"""
    
    @staticmethod
    def create_matrix_vector_file(filepath: str):
        """Tạo file matrix_vector.txt"""
        content = """
# Matrix-Vector Multiplication
# Ma trận A (4x4) nhân với vector X (4)
DATA A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
DATA X = [1, 2, 3, 4]
DATA Y = [0, 0, 0, 0]

# Load vector
VLOAD V1, A
VLOAD V2, X

# Compute each row (simplified - just element-wise multiply for demo)
VMUL V3, V1, V2

# Store result
VSTORE Y, V3
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    
    @staticmethod
    def create_image_filter_file(filepath: str):
        """Tạo file image_filter_1d.txt"""
        content = """
# 1D Image Filter
# Y[i] = a*X[i-1] + b*X[i] + c*X[i+1]
DATA X = [10, 20, 30, 40, 50, 60, 70, 80]
SCALAR a = 1
SCALAR b = 2
SCALAR c = 1

# Load original image
VLOAD V1, X

# For simplicity, just do a weighted sum
VMULS V2, V1, b

# Store filtered image
VSTORE Y, V2
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    
    @staticmethod
    def create_stride_access_file(filepath: str):
        """Tạo file stride_access.txt"""
        content = """
# Stride Access Example
DATA A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Load with stride 2 (only even indices)
VLOAD_STRIDE V1, A, 2

# Double the values
VADDS V2, V1, 0

# Store result
VSTORE B, V2
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    
    @staticmethod
    def create_all_examples(output_dir: str = "examples"):
        """Tạo tất cả các file mẫu"""
        from pathlib import Path
        Path(output_dir).mkdir(exist_ok=True)
        
        ExamplePrograms.create_matrix_vector_file(f"{output_dir}/matrix_vector.txt")
        ExamplePrograms.create_image_filter_file(f"{output_dir}/image_filter_1d.txt")
        ExamplePrograms.create_stride_access_file(f"{output_dir}/stride_access.txt")
        
        print(f"✓ Created example programs in {output_dir}/")


# ===== TEST NHANH =====
if __name__ == "__main__":
    # Tạo file mẫu
    ExamplePrograms.create_all_examples()
    
    # Test parser
    parser = ProgramParser()
    
    # Test với vector_add.txt
    print("\n" + "=" * 60)
    print("TEST 1: vector_add.txt")
    print("=" * 60)
    try:
        data, scalars, instructions = parser.parse_file("examples/vector_add.txt")
        parser.print_summary()
        
        # Test ProgramAnalyzer
        analyzer = parser.get_program_analyzer()
        print(f"\nRAW Dependencies: {analyzer.find_dependencies()}")
        
    except ParseError as e:
        print(f"Error: {e}")
    
    # Test với matrix_vector.txt
    print("\n" + "=" * 60)
    print("TEST 2: matrix_vector.txt")
    print("=" * 60)
    try:
        data, scalars, instructions = parser.parse_file("examples/matrix_vector.txt")
        parser.print_summary()
    except ParseError as e:
        print(f"Error: {e}")
    
    # Test với stride_access.txt
    print("\n" + "=" * 60)
    print("TEST 3: stride_access.txt")
    print("=" * 60)
    try:
        data, scalars, instructions = parser.parse_file("examples/stride_access.txt")
        parser.print_summary()
        
        # Export to JSON
        parser.export_to_json("outputs/program_analysis.json")
        print("\n✓ Exported to outputs/program_analysis.json")
        
    except ParseError as e:
        print(f"Error: {e}")
