from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum

class InstructionType(Enum):
    """Phân loại loại lệnh"""
    MEMORY_LOAD = "memory_load"
    MEMORY_STORE = "memory_store"
    MEMORY_STRIDE = "memory_stride"
    MEMORY_GATHER = "memory_gather"
    MEMORY_SCATTER = "memory_scatter"
    ARITHMETIC_BINARY = "arithmetic_binary"
    ARITHMETIC_SCALAR = "arithmetic_scalar"
    ARITHMETIC_REDUCTION = "arithmetic_reduction"
    ARITHMETIC_DOT = "arithmetic_dot"
    LOGICAL = "logical"
    MASK = "mask"
    CONFIG = "config"

class FunctionalUnit(Enum):
    """Đơn vị chức năng thực thi lệnh"""
    MEMORY = "Memory"
    ALU = "Vector ALU"
    REDUCTION = "Reduction Unit"
    MASK = "Mask Unit"

@dataclass
class Instruction:
    """
    Định nghĩa cấu trúc lệnh vector

    Attributes:
        opcode: Mã lệnh (VLOAD, VADD, VMUL, ...)
        dst: Thanh ghi đích
        src: Danh sách toán hạng nguồn
        raw_text: Câu lệnh gốc từ file
        line_number: Số thứ tự dòng (để debug)
        comment: Chú thích (nếu có)
        mask: Thanh ghi predicate per-lane (vd "V9"). None = không mask
               (mọi lane active). Đây là phần MỞ RỘNG cho "true masked SIMD":
               mỗi lệnh vector có thể mang theo một mask register, simulator
               truy cập trạng thái mask này tại runtime để quyết định lane nào
               được thực thi (active) và lane nào bị skip (inactive).
        mask_mode: Chính sách cho lane inactive ở giai đoạn writeback:
               "merge"/"undisturbed" = giữ nguyên giá trị cũ của thanh ghi đích
               (chuẩn RVV/AVX-512 masking). "zero" = ghi 0 cho lane inactive.
    """
    opcode: str
    dst: str
    src: List[str]
    raw_text: str
    line_number: int = 0
    comment: str = ""
    mask: Optional[str] = None
    mask_mode: str = "merge"

    @property
    def inst_type(self) -> InstructionType:
        """Xác định loại lệnh"""
        if self.opcode in ["VLOAD"]:
            return InstructionType.MEMORY_LOAD
        elif self.opcode in ["VSTORE"]:
            return InstructionType.MEMORY_STORE
        elif self.opcode in ["VLOAD_STRIDE"]:
            return InstructionType.MEMORY_STRIDE
        elif self.opcode in ["VGATHER"]:
            return InstructionType.MEMORY_GATHER
        elif self.opcode in ["VSCATTER"]:
            return InstructionType.MEMORY_SCATTER
        elif self.opcode in ["VADD", "VSUB", "VMUL", "VDIV", "VMAX", "VMIN"]:
            return InstructionType.ARITHMETIC_BINARY
        elif self.opcode in ["VADDS", "VSUBS", "VMULS", "VDIVS"]:
            return InstructionType.ARITHMETIC_SCALAR
        elif self.opcode in ["VREDUCE_SUM", "VREDUCE_MAX", "VREDUCE_MIN"]:
            return InstructionType.ARITHMETIC_REDUCTION
        elif self.opcode in ["VDOT"]:
            return InstructionType.ARITHMETIC_DOT
        elif self.opcode in ["VADD_MASKED", "VSUB_MASKED"]:
            # Lệnh số học có mask cố định trong toán hạng — về bản chất là ALU
            return InstructionType.ARITHMETIC_BINARY
        elif self.opcode in ["VAND", "VOR", "VXOR"]:
            return InstructionType.LOGICAL
        elif self.opcode in ["VMASK", "VMASK_COND", "VMOV_MASKED"]:
            return InstructionType.MASK
        else:
            return InstructionType.CONFIG
    
    @property
    def functional_unit(self) -> FunctionalUnit:
        """Xác định đơn vị chức năng thực thi lệnh"""
        if self.inst_type in [
            InstructionType.MEMORY_LOAD,
            InstructionType.MEMORY_STORE,
            InstructionType.MEMORY_STRIDE,
            InstructionType.MEMORY_GATHER,
            InstructionType.MEMORY_SCATTER
        ]:
            return FunctionalUnit.MEMORY
        elif self.inst_type in [
            InstructionType.ARITHMETIC_BINARY,
            InstructionType.ARITHMETIC_SCALAR,
            InstructionType.LOGICAL
        ]:
            return FunctionalUnit.ALU
        elif self.inst_type in [
            InstructionType.ARITHMETIC_REDUCTION,
            InstructionType.ARITHMETIC_DOT
        ]:
            return FunctionalUnit.REDUCTION
        else:
            return FunctionalUnit.ALU
    
    @property
    def src_registers(self) -> List[str]:
        """Lấy danh sách các thanh ghi vector nguồn"""
        return [s for s in self.src if s.startswith('V')]
    
    @property
    def dst_register(self) -> Optional[str]:
        """Lấy thanh ghi đích (nếu có)"""
        return self.dst if self.dst.startswith('V') else None
    
    @property
    def scalar_operands(self) -> List[str]:
        """Lấy danh sách toán hạng scalar (hằng số hoặc biến)"""
        return [s for s in self.src if not s.startswith('V') and s not in ['A', 'B', 'C', 'X', 'Y']]
    
    @property
    def memory_operands(self) -> List[str]:
        """Lấy danh sách tên mảng trong memory"""
        if self.inst_type in [InstructionType.MEMORY_LOAD, InstructionType.MEMORY_STRIDE, InstructionType.MEMORY_GATHER]:
            return [self.src[0]]
        elif self.inst_type in [InstructionType.MEMORY_STORE, InstructionType.MEMORY_SCATTER]:
            return [self.dst]
        return []
    
    @property
    def has_dependency(self) -> bool:
        """Kiểm tra lệnh có phụ thuộc dữ liệu không (có src register)"""
        return len(self.src_registers) > 0
    
    @property
    def is_memory_instruction(self) -> bool:
        """Kiểm tra có phải lệnh truy xuất bộ nhớ không"""
        return self.inst_type in [
            InstructionType.MEMORY_LOAD,
            InstructionType.MEMORY_STORE,
            InstructionType.MEMORY_STRIDE,
            InstructionType.MEMORY_GATHER,
            InstructionType.MEMORY_SCATTER
        ]
    
    @property
    def is_arithmetic(self) -> bool:
        """Kiểm tra có phải lệnh số học không"""
        return self.inst_type in [
            InstructionType.ARITHMETIC_BINARY,
            InstructionType.ARITHMETIC_SCALAR,
            InstructionType.ARITHMETIC_REDUCTION,
            InstructionType.ARITHMETIC_DOT
        ]
    
    @property
    def is_masked(self) -> bool:
        """Lệnh có mang predicate per-lane không (true masked SIMD)."""
        return self.mask is not None

    @property
    def is_legacy_masked_opcode(self) -> bool:
        """Các opcode masked 'tường minh' nhúng mask trong toán hạng nguồn
        (VADD_MASKED/VSUB_MASKED có mask = src cuối; VMOV_MASKED tương tự).
        Giữ nguyên để tương thích ngược với ISA hiện có."""
        return self.opcode in ["VADD_MASKED", "VSUB_MASKED", "VMOV_MASKED"]

    def with_mask(self, mask_reg: Optional[str], mask_mode: str = "merge") -> "Instruction":
        """Gắn predicate per-lane cho lệnh (trả về chính nó để tiện chaining)."""
        self.mask = mask_reg
        self.mask_mode = mask_mode
        return self

    def get_dependency_registers(self) -> Set[str]:
        """Lấy tập hợp các thanh ghi gây phụ thuộc (RAW).

        Bao gồm cả mask register (nếu có) vì mask cũng là một nguồn dữ liệu
        mà lệnh phải đọc trước khi thực thi → tạo RAW dependency thực sự."""
        regs = set(self.src_registers)
        if self.mask is not None and self.mask.startswith('V'):
            regs.add(self.mask)
        return regs
    
    def __str__(self):
        """Hiển thị lệnh dạng đẹp"""
        base = f"{self.opcode} {self.dst}"
        if self.src:
            base += ", " + ", ".join(self.src)
        if self.mask is not None:
            base += f" @{self.mask}"
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển lệnh thành dictionary (dùng cho JSON export)"""
        return {
            "opcode": self.opcode,
            "dst": self.dst,
            "src": self.src,
            "raw_text": self.raw_text,
            "line_number": self.line_number,
            "type": self.inst_type.value,
            "functional_unit": self.functional_unit.value,
            "src_registers": self.src_registers,
            "dst_register": self.dst_register,
            "is_memory": self.is_memory_instruction,
            "mask": self.mask,
            "mask_mode": self.mask_mode,
        }


# ===== CÁC LỆNH MẪU (DÙNG CHO TEST) =====

class InstructionTemplates:
    """Các mẫu lệnh thường dùng"""
    
    @staticmethod
    def vload(dst: str, array: str) -> Instruction:
        """Tạo lệnh VLOAD"""
        return Instruction(
            opcode="VLOAD",
            dst=dst,
            src=[array],
            raw_text=f"VLOAD {dst}, {array}"
        )
    
    @staticmethod
    def vstore(array: str, src: str) -> Instruction:
        """Tạo lệnh VSTORE"""
        return Instruction(
            opcode="VSTORE",
            dst=array,
            src=[src],
            raw_text=f"VSTORE {array}, {src}"
        )
    
    @staticmethod
    def vadd(dst: str, src1: str, src2: str) -> Instruction:
        """Tạo lệnh VADD"""
        return Instruction(
            opcode="VADD",
            dst=dst,
            src=[src1, src2],
            raw_text=f"VADD {dst}, {src1}, {src2}"
        )
    
    @staticmethod
    def vsub(dst: str, src1: str, src2: str) -> Instruction:
        """Tạo lệnh VSUB"""
        return Instruction(
            opcode="VSUB",
            dst=dst,
            src=[src1, src2],
            raw_text=f"VSUB {dst}, {src1}, {src2}"
        )
    
    @staticmethod
    def vmul(dst: str, src1: str, src2: str) -> Instruction:
        """Tạo lệnh VMUL"""
        return Instruction(
            opcode="VMUL",
            dst=dst,
            src=[src1, src2],
            raw_text=f"VMUL {dst}, {src1}, {src2}"
        )
    
    @staticmethod
    def vadds(dst: str, src: str, scalar: str) -> Instruction:
        """Tạo lệnh VADDS (vector + scalar)"""
        return Instruction(
            opcode="VADDS",
            dst=dst,
            src=[src, scalar],
            raw_text=f"VADDS {dst}, {src}, {scalar}"
        )
    
    @staticmethod
    def vmuls(dst: str, src: str, scalar: str) -> Instruction:
        """Tạo lệnh VMULS (vector * scalar)"""
        return Instruction(
            opcode="VMULS",
            dst=dst,
            src=[src, scalar],
            raw_text=f"VMULS {dst}, {src}, {scalar}"
        )
    
    @staticmethod
    def vreduce_sum(dst: str, src: str) -> Instruction:
        """Tạo lệnh VREDUCE_SUM"""
        return Instruction(
            opcode="VREDUCE_SUM",
            dst=dst,
            src=[src],
            raw_text=f"VREDUCE_SUM {dst}, {src}"
        )
    
    @staticmethod
    def vdot(dst: str, src1: str, src2: str) -> Instruction:
        """Tạo lệnh VDOT"""
        return Instruction(
            opcode="VDOT",
            dst=dst,
            src=[src1, src2],
            raw_text=f"VDOT {dst}, {src1}, {src2}"
        )
    
    @staticmethod
    def vload_stride(dst: str, array: str, stride: int) -> Instruction:
        """Tạo lệnh VLOAD_STRIDE"""
        return Instruction(
            opcode="VLOAD_STRIDE",
            dst=dst,
            src=[array, str(stride)],
            raw_text=f"VLOAD_STRIDE {dst}, {array}, {stride}"
        )
    
    @staticmethod
    def vgather(dst: str, array: str, idx_array: str) -> Instruction:
        """Tạo lệnh VGATHER"""
        return Instruction(
            opcode="VGATHER",
            dst=dst,
            src=[array, idx_array],
            raw_text=f"VGATHER {dst}, {array}, {idx_array}"
        )
    
    @staticmethod
    def vscatter(array: str, idx_array: str, src: str) -> Instruction:
        """Tạo lệnh VSCATTER"""
        return Instruction(
            opcode="VSCATTER",
            dst=array,
            src=[idx_array, src],
            raw_text=f"VSCATTER {array}, {idx_array}, {src}"
        )

    @staticmethod
    def vmask(dst: str, src: str, condition: str) -> Instruction:
        """Tạo lệnh VMASK (sinh predicate từ so sánh với mean của vector)"""
        return Instruction(
            opcode="VMASK",
            dst=dst,
            src=[src, condition],
            raw_text=f"VMASK {dst}, {src}, {condition}"
        )

    @staticmethod
    def masked(inst: Instruction, mask_reg: str, mask_mode: str = "merge") -> Instruction:
        """Gắn predicate per-lane vào một lệnh đã tạo sẵn (true masked SIMD).

        Ví dụ:
            InstructionTemplates.masked(
                InstructionTemplates.vadd("V3", "V1", "V2"), "V9")
        → VADD V3, V1, V2 @V9  (chỉ các lane mask=1 mới được cộng & ghi lại).
        """
        inst.mask = mask_reg
        inst.mask_mode = mask_mode
        if mask_reg is not None and f"@{mask_reg}" not in inst.raw_text:
            inst.raw_text = f"{inst.raw_text} @{mask_reg}"
        return inst


# ===== PHÂN TÍCH CHƯƠNG TRÌNH =====

class ProgramAnalyzer:
    """Phân tích chương trình vector instructions"""
    
    def __init__(self, instructions: List[Instruction]):
        self.instructions = instructions
    
    def get_instruction_counts(self) -> Dict[str, int]:
        """Đếm số lượng từng loại lệnh"""
        counts = {}
        for inst in self.instructions:
            counts[inst.opcode] = counts.get(inst.opcode, 0) + 1
        return counts
    
    def get_register_usage(self) -> Dict[str, int]:
        """Phân tích tần suất sử dụng thanh ghi"""
        usage = {}
        for inst in self.instructions:
            # Thanh ghi đích
            if inst.dst_register:
                usage[inst.dst_register] = usage.get(inst.dst_register, 0) + 1
            # Thanh ghi nguồn
            for reg in inst.src_registers:
                usage[reg] = usage.get(reg, 0) + 1
        return usage
    
    def find_dependencies(self) -> List[tuple]:
        """
        Tìm các phụ thuộc dữ liệu giữa các lệnh
        Returns: List of (src_inst_index, dst_inst_index, register)
        """
        dependencies = []
        last_write = {}  # register -> last instruction index that wrote to it
        
        for i, inst in enumerate(self.instructions):
            # Kiểm tra RAW dependency: read after write
            for reg in inst.src_registers:
                if reg in last_write:
                    dependencies.append((last_write[reg], i, reg))
            
            # Update last write
            if inst.dst_register:
                last_write[inst.dst_register] = i
        
        return dependencies
    
    def get_memory_access_pattern(self) -> Dict[str, List[str]]:
        """Phân tích pattern truy xuất bộ nhớ"""
        loads = []
        stores = []
        
        for inst in self.instructions:
            if inst.inst_type == InstructionType.MEMORY_LOAD:
                loads.extend(inst.memory_operands)
            elif inst.inst_type == InstructionType.MEMORY_STORE:
                stores.extend(inst.memory_operands)
        
        return {"loads": loads, "stores": stores}
    
    def estimate_vector_length(self) -> int:
        """Ước lượng vector length từ chương trình (dùng cho scalar simulator)"""
        # Dựa vào số lượng lệnh VLOAD đầu tiên
        for inst in self.instructions:
            if inst.opcode == "VLOAD":
                # Giả sử độ dài vector từ dữ liệu sẽ được xác định sau
                return 0  # Return 0 để báo chưa xác định
        return 8  # Default
    
    def summary(self) -> Dict[str, Any]:
        """Tóm tắt chương trình"""
        return {
            "total_instructions": len(self.instructions),
            "instruction_counts": self.get_instruction_counts(),
            "register_usage": self.get_register_usage(),
            "dependencies": len(self.find_dependencies()),
            "memory_accesses": self.get_memory_access_pattern()
        }


# ===== TEST NHANH =====
if __name__ == "__main__":
    # Tạo một số lệnh mẫu
    instructions = [
        InstructionTemplates.vload("V1", "A"),
        InstructionTemplates.vload("V2", "B"),
        InstructionTemplates.vadd("V3", "V1", "V2"),
        InstructionTemplates.vstore("C", "V3"),
        InstructionTemplates.vmul("V4", "V1", "V2"),
        InstructionTemplates.vadds("V5", "V4", "a"),
        InstructionTemplates.vreduce_sum("S", "V5"),
    ]
    
    # Thêm line number cho từng lệnh
    for i, inst in enumerate(instructions, 1):
        inst.line_number = i
    
    print("=" * 60)
    print("INSTRUCTION LIST")
    print("=" * 60)
    for inst in instructions:
        print(f"  {inst}")
        print(f"    Type: {inst.inst_type.value}")
        print(f"    Unit: {inst.functional_unit.value}")
        print(f"    Src registers: {inst.src_registers}")
        print(f"    Dst register: {inst.dst_register}")
        print()
    
    # Phân tích chương trình
    analyzer = ProgramAnalyzer(instructions)
    print("\n" + "=" * 60)
    print("PROGRAM ANALYSIS")
    print("=" * 60)
    print(f"Total instructions: {analyzer.summary()['total_instructions']}")
    print(f"Instruction counts: {analyzer.get_instruction_counts()}")
    print(f"Register usage: {analyzer.get_register_usage()}")
    print(f"RAW dependencies: {analyzer.find_dependencies()}")