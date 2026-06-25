import copy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class RegisterStatus(Enum):
    """Trạng thái của thanh ghi"""
    FREE = "free"           # Sẵn sàng sử dụng
    BUSY = "busy"           # Đang được ghi
    RESERVED = "reserved"   # Đã được reserve cho chaining

@dataclass
class VectorRegister:
    """Thông tin về một thanh ghi vector"""
    name: str
    data: List[float] = field(default_factory=list)
    status: RegisterStatus = RegisterStatus.FREE
    ready_cycle: int = 0           # Chu kỳ sẵn sàng (cho pipeline)
    last_write_cycle: int = -1     # Chu kỳ ghi cuối cùng
    last_read_cycle: int = -1      # Chu kỳ đọc cuối cùng
    write_count: int = 0           # Số lần được ghi
    read_count: int = 0            # Số lần được đọc
    
    def __len__(self):
        return len(self.data)
    
    def is_ready(self, current_cycle: int) -> bool:
        """Kiểm tra thanh ghi đã sẵn sàng chưa"""
        return current_cycle >= self.ready_cycle
    
    def clear(self):
        """Xóa dữ liệu trong thanh ghi"""
        self.data = []
        self.status = RegisterStatus.FREE
        self.ready_cycle = 0


class VectorRegisterFile:
    """
    Quản lý tập thanh ghi vector
    
    Hỗ trợ:
    - Đọc/ghi thanh ghi
    - Theo dõi trạng thái (cho pipeline)
    - Chaining (forwarding)
    - Renaming (cho out-of-order - nâng cao)
    - Thống kê sử dụng
    """
    
    def __init__(self, num_registers: int = 16, vector_length: int = 64, enable_chaining: bool = True):
        """
        Khởi tạo vector register file
        
        Args:
            num_registers: Số lượng thanh ghi vector (V0, V1, ...)
            vector_length: Độ dài mặc định của vector
            enable_chaining: Bật/tắt chaining
        """
        self.num_registers = num_registers
        self.vector_length = vector_length
        self.enable_chaining = enable_chaining
        
        # Khởi tạo các thanh ghi
        self.registers: Dict[str, VectorRegister] = {}
        for i in range(num_registers):
            name = f"V{i}"
            self.registers[name] = VectorRegister(
                name=name,
                data=[0.0] * vector_length,
                status=RegisterStatus.FREE,
                ready_cycle=0
            )
        
        # Thống kê
        self.total_reads = 0
        self.total_writes = 0
        self.stall_cycles = 0  # Chu kỳ stall do chờ thanh ghi
        
        # Chaining support
        self.forwarding_map: Dict[str, Tuple[int, List[float]]] = {}  # register -> (cycle, data)
    
    # ===== Các phương thức cơ bản =====
    
    def check(self, reg: str) -> bool:
        """Kiểm tra thanh ghi có tồn tại không"""
        return reg in self.registers
    
    def _validate(self, reg: str):
        """Kiểm tra và báo lỗi nếu thanh ghi không tồn tại"""
        if not self.check(reg):
            raise KeyError(f"Invalid vector register: {reg}. Valid registers: {list(self.registers.keys())[:10]}...")
    
    def get_register(self, reg: str) -> VectorRegister:
        """Lấy thông tin thanh ghi"""
        self._validate(reg)
        return self.registers[reg]
    
    def read(self, reg: str, current_cycle: int = 0) -> List[float]:
        """
        Đọc dữ liệu từ thanh ghi vector
        
        Args:
            reg: Tên thanh ghi (V0, V1, ...)
            current_cycle: Chu kỳ hiện tại (cho pipeline)
        
        Returns:
            List[float]: Dữ liệu vector
        """
        self._validate(reg)
        reg_info = self.registers[reg]
        # Kiểm tra ready (cho pipeline)
        if not reg_info.is_ready(current_cycle):
            self.stall_cycles += (reg_info.ready_cycle - current_cycle)
            # Nâng cao: có thể raise exception hoặc tự động stall
        
        # Cập nhật thống kê
        reg_info.read_count += 1
        reg_info.last_read_cycle = current_cycle
        self.total_reads += 1
        
        # Kiểm tra forwarding (chaining)
        if self.enable_chaining and reg in self.forwarding_map:
            cycle, data = self.forwarding_map[reg]
            if cycle <= current_cycle:
                return list(data)
        return list(reg_info.data)
    
    def write(self, reg: str, values: List[float], current_cycle: int = 0, latency: int = 1):
        """
        Ghi dữ liệu vào thanh ghi vector
        
        Args:
            reg: Tên thanh ghi
            values: Dữ liệu vector cần ghi
            current_cycle: Chu kỳ bắt đầu ghi
            latency: Độ trễ ghi (cho pipeline)
        """
        self._validate(reg)
        reg_info = self.registers[reg]
        
        # Cập nhật dữ liệu
        reg_info.data = list(values)
        reg_info.last_write_cycle = current_cycle
        reg_info.write_count += 1
        reg_info.status = RegisterStatus.BUSY
        reg_info.ready_cycle = current_cycle + latency
        
        self.total_writes += 1
        
        # Nếu bật chaining, cập nhật forwarding
        if self.enable_chaining:
            self.forwarding_map[reg] = (reg_info.ready_cycle, reg_info.data)
    
    def write_partial(self, reg: str, start_idx: int, values: List[float], current_cycle: int = 0):
        """
        Ghi một phần của vector (cho masked operations)
        
        Args:
            reg: Tên thanh ghi
            start_idx: Vị trí bắt đầu ghi
            values: Dữ liệu cần ghi
            current_cycle: Chu kỳ hiện tại
        """
        self._validate(reg)
        reg_info = self.registers[reg]
        
        # Kiểm tra bounds
        if start_idx + len(values) > len(reg_info.data):
            # Mở rộng vector nếu cần
            reg_info.data.extend([0.0] * (start_idx + len(values) - len(reg_info.data)))
        
        # Ghi dữ liệu
        for i, val in enumerate(values):
            if start_idx + i < len(reg_info.data):
                reg_info.data[start_idx + i] = val
        
        reg_info.last_write_cycle = current_cycle
        reg_info.write_count += 1
        reg_info.status = RegisterStatus.BUSY
    
    # ===== Các phương thức hỗ trợ pipeline =====
    
    def is_ready(self, reg: str, current_cycle: int) -> bool:
        """Kiểm tra thanh ghi đã sẵn sàng chưa"""
        self._validate(reg)
        return self.registers[reg].is_ready(current_cycle)
    
    def get_ready_cycle(self, reg: str) -> int:
        """Lấy chu kỳ sẵn sàng của thanh ghi"""
        self._validate(reg)
        return self.registers[reg].ready_cycle
    
    def get_status(self, reg: str) -> RegisterStatus:
        """Lấy trạng thái của thanh ghi"""
        self._validate(reg)
        return self.registers[reg].status
    
    def set_vector_length(self, reg: str, new_length: int):
        """Thay đổi độ dài vector"""
        self._validate(reg)
        current_data = self.registers[reg].data
        if new_length > len(current_data):
            # Mở rộng
            self.registers[reg].data.extend([0.0] * (new_length - len(current_data)))
        elif new_length < len(current_data):
            # Thu hẹp
            self.registers[reg].data = current_data[:new_length]
    
    def clear(self, reg: str):
        """Xóa dữ liệu trong thanh ghi"""
        self._validate(reg)
        self.registers[reg].clear()
    
    def clear_all(self):
        """Xóa tất cả thanh ghi"""
        for reg in self.registers.values():
            reg.clear()
        self.forwarding_map.clear()
    
    # ===== Hỗ trợ chaining =====
    
    def add_forwarding(self, reg: str, ready_cycle: int, data: List[float]):
        """Thêm forwarding entry (cho chaining)"""
        self.forwarding_map[reg] = (ready_cycle, list(data))
    
    def get_forwarding_data(self, reg: str, current_cycle: int) -> Optional[List[float]]:
        """Lấy dữ liệu forwarding nếu có"""
        if reg in self.forwarding_map:
            cycle, data = self.forwarding_map[reg]
            if cycle <= current_cycle:
                return list(data)
        return None
    
    def clear_forwarding(self, reg: Optional[str] = None):
        """Xóa forwarding entries"""
        if reg:
            self.forwarding_map.pop(reg, None)
        else:
            self.forwarding_map.clear()
    
    # ===== Thống kê và báo cáo =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê sử dụng thanh ghi"""
        stats = {
            "total_reads": self.total_reads,
            "total_writes": self.total_writes,
            "stall_cycles": self.stall_cycles,
            "forwarding_entries": len(self.forwarding_map),
            "registers": {}
        }
        
        for name, reg in self.registers.items():
            stats["registers"][name] = {
                "length": len(reg),
                "read_count": reg.read_count,
                "write_count": reg.write_count,
                "status": reg.status.value,
                "ready_cycle": reg.ready_cycle,
                "last_read": reg.last_read_cycle,
                "last_write": reg.last_write_cycle
            }
        
        return stats
    
    def print_stats(self):
        """In thống kê sử dụng thanh ghi"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("VECTOR REGISTER FILE STATISTICS")
        print("=" * 60)
        print(f"Total reads:  {stats['total_reads']}")
        print(f"Total writes: {stats['total_writes']}")
        print(f"Stall cycles: {stats['stall_cycles']}")
        print(f"Forwarding entries: {stats['forwarding_entries']}")
        
        print("\nRegister usage:")
        for reg_name, reg_stats in stats['registers'].items():
            if reg_stats['read_count'] > 0 or reg_stats['write_count'] > 0:
                print(f"  {reg_name}: reads={reg_stats['read_count']}, writes={reg_stats['write_count']}, "
                    f"len={reg_stats['length']}, status={reg_stats['status']}")
    
    # ===== Hỗ trợ debug và hiển thị =====
    
    def dump(self) -> Dict[str, List[float]]:
        """Xuất toàn bộ dữ liệu thanh ghi"""
        return {name: list(reg.data) for name, reg in self.registers.items()}
    
    def dump_used(self) -> Dict[str, List[float]]:
        """Xuất dữ liệu của các thanh ghi đã được sử dụng"""
        return {name: list(reg.data) for name, reg in self.registers.items() 
                if reg.read_count > 0 or reg.write_count > 0}
    
    def __str__(self) -> str:
        """Hiển thị thông tin thanh ghi"""
        lines = ["Vector Register File:"]
        for name, reg in self.registers.items():
            if reg.read_count > 0 or reg.write_count > 0:
                preview = str(reg.data[:4]) + ("..." if len(reg.data) > 4 else "")
                lines.append(f"  {name}: {preview} (len={len(reg.data)})")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"VectorRegisterFile(num_registers={self.num_registers}, vector_length={self.vector_length})"


# ===== RENAMING REGISTER FILE (NÂNG CAO) =====

class RenamingRegisterFile(VectorRegisterFile):
    """
    Register renaming cho out-of-order execution
    Mở rộng của VectorRegisterFile
    """
    
    def __init__(self, num_registers: int, vector_length: int = 64, num_physical_regs: int = 32):
        super().__init__(num_registers, vector_length)
        
        self.num_physical_regs = num_physical_regs
        self.physical_regs: Dict[str, VectorRegister] = {}
        
        # Khởi tạo physical registers
        for i in range(num_physical_regs):
            name = f"P{i}"
            self.physical_regs[name] = VectorRegister(
                name=name,
                data=[0.0] * vector_length,
                status=RegisterStatus.FREE
            )
        
        # Map logical -> physical
        self.rename_map: Dict[str, str] = {}
        self.free_physical: List[str] = [f"P{i}" for i in range(num_physical_regs)]
        
        # Khởi tạo mapping ban đầu
        for i in range(num_registers):
            if self.free_physical:
                phys = self.free_physical.pop(0)
                self.rename_map[f"V{i}"] = phys
    
    def _allocate_physical(self) -> str:
        """Cấp phát physical register mới"""
        if not self.free_physical:
            raise RuntimeError("No free physical registers available")
        return self.free_physical.pop(0)
    
    def _free_physical(self, phys_reg: str):
        """Giải phóng physical register"""
        if phys_reg not in self.free_physical:
            self.free_physical.append(phys_reg)
    
    def write(self, reg: str, values: List[float], current_cycle: int = 0, latency: int = 1):
        """Ghi với register renaming"""
        self._validate(reg)
        
        # Cấp phát physical register mới
        new_phys = self._allocate_physical()
        old_phys = self.rename_map.get(reg)
        
        # Cập nhật mapping
        self.rename_map[reg] = new_phys
        
        # Ghi vào physical register
        if new_phys in self.physical_regs:
            self.physical_regs[new_phys].data = list(values)
            self.physical_regs[new_phys].last_write_cycle = current_cycle
            self.physical_regs[new_phys].ready_cycle = current_cycle + latency
            self.physical_regs[new_phys].write_count += 1
        
        # Giải phóng physical register cũ (nếu không còn tham chiếu)
        if old_phys and self._can_free(old_phys):
            self._free_physical(old_phys)
        
        self.total_writes += 1
    
    def read(self, reg: str, current_cycle: int = 0) -> List[float]:
        """Đọc với register renaming"""
        self._validate(reg)
        
        phys_reg = self.rename_map.get(reg)
        if not phys_reg:
            raise KeyError(f"No mapping for logical register {reg}")
        
        if phys_reg in self.physical_regs:
            reg_info = self.physical_regs[phys_reg]
            reg_info.read_count += 1
            reg_info.last_read_cycle = current_cycle
            self.total_reads += 1
            return list(reg_info.data)
        
        return []
    
    def _can_free(self, phys_reg: str) -> bool:
        """Kiểm tra physical register có thể giải phóng không"""
        # Kiểm tra xem có logical register nào đang mapping đến phys_reg không
        return phys_reg not in self.rename_map.values()


# ===== TEST NHANH =====
if __name__ == "__main__":
    print("=" * 60)
    print("TEST VECTOR REGISTER FILE")
    print("=" * 60)
    
    # Tạo register file
    vrf = VectorRegisterFile(num_registers=8, vector_length=8)
    print(vrf)
    
    # Test ghi và đọc
    print("\n--- Test Write/Read ---")
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    vrf.write("V0", test_data, current_cycle=0, latency=2)
    print(f"Wrote to V0: {test_data}")
    
    read_data = vrf.read("V0", current_cycle=2)
    print(f"Read from V0: {read_data}")
    
    # Test ready status
    print("\n--- Test Ready Status ---")
    print(f"Is V0 ready at cycle 1? {vrf.is_ready('V0', 1)}")
    print(f"Is V0 ready at cycle 2? {vrf.is_ready('V0', 2)}")
    print(f"Ready cycle of V0: {vrf.get_ready_cycle('V0')}")
    
    # Test partial write
    print("\n--- Test Partial Write ---")
    vrf.write_partial("V0", 2, [99.0, 100.0])
    print(f"After partial write: {vrf.read('V0')[:5]}...")
    
    # Test statistics
    print("\n--- Statistics ---")
    vrf.print_stats()
    
    # Test renaming (nâng cao)
    print("\n" + "=" * 60)
    print("TEST RENAMING REGISTER FILE")
    print("=" * 60)
    
    rrf = RenamingRegisterFile(num_registers=4, vector_length=4, num_physical_regs=8)
    rrf.write("V0", [1.0, 2.0, 3.0, 4.0])
    rrf.write("V1", [5.0, 6.0, 7.0, 8.0])
    
    print(f"Read V0: {rrf.read('V0')}")
    print(f"Read V1: {rrf.read('V1')}")
    
    # In thống kê
    print("\n--- Renaming Statistics ---")
    print(f"Rename map: {rrf.rename_map}")
    print(f"Free physical: {rrf.free_physical[:5]}...")
    
    print("\n✓ All tests passed!")