import math
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from instruction import Instruction, InstructionType, ProgramAnalyzer

@dataclass
class ScalarExecutionStats:
    """Thống kê thực thi scalar"""
    total_cycles: int = 0
    total_instructions: int = 0
    memory_instructions: int = 0
    arithmetic_instructions: int = 0
    reduction_instructions: int = 0
    total_elements_processed: int = 0
    breakdown_by_opcode: Dict[str, int] = field(default_factory=dict)


class ScalarSimulator:
    """
    Mô phỏng xử lý scalar (tuần tự từng phần tử)
    
    Dùng làm baseline để so sánh với vector processor.
    Mỗi phần tử được xử lý riêng rẽ, không có parallel execution.
    """
    
    def __init__(self, config):
        """
        Khởi tạo scalar simulator
        
        Args:
            config: VectorProcessorConfig object
        """
        self.config = config
        self.stats = ScalarExecutionStats()
        
        # Định nghĩa latency cho từng loại lệnh
        self.latency_map = {
            # Memory instructions
            "VLOAD": config.load_latency,
            "VSTORE": config.store_latency,
            "VLOAD_STRIDE": config.load_latency,
            "VGATHER": config.load_latency * 2,  # Gather chậm hơn
            "VSCATTER": config.store_latency * 2,
            
            # Arithmetic binary
            "VADD": config.add_latency,
            "VSUB": config.add_latency,
            "VMUL": config.mul_latency,
            "VDIV": config.div_latency,
            "VMAX": config.add_latency,
            "VMIN": config.add_latency,
            
            # Arithmetic scalar
            "VADDS": config.add_latency,
            "VSUBS": config.add_latency,
            "VMULS": config.mul_latency,
            "VDIVS": config.div_latency,
            
            # Reduction
            "VREDUCE_SUM": config.reduction_latency,
            "VREDUCE_MAX": config.reduction_latency,
            "VREDUCE_MIN": config.reduction_latency,
            
            # Dot product
            "VDOT": config.mul_latency + config.add_latency,
            
            # Logical
            "VAND": config.add_latency,
            "VOR": config.add_latency,
            "VXOR": config.add_latency,
        }
    
    def estimate_cycles(self, instructions: List[Instruction], 
                        data: Dict[str, List[float]]) -> int:
        """
        Ước lượng số chu kỳ thực thi scalar
        
        Công thức:
            cycles_per_instruction = n * latency
            với n = vector_length (số phần tử)
        
        Args:
            instructions: Danh sách lệnh
            data: Dữ liệu đầu vào (để xác định vector length)
        
        Returns:
            int: Tổng số chu kỳ ước lượng
        """
        n = self._infer_vector_length(data)
        total_cycles = 0
        
        for inst in instructions:
            cycles = self._estimate_instruction_cycles(inst, n)
            total_cycles += cycles
            
            # Cập nhật thống kê
            self._update_stats(inst, cycles, n)
        
        return total_cycles
    
    def _estimate_instruction_cycles(self, inst: Instruction, n: int) -> int:
        """
        Ước lượng số chu kỳ cho một lệnh
        
        Args:
            inst: Lệnh cần ước lượng
            n: Số lượng phần tử
        
        Returns:
            int: Số chu kỳ
        """
        opcode = inst.opcode
        latency = self.latency_map.get(opcode, self.config.add_latency)
        
        # Các lệnh đặc biệt
        if opcode in ["VREDUCE_SUM", "VREDUCE_MAX", "VREDUCE_MIN"]:
            # Reduction cần n-1 phép toán
            return max(1, n - 1) * latency
        
        elif opcode == "VDOT":
            # Dot product: n phép nhân + (n-1) phép cộng
            mul_cycles = n * self.config.mul_latency
            add_cycles = max(1, n - 1) * self.config.add_latency
            return mul_cycles + add_cycles
        
        elif opcode in ["VGATHER", "VSCATTER"]:
            # Gather/scatter chậm hơn do truy xuất không tuần tự
            return n * latency
        
        else:
            # Lệnh thông thường: mỗi phần tử 1 lần
            return n * latency
    
    def _infer_vector_length(self, data: Dict[str, List[float]]) -> int:
        """
        Xác định độ dài vector từ dữ liệu đầu vào
        
        Args:
            data: Dictionary chứa các mảng dữ liệu
        
        Returns:
            int: Độ dài vector
        """
        if not data:
            return self.config.vector_length
        
        # Lấy độ dài của mảng đầu tiên
        first_array = next(iter(data.values()))
        return len(first_array)
    
    def _update_stats(self, inst: Instruction, cycles: int, n: int):
        """Cập nhật thống kê"""
        self.stats.total_cycles += cycles
        self.stats.total_instructions += 1
        self.stats.total_elements_processed += n
        
        opcode = inst.opcode
        self.stats.breakdown_by_opcode[opcode] = self.stats.breakdown_by_opcode.get(opcode, 0) + 1
        
        # Phân loại lệnh
        if inst.is_memory_instruction:
            self.stats.memory_instructions += 1
        elif inst.is_arithmetic:
            if inst.inst_type == InstructionType.ARITHMETIC_REDUCTION:
                self.stats.reduction_instructions += 1
            else:
                self.stats.arithmetic_instructions += 1
    
    def estimate_detailed(self, instructions: List[Instruction], 
                            data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Ước lượng chi tiết, phân tích từng lệnh
        
        Returns:
            Dict: Báo cáo chi tiết bao gồm cycles từng lệnh
        """
        n = self._infer_vector_length(data)
        detailed = {
            "vector_length": n,
            "total_cycles": 0,
            "instructions": []
        }
        
        for inst in instructions:
            cycles = self._estimate_instruction_cycles(inst, n)
            detailed["total_cycles"] += cycles
            detailed["instructions"].append({
                "opcode": inst.opcode,
                "dst": inst.dst,
                "src": inst.src,
                "cycles": cycles,
                "formula": self._get_cycle_formula(inst, n)
            })
        
        return detailed
    
    def _get_cycle_formula(self, inst: Instruction, n: int) -> str:
        """Lấy công thức tính cycles cho một lệnh"""
        opcode = inst.opcode
        latency = self.latency_map.get(opcode, self.config.add_latency)
        
        if opcode in ["VREDUCE_SUM", "VREDUCE_MAX", "VREDUCE_MIN"]:
            return f"({n}-1) × {latency} = {max(1, n-1)} × {latency} = {max(1, n-1) * latency}"
        elif opcode == "VDOT":
            mul_cycles = n * self.config.mul_latency
            add_cycles = (n - 1) * self.config.add_latency
            cycles = mul_cycles + add_cycles
            return f"{n} × {self.config.mul_latency} + ({n}-1) × {self.config.add_latency} = {cycles}"
        else:
            return f"{n} × {latency} = {n * latency}"
    
    # ===== Các phương thức tính toán cụ thể =====
    
    def cycles_for_vector_add(self, n: int) -> int:
        """Tính cycles cho VADD: n lần cộng"""
        return n * self.config.add_latency
    
    def cycles_for_vector_mul(self, n: int) -> int:
        """Tính cycles cho VMUL: n lần nhân"""
        return n * self.config.mul_latency
    
    def cycles_for_saxpy(self, n: int) -> int:
        """Tính cycles cho SAXPY: n lần nhân + n lần cộng"""
        mul_cycles = n * self.config.mul_latency
        add_cycles = n * self.config.add_latency
        return mul_cycles + add_cycles
    
    def cycles_for_dot_product(self, n: int) -> int:
        """Tính cycles cho dot product: n nhân + (n-1) cộng"""
        mul_cycles = n * self.config.mul_latency
        add_cycles = max(1, n - 1) * self.config.add_latency
        return mul_cycles + add_cycles
    
    def cycles_for_matrix_vector(self, rows: int, cols: int) -> int:
        """Tính cycles cho matrix-vector: rows * cols phép nhân + rows*(cols-1) phép cộng"""
        mul_cycles = rows * cols * self.config.mul_latency
        add_cycles = rows * max(1, cols - 1) * self.config.add_latency
        return mul_cycles + add_cycles
    
    def cycles_for_load_store(self, n: int, is_load: bool = True) -> int:
        """Tính cycles cho load/store"""
        latency = self.config.load_latency if is_load else self.config.store_latency
        return n * latency
    
    # ===== So sánh với vector processor =====
    
    def compare_with_vector(self, scalar_cycles: int, vector_cycles: int) -> Dict[str, float]:
        """
        So sánh hiệu năng scalar vs vector
        
        Returns:
            Dict: Các chỉ số so sánh
        """
        speedup = scalar_cycles / vector_cycles if vector_cycles > 0 else 0
        efficiency = speedup / self.config.num_lanes  # Hiệu suất sử dụng lanes
        reduction_percentage = (1 - vector_cycles / scalar_cycles) * 100 if scalar_cycles > 0 else 0
        
        return {
            "scalar_cycles": scalar_cycles,
            "vector_cycles": vector_cycles,
            "speedup": speedup,
            "efficiency": efficiency,
            "reduction_percentage": reduction_percentage
        }
    
    # ===== Thống kê và báo cáo =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê scalar simulator"""
        return {
            "total_cycles": self.stats.total_cycles,
            "total_instructions": self.stats.total_instructions,
            "memory_instructions": self.stats.memory_instructions,
            "arithmetic_instructions": self.stats.arithmetic_instructions,
            "reduction_instructions": self.stats.reduction_instructions,
            "total_elements_processed": self.stats.total_elements_processed,
            "breakdown_by_opcode": self.stats.breakdown_by_opcode,
            "avg_cycles_per_instruction": self.stats.total_cycles / self.stats.total_instructions if self.stats.total_instructions > 0 else 0,
            "avg_elements_per_cycle": self.stats.total_elements_processed / self.stats.total_cycles if self.stats.total_cycles > 0 else 0
        }
    
    def print_stats(self):
        """In thống kê scalar simulator"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("SCALAR SIMULATOR STATISTICS")
        print("=" * 60)
        print(f"Total cycles:              {stats['total_cycles']}")
        print(f"Total instructions:        {stats['total_instructions']}")
        print(f"Memory instructions:       {stats['memory_instructions']}")
        print(f"Arithmetic instructions:   {stats['arithmetic_instructions']}")
        print(f"Reduction instructions:    {stats['reduction_instructions']}")
        print(f"Total elements processed:  {stats['total_elements_processed']}")
        print(f"Avg cycles/instruction:    {stats['avg_cycles_per_instruction']:.2f}")
        print(f"Avg elements/cycle:        {stats['avg_elements_per_cycle']:.3f}")
        
        print("\nBreakdown by opcode:")
        for opcode, count in sorted(stats['breakdown_by_opcode'].items()):
            print(f"  {opcode}: {count}")
    
    def print_detailed_report(self, instructions: List[Instruction], data: Dict[str, List[float]]):
        """In báo cáo chi tiết từng lệnh"""
        detailed = self.estimate_detailed(instructions, data)
        
        print("\n" + "=" * 60)
        print("SCALAR EXECUTION DETAILED REPORT")
        print("=" * 60)
        print(f"Vector length: {detailed['vector_length']}")
        print(f"Total scalar cycles: {detailed['total_cycles']}")
        
        print("\nInstruction breakdown:")
        print(f"{'Instruction':<25} {'Cycles':<10} {'Formula'}")
        print("-" * 60)
        for inst in detailed['instructions']:
            inst_str = f"{inst['opcode']} {inst['dst']}"
            if inst['src']:
                inst_str += f", {', '.join(inst['src'])}"
            print(f"{inst_str:<25} {inst['cycles']:<10} {inst['formula']}")


# ===== CÁC HÀM TIỆN ÍCH =====

def estimate_scalar_performance(workload_name: str, n: int, config) -> int:
    """Ước lượng nhanh hiệu năng scalar cho các workload mẫu"""
    sim = ScalarSimulator(config)
    
    workloads = {
        "vector_add": lambda: sim.cycles_for_vector_add(n),
        "saxpy": lambda: sim.cycles_for_saxpy(n),
        "dot_product": lambda: sim.cycles_for_dot_product(n),
        "matrix_vector": lambda: sim.cycles_for_matrix_vector(int(math.sqrt(n)), int(math.sqrt(n))),
    }
    
    if workload_name in workloads:
        return workloads[workload_name]()
    else:
        # Mặc định
        return n * (config.load_latency + config.add_latency + config.store_latency)


def compare_scalar_vector(scalar_cycles: int, vector_cycles: int, num_lanes: int) -> Dict:
    """So sánh nhanh scalar vs vector"""
    speedup = scalar_cycles / vector_cycles if vector_cycles > 0 else 0
    ideal_speedup = num_lanes
    
    return {
        "speedup": speedup,
        "ideal_speedup": ideal_speedup,
        "efficiency": speedup / ideal_speedup if ideal_speedup > 0 else 0,
        "is_optimal": speedup >= ideal_speedup * 0.8  # 80% of ideal
    }


# ===== KIỂM TRA NHANH =====
if __name__ == "__main__":
    from config import VectorProcessorConfig
    from instruction import InstructionTemplates
    
    # Tạo config
    config = VectorProcessorConfig(
        vector_length=8,
        num_lanes=4,
        add_latency=2,
        mul_latency=4,
        load_latency=3,
        store_latency=3
    )
    
    # Tạo danh sách lệnh
    instructions = [
        InstructionTemplates.vload("V1", "A"),
        InstructionTemplates.vload("V2", "B"),
        InstructionTemplates.vadd("V3", "V1", "V2"),
        InstructionTemplates.vstore("C", "V3"),
    ]
    
    # Dữ liệu
    data = {"A": [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0], "B": [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0]}
    
    # Khởi tạo simulator
    sim = ScalarSimulator(config)
    
    # Ước lượng cycles
    total_cycles = sim.estimate_cycles(instructions, data)
    print(f"Estimated scalar cycles: {total_cycles}")
    
    # In báo cáo chi tiết
    sim.print_detailed_report(instructions, data)
    
    # In thống kê
    sim.print_stats()
    
    # Test các workload mẫu
    print("\n" + "=" * 60)
    print("WORKLOAD COMPARISON")
    print("=" * 60)
    
    n = 64
    workloads = ["vector_add", "saxpy", "dot_product"]
    for wl in workloads:
        cycles = estimate_scalar_performance(wl, n, config)
        print(f"{wl:15} (n={n}): {cycles:6} cycles")
    
    # Test compare function
    print("\n" + "=" * 60)
    print("SCALAR VS VECTOR COMPARISON")
    print("=" * 60)
    scalar = 880
    vector = 200
    comparison = compare_scalar_vector(scalar, vector, config.num_lanes)
    print(f"Scalar cycles: {scalar}")
    print(f"Vector cycles: {vector}")
    print(f"Speedup: {comparison['speedup']:.2f}x")
    print(f"Ideal speedup: {comparison['ideal_speedup']}x")
    print(f"Efficiency: {comparison['efficiency']:.2%}")
    print(f"Optimal: {comparison['is_optimal']}")
    
    print("\n✓ All tests passed!")