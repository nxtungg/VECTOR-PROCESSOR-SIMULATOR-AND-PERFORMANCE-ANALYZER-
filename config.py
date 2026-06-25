import json
from dataclasses import dataclass
from typing import Dict, Any
import math

@dataclass
class VectorProcessorConfig:
    """
    Cấu hình bộ xử lý vector
    """
    # ===== Tham số cơ bản =====
    vector_length: int = 8          # Số phần tử tối đa trong vector
    num_lanes: int = 4              # Số phần tử xử lý song song
    num_vector_registers: int = 16  # Số thanh ghi vector
    
    # ===== Độ trễ các lệnh (latency) =====
    load_latency: int = 3           # Độ trễ load từ memory
    store_latency: int = 3          # Độ trễ store xuống memory
    add_latency: int = 2            # Độ trễ phép cộng
    mul_latency: int = 4            # Độ trễ phép nhân
    div_latency: int = 8            # Độ trễ phép chia
    reduction_latency: int = 3      # Độ trễ reduction (cộng dồn)
    
    # ===== Tham số pipeline và bộ nhớ =====
    startup_latency: int = 1        # Độ trễ khởi động pipeline
    memory_bandwidth: int = 4       # Số phần tử đọc/ghi mỗi chu kỳ
    clock_frequency_mhz: int = 1000 # Tần số clock (MHz)
    default_stride: int = 1  # Stride mặc định cho memory access
    
    # ===== Tham số nâng cao =====
    num_memory_units: int = 1       # Số đơn vị memory (tránh structural hazard)
    num_alu_units: int = 1          # Số đơn vị ALU
    enable_chaining: bool = True    # Bật/tắt chaining
    enable_masking: bool = True     # Bật/tắt vector masking
    
    # ===== Tham số mô phỏng =====
    verbose: bool = False           # In chi tiết quá trình mô phỏng
    
    def __post_init__(self):
        """Tự động validate sau khi khởi tạo"""
        self.validate()
    
    def validate(self):
        """Kiểm tra tính hợp lệ của các tham số"""
        if self.vector_length <= 0:
            raise ValueError(f"vector_length must be positive, got {self.vector_length}")
        if self.num_lanes <= 0:
            raise ValueError(f"num_lanes must be positive, got {self.num_lanes}")
        if self.num_vector_registers <= 0:
            raise ValueError(f"num_vector_registers must be positive, got {self.num_vector_registers}")
        if self.memory_bandwidth <= 0:
            raise ValueError(f"memory_bandwidth must be positive, got {self.memory_bandwidth}")
        if self.load_latency < 1:
            raise ValueError(f"load_latency must be >= 1, got {self.load_latency}")
        if self.store_latency < 1:
            raise ValueError(f"store_latency must be >= 1, got {self.store_latency}")
        if self.clock_frequency_mhz <= 0:
            raise ValueError(f"clock_frequency_mhz must be positive, got {self.clock_frequency_mhz}")
    
    # ===== Các phương thức tiện ích =====
    
    def get_latency(self, opcode: str) -> int:
        """Lấy độ trễ của một lệnh cụ thể"""
        latency_map = {
            "VLOAD": self.load_latency,
            "VSTORE": self.store_latency,
            "VADD": self.add_latency,
            "VSUB": self.add_latency,
            "VMUL": self.mul_latency,
            "VDIV": self.div_latency,
            "VADDS": self.add_latency,
            "VSUBS": self.add_latency,
            "VMULS": self.mul_latency,
            "VMAX": self.add_latency,
            "VMIN": self.add_latency,
            "VREDUCE_SUM": self.reduction_latency,
            "VDOT": self.mul_latency + self.add_latency,
        }
        return latency_map.get(opcode, self.add_latency)
    
    def compute_cycles(self, num_elements: int, op_latency: int) -> int:
        """
        Tính số chu kỳ cho một lệnh vector
        Công thức: startup_latency + operation_latency + ceil(N / lanes)
        """
        groups = math.ceil(num_elements / self.num_lanes)
        return self.startup_latency + op_latency + groups
    
    def compute_memory_cycles(self, num_elements: int, is_load: bool = True) -> int:
        """
        Tính số chu kỳ cho lệnh memory
        Công thức: latency + ceil(N / memory_bandwidth)
        """
        latency = self.load_latency if is_load else self.store_latency
        transfers = math.ceil(num_elements / self.memory_bandwidth)
        return latency + transfers
    
    def get_peak_performance(self) -> float:
        """
        Tính hiệu năng đỉnh (FLOPS) của bộ xử lý
        """
        # Giả sử mỗi lane có thể thực hiện 1 phép toán mỗi chu kỳ
        return self.num_lanes * self.clock_frequency_mhz * 1e6
    
    # ===== Phương thức xuất/nhập cấu hình =====
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi config thành dictionary"""
        return {
            "vector_length": self.vector_length,
            "num_lanes": self.num_lanes,
            "num_vector_registers": self.num_vector_registers,
            "load_latency": self.load_latency,
            "store_latency": self.store_latency,
            "add_latency": self.add_latency,
            "mul_latency": self.mul_latency,
            "div_latency": self.div_latency,
            "reduction_latency": self.reduction_latency,
            "startup_latency": self.startup_latency,
            "memory_bandwidth": self.memory_bandwidth,
            "clock_frequency_mhz": self.clock_frequency_mhz,
            "num_memory_units": self.num_memory_units,
            "num_alu_units": self.num_alu_units,
            "enable_chaining": self.enable_chaining,
            "enable_masking": self.enable_masking,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorProcessorConfig":
        """Tạo config từ dictionary"""
        return cls(**data)
    
    def save_to_file(self, filepath: str):
        """Lưu config ra file JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "VectorProcessorConfig":
        """Đọc config từ file JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    # ===== Phương thức hiển thị =====
    
    def __str__(self) -> str:
        """Hiển thị cấu hình dạng đẹp"""
        lines = [
            "=" * 50,
            "VECTOR PROCESSOR CONFIGURATION",
            "=" * 50,
            f"  Vector Length:         {self.vector_length} elements",
            f"  Number of Lanes:       {self.num_lanes}",
            f"  Vector Registers:      {self.num_vector_registers}",
            "",
            "Latency (cycles):",
            f"  Load:                  {self.load_latency}",
            f"  Store:                 {self.store_latency}",
            f"  Add:                   {self.add_latency}",
            f"  Multiply:              {self.mul_latency}",
            f"  Divide:                {self.div_latency}",
            f"  Reduction:             {self.reduction_latency}",
            f"  Startup:               {self.startup_latency}",
            "",
            f"  Memory Bandwidth:      {self.memory_bandwidth} elements/cycle",
            f"  Clock Frequency:       {self.clock_frequency_mhz} MHz",
            f"  Peak Performance:      {self.get_peak_performance()/1e9:.1f} GFLOPS",
            "=" * 50
        ]
        return "\n".join(lines)
    
    def summary(self) -> str:
        """Tóm tắt nhanh cấu hình (dùng cho báo cáo)"""
        return f"VL={self.vector_length}, Lanes={self.num_lanes}, BW={self.memory_bandwidth}"


# ===== CÁC CẤU HÌNH MẪU CHO THÍ NGHIỆM =====

class ExperimentConfigs:
    """Các cấu hình mẫu cho thí nghiệm (trang 5-6 PDF)"""
    
    @staticmethod
    def exp1_vector_length() -> list:
        """Thí nghiệm 1: Ảnh hưởng của vector length"""
        configs = []
        for vl in [4, 8, 16, 32, 64]:
            config = VectorProcessorConfig(
                vector_length=vl,
                num_lanes=4,
                memory_bandwidth=4
            )
            configs.append(config)
        return configs
    
    @staticmethod
    def exp2_num_lanes() -> list:
        """Thí nghiệm 2: Ảnh hưởng của số lanes"""
        configs = []
        for lanes in [1, 2, 4, 8, 16]:
            config = VectorProcessorConfig(
                vector_length=64,
                num_lanes=lanes,
                memory_bandwidth=8
            )
            configs.append(config)
        return configs
    
    @staticmethod
    def exp3_memory_bandwidth() -> list:
        """Thí nghiệm 3: Ảnh hưởng của memory bandwidth"""
        configs = []
        for bw in [1, 2, 4, 8, 16]:
            config = VectorProcessorConfig(
                vector_length=64,
                num_lanes=8,
                memory_bandwidth=bw
            )
            configs.append(config)
        return configs
    
    @staticmethod
    def exp4_workload_comparison() -> dict:
        """Thí nghiệm 4: So sánh workload (dùng config cố định)"""
        base_config = VectorProcessorConfig(
            vector_length=16,
            num_lanes=4,
            memory_bandwidth=4
        )
        workloads = ["vector_add", "saxpy", "dot_product", "matrix_vector", "image_filter"]
        return {"config": base_config, "workloads": workloads}
    
    @staticmethod
    def exp5_stride_access() -> list:
        """Thí nghiệm 5: Ảnh hưởng của stride access"""
        configs = []
        for stride in [1, 2, 4, 8, 16]:
            config = VectorProcessorConfig(
                vector_length=64,
                num_lanes=4,
                memory_bandwidth=8
            )
            # Thêm stride vào config (sẽ dùng trong memory)
            config.default_stride = stride
            configs.append(config)
        return configs


# ===== HÀM TIỆN ÍCH =====

def get_default_config() -> VectorProcessorConfig:
    """Lấy cấu hình mặc định"""
    return VectorProcessorConfig()


def get_small_config() -> VectorProcessorConfig:
    """Cấu hình nhỏ (test nhanh)"""
    return VectorProcessorConfig(
        vector_length=4,
        num_lanes=2,
        num_vector_registers=4,
        memory_bandwidth=2
    )


def get_high_performance_config() -> VectorProcessorConfig:
    """Cấu hình hiệu năng cao"""
    return VectorProcessorConfig(
        vector_length=256,
        num_lanes=16,
        num_vector_registers=32,
        load_latency=2,
        store_latency=2,
        add_latency=1,
        mul_latency=2,
        memory_bandwidth=16,
        clock_frequency_mhz=2000
    )


def get_memory_bound_config() -> VectorProcessorConfig:
    """Cấu hình bị giới hạn bởi bộ nhớ"""
    return VectorProcessorConfig(
        vector_length=64,
        num_lanes=16,
        memory_bandwidth=2,  # Bandwidth thấp → bottleneck
        load_latency=10,
        store_latency=10
    )


# ===== TEST NHANH =====
if __name__ == "__main__":
    # Test config mặc định
    config = get_default_config()
    print(config)
    
    # Test các thí nghiệm
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Vector Length Variation")
    print("=" * 50)
    for cfg in ExperimentConfigs.exp1_vector_length():
        print(f"  {cfg.summary()}")
    
    # Test lưu/đọc JSON
    config.save_to_file("config_backup.json")
    loaded_config = VectorProcessorConfig.load_from_file("config_backup.json")
    print(f"\n✓ Config saved and loaded successfully")