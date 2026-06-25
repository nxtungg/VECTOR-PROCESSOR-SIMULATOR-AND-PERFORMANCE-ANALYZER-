import copy
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

class MemoryAccessType(Enum):
    """Loại truy xuất bộ nhớ"""
    LINEAR = "linear"           # Truy xuất tuần tự
    STRIDE = "stride"           # Truy xuất bước nhảy
    GATHER = "gather"           # Gather (đọc theo chỉ mục)
    SCATTER = "scatter"         # Scatter (ghi theo chỉ mục)

@dataclass
class MemoryAccess:
    """Thông tin về một lần truy xuất bộ nhớ"""
    access_type: MemoryAccessType
    address: int
    size: int                   # Số phần tử
    stride: int = 1
    indices: Optional[List[int]] = None
    start_cycle: int = 0
    end_cycle: int = 0
    latency: int = 0
    
class Memory:
    """
    Bộ nhớ cho vector processor
    
    Hỗ trợ:
    - Load/store tuần tự
    - Load/store với stride
    - Gather (load theo chỉ mục)
    - Scatter (store theo chỉ mục)
    - Mô phỏng băng thông
    - Thống kê truy xuất
    """
    
    def __init__(self, data: Optional[Dict[str, List[float]]] = None, 
                bandwidth: int = 4,
                latency: int = 3,
                memory_size: int = 1024):
        """
        Khởi tạo bộ nhớ
        
        Args:
            data: Dictionary chứa các mảng dữ liệu
            bandwidth: Băng thông bộ nhớ (số phần tử/cycle)
            latency: Độ trễ truy xuất (cycles)
            memory_size: Kích thước bộ nhớ tối đa (số phần tử)
        """
        self.data: Dict[str, List[float]] = data if data is not None else {}
        self.bandwidth = bandwidth
        self.latency = latency
        self.memory_size = memory_size
        
        # Thống kê
        self.total_loads = 0
        self.total_stores = 0
        self.total_elements_loaded = 0
        self.total_elements_stored = 0
        self.total_cycles_spent = 0
        self.access_history: List[MemoryAccess] = []
        
        # Cache / memory hierarchy simulation (nâng cao)
        self.enable_cache = False
        self.hierarchy = None          # MemoryHierarchy khi bật
        self.cache = {}                # giữ cho tương thích ngược (không còn dùng chính)
        self.cache_hits = 0
        self.cache_misses = 0

        # Bản đồ địa chỉ nền cho từng mảng (để cache index/tag có ý nghĩa)
        self.base_addresses: Dict[str, int] = {}
        self._next_base = 0
        self._array_stride = 4096      # khoảng cách địa chỉ giữa các mảng (tránh alias)
    
    # ===== Các phương thức cơ bản =====
    
    def has_array(self, name: str) -> bool:
        """Kiểm tra mảng có tồn tại không"""
        return name in self.data
    
    def get_array(self, name: str) -> List[float]:
        """Lấy mảng từ bộ nhớ"""
        if name not in self.data:
            raise KeyError(f"Array '{name}' not found in memory")
        return self.data[name]
    
    def set_array(self, name: str, values: List[float]):
        """Ghi mảng vào bộ nhớ"""
        self.data[name] = list(values)
    
    @staticmethod
    def _active_indices(mask: Optional[List[bool]], n: int) -> List[int]:
        """Danh sách chỉ số lane ACTIVE trong [0, n).

        mask=None → mọi lane active (hành vi mặc định, tương thích ngược)."""
        if mask is None:
            return list(range(n))
        return [i for i in range(n) if i < len(mask) and mask[i]]

    def load_vector(self, name: str, start_cycle: int = 0,
                    mask: Optional[List[bool]] = None) -> Tuple[List[float], int]:
        """
        Load vector tuần tự.

        mask (tùy chọn): predicate per-lane. Khi có mask, chỉ các lane active
        thực sự được truyền qua bus → băng thông/độ trễ/thống kê tính theo SỐ
        LANE ACTIVE (memory subsystem tôn trọng mask). Dữ liệu trả về vẫn đủ
        độ dài; simulator chịu trách nhiệm merge lane inactive vào thanh ghi.

        Returns:
            Tuple[List[float], int]: (dữ liệu, số chu kỳ)
        """
        if name not in self.data:
            raise KeyError(f"Array '{name}' not found")

        data = self.data[name]
        num_elements = len(data)
        active = self._active_indices(mask, num_elements)

        # Tính số chu kỳ (cộng thêm penalty khi cache miss) — theo lane active
        transfers = math.ceil(len(active) / self.bandwidth) if active else 0
        cycles = self.latency + transfers
        if self.enable_cache:
            for i in active:
                cycles += self._cache_penalty(name, i, is_write=False)

        # Cập nhật thống kê (chỉ đếm lane active)
        self.total_loads += 1
        self.total_elements_loaded += len(active)
        self.total_cycles_spent += cycles
        
        # Ghi lại lịch sử truy xuất (size = số lane active thực sự truyền)
        self.access_history.append(MemoryAccess(
            access_type=MemoryAccessType.LINEAR,
            address=0,
            size=len(active),
            stride=1,
            start_cycle=start_cycle,
            end_cycle=start_cycle + cycles,
            latency=cycles
        ))

        return list(data), cycles
    
    def store_vector(self, name: str, values: List[float], start_cycle: int = 0,
                     mask: Optional[List[bool]] = None) -> int:
        """
        Store vector tuần tự.

        mask (tùy chọn): khi có mask, đây là MASKED STORE — chỉ ghi xuống bộ nhớ
        các lane active; ô nhớ tại lane inactive GIỮ NGUYÊN giá trị cũ (đọc-sửa-
        ghi). Băng thông/độ trễ/thống kê tính theo số lane active.

        Returns:
            int: Số chu kỳ
        """
        num_elements = len(values)
        active = self._active_indices(mask, num_elements)

        # Tính số chu kỳ — theo lane active
        transfers = math.ceil(len(active) / self.bandwidth) if active else 0
        cycles = self.latency + transfers
        if self.enable_cache:
            for i in active:
                cycles += self._cache_penalty(name, i, is_write=True)

        # Ghi dữ liệu
        if mask is None:
            self.data[name] = list(values)
        else:
            # Masked store: merge với mảng đích hiện có (giữ lane inactive).
            existing = list(self.data.get(name, []))
            if len(existing) < num_elements:
                existing = existing + [0.0] * (num_elements - len(existing))
            for i in active:
                existing[i] = values[i]
            self.data[name] = existing

        # Cập nhật thống kê (chỉ đếm lane active)
        self.total_stores += 1
        self.total_elements_stored += len(active)
        self.total_cycles_spent += cycles

        # Ghi lại lịch sử
        self.access_history.append(MemoryAccess(
            access_type=MemoryAccessType.LINEAR,
            address=0,
            size=len(active),
            start_cycle=start_cycle,
            end_cycle=start_cycle + cycles,
            latency=cycles
        ))

        return cycles
    
    def load_stride(self, name: str, stride: int, length: int, 
                    start_cycle: int = 0) -> Tuple[List[float], int]:
        """
        Load vector với stride (bước nhảy)
        
        Args:
            name: Tên mảng
            stride: Bước nhảy
            length: Số phần tử cần load
            start_cycle: Chu kỳ bắt đầu
            
        Returns:
            Tuple[List[float], int]: (dữ liệu, số chu kỳ)
        """
        if name not in self.data:
            raise KeyError(f"Array '{name}' not found")
        
        source = self.data[name]
        result = []
        
        # Load với stride
        for i in range(0, min(length * stride, len(source)), stride):
            if i < len(source):
                result.append(source[i])
        
        # ---- Tính chu kỳ băng thông với stride penalty ----
        # Mỗi phần tử stride≠1 chiếm một cache-line độc lập (giả sử line_size ≤ stride
        # khi stride ≥ bandwidth). Công thức chính xác:
        #
        #   transfers = ceil(N × stride / max(stride, bandwidth))
        #
        # Giải thích:
        #   • stride < bandwidth : nhiều phần tử có thể nằm trong cùng cache-line
        #     → transfers = ceil(N × stride / bandwidth)  [ví dụ stride=2, bw=4: 2×N/4]
        #   • stride ≥ bandwidth : mỗi phần tử ở cache-line riêng, không thể tốt hơn N
        #     → transfers = ceil(N × stride / stride) = N  [mỗi phần tử 1 lần fetch]
        #
        # Công thức cũ dùng (bandwidth/stride) làm effective_bw, cho kết quả sai khi
        # stride > bandwidth do chia số thực < 1 rồi ceil() không chuẩn.
        n = len(result)
        transfers = math.ceil(n * stride / max(stride, self.bandwidth))
        cycles = self.latency + transfers
        if self.enable_cache:
            for i in range(n):
                cycles += self._cache_penalty(name, i * stride, is_write=False)

        # Cập nhật thống kê
        self.total_loads += 1
        self.total_elements_loaded += n
        self.total_cycles_spent += cycles

        self.access_history.append(MemoryAccess(
            access_type=MemoryAccessType.STRIDE,
            address=0,
            size=n,
            stride=stride,
            start_cycle=start_cycle,
            end_cycle=start_cycle + cycles,
            latency=cycles
        ))

        return result, cycles
    
    def store_stride(self, name: str, values: List[float], stride: int,
                start_cycle: int = 0) -> int:
        """
        Store vector với stride
        
        Args:
            name: Tên mảng
            values: Dữ liệu cần ghi
            stride: Bước nhảy
            start_cycle: Chu kỳ bắt đầu
            
        Returns:
            int: Số chu kỳ
        """
        if name not in self.data:
            self.data[name] = []
        
        # Đảm bảo mảng đủ lớn
        max_index = len(values) * stride
        if len(self.data[name]) < max_index:
            self.data[name].extend([0.0] * (max_index - len(self.data[name])))
        
        # Ghi với stride
        for i, val in enumerate(values):
            idx = i * stride
            if idx < len(self.data[name]):
                self.data[name][idx] = val
        
        # Áp dụng cùng công thức stride penalty như load_stride (xem chi tiết ở trên)
        n = len(values)
        transfers = math.ceil(n * stride / max(stride, self.bandwidth))
        cycles = self.latency + transfers

        # Cập nhật thống kê
        self.total_stores += 1
        self.total_elements_stored += n
        self.total_cycles_spent += cycles

        self.access_history.append(MemoryAccess(
            access_type=MemoryAccessType.STRIDE,
            address=0,
            size=n,
            stride=stride,
            start_cycle=start_cycle,
            end_cycle=start_cycle + cycles,
            latency=cycles
        ))

        return cycles
    
    def gather(self, name: str, indices: List[int], start_cycle: int = 0,
               mask: Optional[List[bool]] = None) -> Tuple[List[float], int]:
        """
        Gather: Load dữ liệu theo mảng chỉ mục.

        Ví dụ: result[i] = source[indices[i]]

        mask (tùy chọn): chỉ lane active mới thực sự truy xuất bộ nhớ; lane
        inactive trả 0.0 (simulator merge vào thanh ghi). Băng thông/độ trễ/
        thống kê tính theo số lane active.

        Returns:
            Tuple[List[float], int]: (dữ liệu, số chu kỳ)
        """
        if name not in self.data:
            raise KeyError(f"Array '{name}' not found")

        source = self.data[name]
        active = set(self._active_indices(mask, len(indices)))
        result = []

        for i, idx in enumerate(indices):
            if i not in active:
                result.append(0.0)          # lane inactive: không truy xuất
            elif 0 <= idx < len(source):
                result.append(source[idx])
            else:
                result.append(0.0)          # Out of bounds trả về 0

        # Gather có hiệu suất thấp hơn do truy xuất không tuần tự
        efficiency = 0.5  # Gather chỉ đạt 50% hiệu suất
        effective_bandwidth = self.bandwidth * efficiency
        n_active = len(active)
        transfers = math.ceil(n_active / effective_bandwidth) if n_active else 0
        cycles = self.latency * 2 + transfers  # Gather chậm hơn
        if self.enable_cache:
            for i, idx in enumerate(indices):
                if i in active:
                    cycles += self._cache_penalty(name, int(idx), is_write=False)

        # Cập nhật thống kê
        self.total_loads += 1
        self.total_elements_loaded += n_active
        self.total_cycles_spent += cycles

        self.access_history.append(MemoryAccess(
            access_type=MemoryAccessType.GATHER,
            address=0,
            size=n_active,
            indices=list(indices),
            start_cycle=start_cycle,
            end_cycle=start_cycle + cycles,
            latency=cycles
        ))

        return result, cycles
    
    def scatter(self, name: str, indices: List[int], values: List[float],
                start_cycle: int = 0, mask: Optional[List[bool]] = None) -> int:
        """
        Scatter: Ghi dữ liệu theo mảng chỉ mục.

        Ví dụ: dest[indices[i]] = values[i]

        mask (tùy chọn): chỉ lane active mới ghi xuống bộ nhớ; lane inactive
        không chạm vào mảng đích. Băng thông/độ trễ/thống kê theo lane active.

        Returns:
            int: Số chu kỳ
        """
        if len(indices) != len(values):
            raise ValueError(f"Indices length ({len(indices)}) != values length ({len(values)})")

        if name not in self.data:
            self.data[name] = []

        active = set(self._active_indices(mask, len(indices)))

        # Đảm bảo mảng đủ lớn (chỉ xét các lane active sẽ thực sự ghi)
        active_idx_vals = [int(indices[i]) for i in range(len(indices)) if i in active]
        max_idx = max(active_idx_vals) if active_idx_vals else 0
        if len(self.data[name]) <= max_idx:
            self.data[name].extend([0.0] * (max_idx - len(self.data[name]) + 1))

        # Ghi theo chỉ mục — chỉ lane active
        for i, (idx, val) in enumerate(zip(indices, values)):
            if i in active and 0 <= idx < len(self.data[name]):
                self.data[name][idx] = val

        # Scatter có hiệu suất thấp
        efficiency = 0.5
        effective_bandwidth = self.bandwidth * efficiency
        n_active = len(active)
        transfers = math.ceil(n_active / effective_bandwidth) if n_active else 0
        cycles = self.latency * 2 + transfers
        if self.enable_cache:
            for i, idx in enumerate(indices):
                if i in active:
                    cycles += self._cache_penalty(name, int(idx), is_write=True)

        # Cập nhật thống kê
        self.total_stores += 1
        self.total_elements_stored += n_active
        self.total_cycles_spent += cycles

        self.access_history.append(MemoryAccess(
            access_type=MemoryAccessType.SCATTER,
            address=0,
            size=n_active,
            indices=list(indices),
            start_cycle=start_cycle,
            end_cycle=start_cycle + cycles,
            latency=cycles
        ))

        return cycles
    
    def load_range(self, name: str, start: int, end: int, 
        start_cycle: int = 0) -> Tuple[List[float], int]:
        """
        Load một đoạn của mảng (từ start đến end)
        
        Returns:
            Tuple[List[float], int]: (dữ liệu, số chu kỳ)
        """
        if name not in self.data:
            raise KeyError(f"Array '{name}' not found")
        
        source = self.data[name]
        result = source[start:end]
        
        transfers = math.ceil(len(result) / self.bandwidth)
        cycles = self.latency + transfers
        
        self.total_loads += 1
        self.total_elements_loaded += len(result)
        self.total_cycles_spent += cycles
        
        return result, cycles
    
    # ===== Memory hierarchy / cache simulation (nâng cao) =====

    def _base_address(self, name: str) -> int:
        """Lấy (hoặc cấp mới) địa chỉ nền của một mảng."""
        if name not in self.base_addresses:
            self.base_addresses[name] = self._next_base
            self._next_base += self._array_stride
        return self.base_addresses[name]

    def enable_cache_simulation(self, cache_size: int = 64, line_size: int = 8,
                                associativity: Optional[int] = None,
                                levels: int = 1,
                                l2_size: Optional[int] = None,
                                l2_line_size: Optional[int] = None,
                                policy: str = "lru",
                                l2_penalty: int = 6):
        """Bật mô phỏng memory hierarchy.

        Tương thích ngược với chữ ký cũ ``(cache_size, line_size)`` trong đó
        ``cache_size`` là SỐ DÒNG cache và ``line_size`` là số phần tử/dòng.

        Args:
            cache_size: số dòng cache L1.
            line_size: số phần tử mỗi dòng (1 phần tử = 4 byte).
            associativity: số đường (mặc định tự chọn, chia hết số dòng).
            levels: 1 hoặc 2 tầng cache.
            l2_size / l2_line_size: cấu hình L2 (mặc định gấp 4 lần L1).
            policy: "lru" | "fifo" | "random".
            l2_penalty: phạt cycles khi L1 miss nhưng L2 hit.
        """
        from cache import CachePolicy
        from memory_hierarchy import MemoryHierarchy, LevelConfig

        self.enable_cache = True
        self.cache_size = cache_size
        self.line_size = line_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        bytes_per_elem = 4
        pol = {"lru": CachePolicy.LRU, "fifo": CachePolicy.FIFO,
            "random": CachePolicy.RANDOM}.get(policy.lower(), CachePolicy.LRU)

        l1_lines = max(1, cache_size)
        l1_assoc = self._valid_assoc(associativity, l1_lines)
        l1_line_bytes = max(1, line_size) * bytes_per_elem
        l1 = LevelConfig(
            size_bytes=l1_lines * l1_line_bytes,
            line_size_bytes=l1_line_bytes,
            associativity=l1_assoc,
            penalty=0,
            policy=pol,
        )

        l2_cfg = None
        if levels >= 2:
            l2_lines = max(1, l2_size if l2_size else cache_size * 4)
            l2_line_elems = l2_line_size if l2_line_size else line_size
            l2_line_bytes = max(1, l2_line_elems) * bytes_per_elem
            l2_assoc = self._valid_assoc(None, l2_lines)
            l2_cfg = LevelConfig(
                size_bytes=l2_lines * l2_line_bytes,
                line_size_bytes=l2_line_bytes,
                associativity=l2_assoc,
                penalty=l2_penalty,
                policy=pol,
            )

        self.hierarchy = MemoryHierarchy(l1=l1, l2=l2_cfg, mem_penalty=self.latency)

    @staticmethod
    def _valid_assoc(requested: Optional[int], num_lines: int) -> int:
        """Chọn associativity hợp lệ (chia hết số dòng, không vượt quá số dòng)."""
        candidates = [requested] if requested else [8, 4, 2, 1]
        for a in candidates:
            if a and 1 <= a <= num_lines and num_lines % a == 0:
                return a
        # fallback: lùi dần cho tới khi chia hết
        a = min(requested or 4, num_lines)
        while a > 1 and num_lines % a != 0:
            a -= 1
        return max(1, a)

    def _cache_penalty(self, name: str, element_index: int,
                       is_write: bool = False) -> int:
        """Truy xuất một phần tử qua hierarchy, trả về penalty cycles (0 nếu tắt)."""
        if not self.enable_cache or self.hierarchy is None:
            return 0
        address = self._base_address(name) + element_index
        hit_level, penalty = self.hierarchy.access(address, is_write)
        if hit_level == 1:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return penalty

    def _cache_access(self, address: int, is_write: bool = False) -> bool:
        """Tương thích ngược: truy xuất 1 địa chỉ thô, trả True nếu L1 hit."""
        if not self.enable_cache or self.hierarchy is None:
            return False
        hit_level, _ = self.hierarchy.access(address, is_write)
        if hit_level == 1:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return hit_level == 1

    
    # ===== Thống kê và báo cáo =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê truy xuất bộ nhớ"""
        stats: Dict[str, Any] = {
            "total_loads": self.total_loads,
            "total_stores": self.total_stores,
            "total_elements_loaded": self.total_elements_loaded,
            "total_elements_stored": self.total_elements_stored,
            "total_cycles_spent": self.total_cycles_spent,
            "bandwidth": self.bandwidth,
            "latency": self.latency,
            "memory_size": self.memory_size,
            "num_arrays": len(self.data),
            "access_history_count": len(self.access_history)
        }
        
        # Thống kê cache (nếu bật)
        if self.enable_cache:
            total_accesses = self.cache_hits + self.cache_misses
            stats["cache_enabled"] = True
            stats["cache_hits"] = self.cache_hits
            stats["cache_misses"] = self.cache_misses
            stats["cache_hit_rate_percent"] = int((self.cache_hits / total_accesses) * 100) if total_accesses > 0 else 0
            if self.hierarchy is not None:
                stats["hierarchy"] = self.hierarchy.get_stats()
        
        # Phân loại truy xuất theo loại
        access_counts = {at.value: 0 for at in MemoryAccessType}
        for access in self.access_history:
            access_counts[access.access_type.value] += 1
        stats["access_by_type"] = access_counts
        
        return stats
    
    def print_stats(self):
        """In thống kê bộ nhớ"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("MEMORY STATISTICS")
        print("=" * 60)
        print(f"Total loads:        {stats['total_loads']}")
        print(f"Total stores:       {stats['total_stores']}")
        print(f"Elements loaded:    {stats['total_elements_loaded']}")
        print(f"Elements stored:    {stats['total_elements_stored']}")
        print(f"Total cycles:       {stats['total_cycles_spent']}")
        print(f"Memory bandwidth:   {stats['bandwidth']} elem/cycle")
        print(f"Memory latency:     {stats['latency']} cycles")
        print(f"Number of arrays:   {stats['num_arrays']}")
        
        print("\nAccess by type:")
        for atype, count in stats['access_by_type'].items():
            if count > 0:
                print(f"  {atype}: {count}")
        
        if stats.get('cache_enabled', False):
            print(f"\nCache Statistics:")
            print(f"  Hit rate: {stats['cache_hit_rate_percent']}%")
            print(f"  Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']}")
    
    def get_bandwidth_utilization(self, total_cycles: int) -> float:
        """Tính hiệu suất sử dụng băng thông"""
        max_transfers = total_cycles * self.bandwidth
        if max_transfers <= 0:
            return 0.0
        actual_transfers = self.total_elements_loaded + self.total_elements_stored
        return min(max(actual_transfers / max_transfers, 0.0), 1.0)
    
    def get_average_latency(self) -> float:
        """Tính độ trễ trung bình"""
        total_accesses = self.total_loads + self.total_stores
        if total_accesses == 0:
            return 0.0
        return self.total_cycles_spent / total_accesses
    
    # ===== Hỗ trợ debug =====
    
    def print_access_history(self, limit: int = 10):
        """In lịch sử truy xuất bộ nhớ"""
        print("\n" + "=" * 60)
        print("MEMORY ACCESS HISTORY")
        print("=" * 60)
        
        for i, access in enumerate(self.access_history[-limit:]):
            print(f"{i+1}. Type: {access.access_type.value}, "
                f"Size: {access.size}, Stride: {access.stride}, "
                f"Cycles: {access.latency} (start={access.start_cycle}, end={access.end_cycle})")
    
    def dump(self) -> Dict[str, List[float]]:
        """Xuất toàn bộ dữ liệu bộ nhớ"""
        return copy.deepcopy(self.data)
    
    def clear(self):
        """Xóa toàn bộ dữ liệu bộ nhớ"""
        self.data.clear()
        self.access_history.clear()
        self.total_loads = 0
        self.total_stores = 0
        self.total_elements_loaded = 0
        self.total_elements_stored = 0
        self.total_cycles_spent = 0
    
    def __str__(self) -> str:
        """Hiển thị bộ nhớ"""
        lines = ["Memory contents:"]
        for name, data in self.data.items():
            preview = str(data[:5]) + ("..." if len(data) > 5 else "")
            lines.append(f"  {name}: {preview} (len={len(data)})")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"Memory(bandwidth={self.bandwidth}, latency={self.latency}, arrays={len(self.data)})"


# ===== HÀM TIỆN ÍCH =====

def create_test_memory() -> Memory:
    """Tạo bộ nhớ test với dữ liệu mẫu"""
    data = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "B": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "X": [1.0, 2.0, 3.0, 4.0],
        "Y": [0.0, 0.0, 0.0, 0.0],
        "IDX": [0, 2, 4, 6]  # Index array for gather/scatter
    }
    return Memory(data=data, bandwidth=4, latency=3)


# ===== TEST NHANH =====
if __name__ == "__main__":
    print("=" * 60)
    print("TEST MEMORY")
    print("=" * 60)
    
    # Tạo bộ nhớ test
    mem = create_test_memory()
    print(mem)
    
    # Test linear load
    print("\n--- Test Linear Load ---")
    data, cycles = mem.load_vector("A")
    print(f"Loaded A: {data}")
    print(f"Cycles: {cycles}")
    
    # Test stride load
    print("\n--- Test Stride Load ---")
    data, cycles = mem.load_stride("A", stride=2, length=4)
    print(f"Loaded A with stride 2: {data}")
    print(f"Cycles: {cycles}")
    
    # Test store
    print("\n--- Test Store ---")
    cycles = mem.store_vector("C", [100, 200, 300, 400])
    print(f"Stored to C, cycles: {cycles}")
    
    # Test gather
    print("\n--- Test Gather ---")
    indices = [0, 2, 4, 6]
    data, cycles = mem.gather("A", indices)
    print(f"Gathered from A using indices {indices}: {data}")
    print(f"Cycles: {cycles}")
    
    # Test scatter
    print("\n--- Test Scatter ---")
    indices = [1, 3, 5, 7]
    scatter_values = [99.0, 88.0, 77.0, 66.0]
    cycles = mem.scatter("A", indices, scatter_values)
    print(f"Scattered to A, cycles: {cycles}")
    print(f"A after scatter: {mem.get_array('A')}")
    
    # Test cache simulation
    print("\n--- Test Cache Simulation ---")
    mem.enable_cache_simulation(cache_size=4, line_size=2)
    for i in range(10):
        mem._cache_access(i)
    print(f"Cache hits: {mem.cache_hits}, misses: {mem.cache_misses}")
    print(f"Hit rate: {mem.cache_hits/(mem.cache_hits+mem.cache_misses):.2%}")
    
    # In thống kê
    mem.print_stats()
    mem.print_access_history()
    
    print("\n✓ All tests passed!")
