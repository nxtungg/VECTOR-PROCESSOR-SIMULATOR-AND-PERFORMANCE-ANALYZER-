"""Cache simulation cho vector processor.

MỨC ĐỘ TRỪU TƯỢNG (đọc trước khi dùng):
- Đây là mô hình cache mức KIẾN TRÚC/giáo dục, KHÔNG phải mô phỏng phần cứng
  chính xác tuyệt đối (không mô hình hoá MSHR, bus, banking, coherence...).
- Mục tiêu: phản ánh đúng hit/miss theo set-associative mapping + chính sách
  thay thế, và cung cấp một mô hình LATENCY rõ ràng để cache thực sự ảnh hưởng
  tới số chu kỳ trong performance model.

CHUẨN HOÁ ĐỊA CHỈ:
- ``access(address)`` nhận ELEMENT INDEX (chỉ số phần tử trong vector workload),
  KHÔNG phải byte address. Điều này phù hợp với cách Memory/VectorSimulator đánh
  địa chỉ theo phần tử.
- Các tham số kích thước (``size_bytes``, ``line_size_bytes``) vẫn tính theo BYTE
  để tương thích cấu hình phần cứng; nội bộ quy đổi sang "số phần tử / dòng" qua
  ``bytes_per_element`` rồi mọi tính toán index đều theo phần tử.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable


class CachePolicy(Enum):
    LRU = "lru"      # Least Recently Used
    FIFO = "fifo"    # First In First Out
    RANDOM = "random"


@dataclass
class CacheLine:
    """Một dòng cache (chỉ giữ METADATA: tag/valid/dirty + thông tin thay thế).

    Dữ liệu thực nằm ở Memory; cache chỉ theo dõi trạng thái nên ``data`` thường
    để trống — đủ cho mô hình hit/miss + write-back ở mức kiến trúc.
    """
    tag: int = -1
    valid: bool = False
    dirty: bool = False
    last_used: int = 0     # cho LRU (cập nhật mỗi lần truy xuất)
    fill_time: int = 0     # cho FIFO (đặt một lần khi nạp dòng)
    data: List[float] = field(default_factory=list)


class Cache:
    """Mô phỏng cache set-associative cho vector processor.

    Hỗ trợ:
    - Cache size / line size / associativity (set-associative mapping).
    - Thay thế LRU / FIFO / RANDOM.
    - Write-through / Write-back (có flush/buffer write-back tối thiểu).
    - Thống kê hit/miss và mô hình latency (hit_latency / miss_penalty).
    """

    def __init__(self, size_bytes: int = 32768,        # 32KB
                line_size_bytes: int = 64,            # 64 bytes/line
                associativity: int = 8,                # 8-way set associative
                policy: CachePolicy = CachePolicy.LRU,
                write_back: bool = True,
                hit_latency: int = 1,                  # cycles khi hit
                miss_penalty: int = 10,                # cycles phạt thêm khi miss
                flush_callback: Optional[Callable[[int, List[float]], None]] = None):

        self.size_bytes = size_bytes
        self.line_size_bytes = line_size_bytes
        self.associativity = max(1, associativity)
        self.policy = policy
        self.write_back = write_back

        # Mô hình latency rõ ràng (cache thực sự ảnh hưởng cycles)
        self.hit_latency = max(0, hit_latency)
        self.miss_penalty = max(0, miss_penalty)

        # Quy đổi byte -> phần tử (mọi index nội bộ theo PHẦN TỬ)
        self.bytes_per_element = 4  # float = 4 bytes
        self.line_size_elements = max(1, line_size_bytes // self.bytes_per_element)

        # Tính số dòng / số set
        self.num_lines = max(1, size_bytes // line_size_bytes)
        self.num_sets = max(1, self.num_lines // self.associativity)

        # Cache storage
        self.cache: Dict[int, List[CacheLine]] = {}
        for i in range(self.num_sets):
            self.cache[i] = [CacheLine() for _ in range(self.associativity)]

        # Thống kê hit/miss
        self.hits = 0
        self.misses = 0
        self.access_count = 0

        # Thống kê latency (để tích hợp vào performance model)
        self.total_latency = 0
        self.last_latency = 0

        # Write-back: buffer + counter (memory consistency tối thiểu)
        self.write_backs = 0
        self.write_back_buffer: List[Tuple[int, int]] = []  # (set_idx, tag) đã evict
        self.flush_callback = flush_callback

        # Counter cho chính sách thay thế
        self.access_counter = 0   # tick cho LRU
        self.fill_counter = 0     # thứ tự nạp dòng cho FIFO

    # ===== Ánh xạ địa chỉ (theo ELEMENT INDEX) =====

    def _get_set_index(self, address: int) -> int:
        """Set index từ element index."""
        line_address = address // self.line_size_elements
        return line_address % self.num_sets

    def _get_tag(self, address: int) -> int:
        """Tag từ element index."""
        return address // (self.line_size_elements * self.num_sets)

    def _find_line(self, set_idx: int, tag: int) -> Optional[int]:
        """Tìm line khớp tag trong set (None nếu không có)."""
        for i, line in enumerate(self.cache[set_idx]):
            if line.valid and line.tag == tag:
                return i
        return None

    def _find_victim(self, set_idx: int) -> int:
        """Chọn line để thay thế. Mọi chính sách ưu tiên dòng trống (invalid)."""
        lines = self.cache[set_idx]

        # Ưu tiên lấp dòng trống trước (cold miss không nên evict dữ liệu hợp lệ)
        for i, line in enumerate(lines):
            if not line.valid:
                return i

        if self.policy == CachePolicy.LRU:
            # Line có last_used nhỏ nhất = lâu chưa dùng nhất
            victim_idx = 0
            oldest = lines[0].last_used
            for i, line in enumerate(lines):
                if line.last_used < oldest:
                    oldest = line.last_used
                    victim_idx = i
            return victim_idx

        if self.policy == CachePolicy.FIFO:
            # Line có fill_time nhỏ nhất = nạp vào sớm nhất
            victim_idx = 0
            oldest = lines[0].fill_time
            for i, line in enumerate(lines):
                if line.fill_time < oldest:
                    oldest = line.fill_time
                    victim_idx = i
            return victim_idx

        # RANDOM
        return random.randint(0, self.associativity - 1)

    def _update_lru(self, set_idx: int, line_idx: int):
        """Cập nhật tick LRU khi truy xuất một line."""
        self.access_counter += 1
        self.cache[set_idx][line_idx].last_used = self.access_counter

    def _evict(self, set_idx: int, victim: CacheLine):
        """Xử lý write-back tối thiểu khi evict một dòng dirty.

        Write-back: thay vì bỏ qua, ta GHI LẠI (mô hình hoá) bằng cách đẩy vào
        write-back buffer + đếm write_backs, và gọi flush_callback nếu có để tầng
        memory áp dụng. Tránh tình trạng dirty bit không bao giờ được flush.
        """
        if victim.valid and victim.dirty and self.write_back:
            self.write_backs += 1
            self.write_back_buffer.append((set_idx, victim.tag))
            if self.flush_callback is not None:
                self.flush_callback(victim.tag, victim.data)

    # ===== Truy xuất =====

    def _access_core(self, address: int, is_write: bool) -> Tuple[bool, int]:
        """Lõi truy xuất: cập nhật trạng thái + thống kê, trả (hit, latency)."""
        self.access_count += 1
        set_idx = self._get_set_index(address)
        tag = self._get_tag(address)

        line_idx = self._find_line(set_idx, tag)

        if line_idx is not None:
            # ---- HIT ----
            self.hits += 1
            self._update_lru(set_idx, line_idx)
            if is_write:
                if self.write_back:
                    self.cache[set_idx][line_idx].dirty = True
                # write-through: ghi thẳng memory (mô hình ở tầng memory), không
                # cần dirty bit ở đây.
            latency = self.hit_latency
        else:
            # ---- MISS ----
            self.misses += 1
            victim_idx = self._find_victim(set_idx)
            victim = self.cache[set_idx][victim_idx]

            # Flush dòng cũ nếu dirty (write-back) trước khi nạp dòng mới
            self._evict(set_idx, victim)

            # Nạp dòng mới
            victim.tag = tag
            victim.valid = True
            victim.dirty = is_write and self.write_back
            victim.data = []
            self.fill_counter += 1
            victim.fill_time = self.fill_counter
            self._update_lru(set_idx, victim_idx)

            # Miss = chi phí truy xuất cache + phạt nạp từ tầng dưới
            latency = self.hit_latency + self.miss_penalty

        self.last_latency = latency
        self.total_latency += latency
        return (line_idx is not None), latency

    def access(self, address: int, is_write: bool = False) -> bool:
        """Truy xuất cache theo element index. Trả True nếu hit, False nếu miss.

        (Giữ chữ ký cũ để tương thích MemoryHierarchy / test.)
        """
        hit, _ = self._access_core(address, is_write)
        return hit

    def probe(self, address: int, is_write: bool = False) -> Tuple[bool, int]:
        """Như ``access`` nhưng trả kèm latency (cycles) — dùng cho performance model."""
        return self._access_core(address, is_write)

    # ===== Write-back consistency =====

    def flush_all(self):
        """Flush mọi dòng dirty xuống memory (đảm bảo nhất quán khi kết thúc).

        Gọi flush_callback cho từng dòng dirty và xoá dirty bit. Đếm vào
        write_backs để thống kê phản ánh đủ lưu lượng ghi lại.
        """
        for set_idx, lines in self.cache.items():
            for line in lines:
                if line.valid and line.dirty:
                    self.write_backs += 1
                    self.write_back_buffer.append((set_idx, line.tag))
                    if self.flush_callback is not None:
                        self.flush_callback(line.tag, line.data)
                    line.dirty = False

    # ===== Thống kê =====

    def get_stats(self) -> Dict[str, Any]:
        """Thống kê cache (hit/miss + mô hình latency + write-back)."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "miss_rate": self.misses / total if total > 0 else 0,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "line_size_bytes": self.line_size_bytes,
            "line_size_elements": self.line_size_elements,
            "associativity": self.associativity,
            "num_sets": self.num_sets,
            "policy": self.policy.value,
            "write_back": self.write_back,
            "hit_latency": self.hit_latency,
            "miss_penalty": self.miss_penalty,
            "total_latency": self.total_latency,
            "avg_access_latency": self.total_latency / total if total > 0 else 0,
            "write_backs": self.write_backs,
        }

    def reset(self):
        """Reset toàn bộ trạng thái cache và thống kê."""
        for i in range(self.num_sets):
            for j in range(self.associativity):
                self.cache[i][j] = CacheLine()
        self.hits = 0
        self.misses = 0
        self.access_count = 0
        self.total_latency = 0
        self.last_latency = 0
        self.write_backs = 0
        self.write_back_buffer = []
        self.access_counter = 0
        self.fill_counter = 0


class CacheSimulator:
    """Tầng cache của bộ nhớ vector processor (MEMORY SUBSYSTEM, không độc lập).

    Đặt một ``Cache`` phía trước một đối tượng Memory: mỗi truy xuất phần tử đi
    qua cache để (1) cập nhật hit/miss và (2) tính latency cộng vào số chu kỳ.
    Dữ liệu thực vẫn lấy từ Memory; cache chỉ quyết định CHI PHÍ truy xuất.

    Địa chỉ dùng ELEMENT INDEX, nhất quán với Cache và VectorSimulator.
    """

    def __init__(self, cache: Cache, memory: Any = None):
        self.cache = cache
        self.memory = memory
        # Memoize kết quả flatten để khỏi nối lại nhiều lần khi memory không đổi.
        self._flat_cache: Optional[List[float]] = None
        self._flat_sig: Optional[tuple] = None

    @staticmethod
    def _arrays_of(memory: Any):
        """Lấy dict mảng từ Memory (.data) hoặc dict thuần (None nếu không có)."""
        if memory is None:
            return None
        arrays = getattr(memory, "data", None)
        if arrays is None and isinstance(memory, dict):
            arrays = memory
        return arrays

    @staticmethod
    def _signature(arrays) -> tuple:
        """Chữ ký nhẹ (O(số mảng)) phát hiện memory thay đổi.

        Gồm (tên, id(list), độ dài) mỗi mảng: bắt được thêm/bớt mảng, đổi độ dài,
        và việc THAY THẾ list (vd Memory.store_vector gán list mới → id đổi). Không
        duyệt toàn bộ phần tử nên rẻ.
        """
        if not arrays:
            return ()
        return tuple((name, id(arr), len(arr)) for name, arr in arrays.items())

    def _flatten(self, memory: Any) -> List[float]:
        """View phẳng theo element index của các mảng trong memory (có memoize).

        Ghép các mảng theo thứ tự khai báo: phần tử thứ k toàn cục = phần tử k khi
        nối tiếp các mảng. Kết quả được cache lại; chỉ tính lại khi chữ ký memory
        đổi. Logic kết quả KHÔNG đổi so với việc nối trực tiếp.
        """
        arrays = self._arrays_of(memory)
        sig = self._signature(arrays)
        if self._flat_cache is not None and sig == self._flat_sig:
            return self._flat_cache

        flat: List[float] = []
        if arrays:
            for arr in arrays.values():
                flat.extend(arr)
        self._flat_cache = flat
        self._flat_sig = sig
        return flat

    def access_vector(self, base_index: int, num_elements: int,
                is_write: bool = False) -> Tuple[int, int, int]:
        """Truy xuất một vector tuần tự qua cache theo element index.

        Returns:
            (latency_cycles, hits, misses)
        """
        latency = hits = misses = 0
        for i in range(num_elements):
            hit, lat = self.cache.probe(base_index + i, is_write)
            latency += lat
            hits += int(hit)
            misses += int(not hit)
        return latency, hits, misses

    def load_with_cache(self, memory: Any, base_index: int,
                        num_elements: int) -> Tuple[List[float], int]:
        """Load ``num_elements`` từ element index ``base_index`` qua cache.

        Returns:
            (data, cycles): dữ liệu đọc từ memory + tổng latency (gồm hit/miss).
        """
        flat = self._flatten(memory if memory is not None else self.memory)
        data: List[float] = []
        cycles = 0
        for i in range(num_elements):
            idx = base_index + i
            _, lat = self.cache.probe(idx, is_write=False)
            cycles += lat
            data.append(flat[idx] if 0 <= idx < len(flat) else 0.0)
        return data, cycles

    def store_with_cache(self, base_index: int,
                        num_elements: int) -> int:
        """Store ``num_elements`` qua cache (write). Trả về tổng latency cycles."""
        latency, _, _ = self.access_vector(base_index, num_elements, is_write=True)
        return latency
