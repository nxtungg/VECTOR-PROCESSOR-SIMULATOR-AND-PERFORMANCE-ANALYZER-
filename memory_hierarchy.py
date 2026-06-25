"""Memory hierarchy hoàn chỉnh (L1 + L2 + main memory) cho vector processor.

Ghép các tầng cache (tái dùng ``Cache`` set-associative trong ``cache.py``) thành
một hệ phân cấp đầy đủ với miss penalty theo từng tầng. Mọi đường truy xuất bộ
nhớ trong ``memory.py`` đi qua đây để cache thực sự ảnh hưởng tới timing.

Mô hình penalty (cycles cộng thêm so với truy xuất lý tưởng):
- L1 hit:        0 (đã tính trong latency cơ bản của Memory)
- L1 miss/L2 hit: l2_penalty
- L2 miss (mem): l2_penalty + mem_penalty
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from cache import Cache, CachePolicy


@dataclass
class LevelConfig:
    """Cấu hình một tầng cache."""
    size_bytes: int
    line_size_bytes: int
    associativity: int
    penalty: int          # cycles phạt khi miss ở tầng này (phải xuống tầng dưới)
    policy: CachePolicy = CachePolicy.LRU


class MemoryHierarchy:
    """Hệ phân cấp bộ nhớ nhiều tầng.

    ``access(address, is_write)`` mô phỏng một truy xuất 1 phần tử và trả về
    ``(hit_level, penalty_cycles)`` trong đó:
    - hit_level = 1 nếu L1 hit, = 2 nếu L2 hit, = 0 nếu phải xuống main memory.
    - penalty_cycles = số chu kỳ phạt cộng thêm.
    """

    def __init__(self,
                l1: Optional[LevelConfig] = None,
                l2: Optional[LevelConfig] = None,
                mem_penalty: int = 0):
        if l1 is None:
            l1 = LevelConfig(size_bytes=1024, line_size_bytes=32,
                            associativity=4, penalty=0)
        self.l1_cfg = l1
        self.l2_cfg = l2
        self.mem_penalty = mem_penalty

        self.l1 = Cache(size_bytes=l1.size_bytes,
                        line_size_bytes=l1.line_size_bytes,
                        associativity=l1.associativity,
                        policy=l1.policy)
        self.l2: Optional[Cache] = None
        if l2 is not None:
            self.l2 = Cache(size_bytes=l2.size_bytes,
                            line_size_bytes=l2.line_size_bytes,
                            associativity=l2.associativity,
                            policy=l2.policy)

        # Thống kê per-level
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.mem_accesses = 0
        self.total_penalty = 0

    def access(self, address: int, is_write: bool = False):
        """Mô phỏng truy xuất một phần tử qua hệ phân cấp.

        Returns:
            (hit_level, penalty_cycles)
        """
        penalty = 0

        # --- L1 ---
        if self.l1.access(address, is_write):
            self.l1_hits += 1
            return 1, 0

        # L1 miss
        self.l1_misses += 1
        penalty += self.l1_cfg.penalty

        # --- L2 (nếu có) ---
        l2 = self.l2
        l2_cfg = self.l2_cfg
        if l2 is not None and l2_cfg is not None:
            penalty += l2_cfg.penalty
            if l2.access(address, is_write):
                self.l2_hits += 1
                self.total_penalty += penalty
                return 2, penalty
            # L2 miss
            self.l2_misses += 1

        # --- Main memory ---
        self.mem_accesses += 1
        penalty += self.mem_penalty
        self.total_penalty += penalty
        return 0, penalty

    # ===== Thống kê =====

    @property
    def hits(self) -> int:
        """Tổng số hit ở mọi tầng (tương thích với thống kê cache cũ)."""
        return self.l1_hits + self.l2_hits

    @property
    def misses(self) -> int:
        """Số truy xuất cuối cùng phải xuống main memory (true miss)."""
        return self.mem_accesses

    def get_stats(self) -> Dict[str, Any]:
        l1_total = self.l1_hits + self.l1_misses
        stats: Dict[str, Any] = {
            "levels": 2 if self.l2 is not None else 1,
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l1_hit_rate": self.l1_hits / l1_total if l1_total else 0.0,
            "mem_accesses": self.mem_accesses,
            "total_penalty_cycles": self.total_penalty,
        }
        if self.l2 is not None:
            l2_total = self.l2_hits + self.l2_misses
            stats.update({
                "l2_hits": self.l2_hits,
                "l2_misses": self.l2_misses,
                "l2_hit_rate": self.l2_hits / l2_total if l2_total else 0.0,
            })
        return stats

    def reset(self):
        self.l1.reset()
        if self.l2 is not None:
            self.l2.reset()
        self.l1_hits = self.l1_misses = 0
        self.l2_hits = self.l2_misses = 0
        self.mem_accesses = 0
        self.total_penalty = 0
