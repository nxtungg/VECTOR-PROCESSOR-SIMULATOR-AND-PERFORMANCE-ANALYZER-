"""Out-of-Order execution simulator (mức giáo dục, dựa trên dependency).

Mô hình 4 stage rõ ràng, mỗi chu kỳ thực hiện theo thứ tự:

COMMIT    – commit lệnh đầu window nếu đã WRITEBACK (in-order, 1 lệnh/cycle)
WRITEBACK – lệnh execute xong → cập nhật ready_at để báo source sẵn sàng
EXECUTE   – lệnh ISSUE bắt đầu execute khi dependency thỏa (out-of-order)
ISSUE     – đưa lệnh tiếp theo vào window nếu còn chỗ (in-order)

Memory ordering (đơn giản, không làm cache coherence):
- Store chờ store trước hoàn thành writeback (store-store ordering)
- Load  chờ store trước hoàn thành writeback (tránh load-store hazard)
- Các load không có store ở giữa có thể chạy song song
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from memory import Memory
from functional_core import FunctionalCore, is_scalar_result
from instruction import FunctionalUnit

_STORE_OPS = {"VSTORE", "VSCATTER"}
_LOAD_OPS  = {"VLOAD", "VLOAD_STRIDE", "VGATHER"}


@dataclass
class _Entry:
    """Một lệnh đang nằm trong instruction window."""
    idx: int            # vị trí trong chương trình, dùng để commit in-order
    inst: Any
    fu: str
    latency: int
    state: str = "ISSUE"   # ISSUE → EXEC → WRITEBACK → COMMIT
    issue_c: int = -1
    exec_start: int = -1
    exec_end: int = -1
    wb_c: int = -1
    commit_c: int = -1
    # Memory ordering: trỏ tới store được issue trước lệnh này (nếu có)
    wait_for: Optional[Any] = field(default=None, repr=False)


class OOOSimulator:
    """Mô phỏng out-of-order đơn giản dựa trên dependency tracking."""

    def __init__(self, config, data: Optional[Dict] = None,
                scalars: Optional[Dict] = None,
                num_rs: int = 8, rob_size: int = 16):
        self.config = config
        self.data = dict(data or {})
        self.scalars = dict(scalars or {})
        self.window_size = num_rs   # kích thước instruction window
        self.num_memory_units = max(1, getattr(config, "num_memory_units", 1))
        self.num_alu_units = max(1, getattr(config, "num_alu_units", 1))
        self.num_mul_units = max(1, getattr(config, "num_mul_units", self.num_alu_units))
        self.num_reduction_units = max(1, getattr(config, "num_reduction_units", 1))
        self.cycle = 0
        self.stats = {
            "instructions_issued": 0,
            "instructions_committed": 0,
            "stalls": 0,
        }

    # ===== Tiện ích =====

    def _fu_of(self, opcode: str) -> str:
        if opcode in _LOAD_OPS | _STORE_OPS:
            return "MEMORY"
        if opcode in ("VREDUCE_SUM", "VREDUCE_MAX", "VREDUCE_MIN", "VDOT"):
            return "REDUCTION"
        if opcode in ("VMUL", "VDIV", "VMULS", "VDIVS"):
            return "MUL"
        return "ALU"

    def _fu_display(self, fu: str) -> str:
        return {
            "MEMORY":    FunctionalUnit.MEMORY.value,
            "ALU":       FunctionalUnit.ALU.value,
            "MUL":       FunctionalUnit.ALU.value,
            "REDUCTION": FunctionalUnit.REDUCTION.value,
        }.get(fu, fu)

    def _latency(self, inst, memory: Optional[Memory] = None) -> int:
        op = inst.opcode
        c = self.config
        if memory is not None:
            if op == "VLOAD":
                n = len(memory.data.get(inst.src[0], []))
                return c.load_latency + math.ceil(n / c.memory_bandwidth)
            if op == "VSTORE":
                n = len(memory.data.get(inst.dst, [])) or c.vector_length
                return c.store_latency + math.ceil(n / c.memory_bandwidth)
            if op == "VLOAD_STRIDE":
                source = memory.data.get(inst.src[0], [])
                stride = int(inst.src[1])
                n = len(range(0, min(c.vector_length * stride, len(source)), stride))
                transfers = math.ceil(n * stride / max(stride, c.memory_bandwidth)) if n else 0
                return c.load_latency + transfers
            if op == "VGATHER":
                n = len(memory.data.get(inst.src[1], []))
                transfers = math.ceil(n / (c.memory_bandwidth * 0.5)) if n else 0
                return c.load_latency * 2 + transfers
            if op == "VSCATTER":
                n = len(memory.data.get(inst.src[0], []))
                transfers = math.ceil(n / (c.memory_bandwidth * 0.5)) if n else 0
                return c.store_latency * 2 + transfers
        table = {
            "VLOAD": c.load_latency,      "VLOAD_STRIDE": c.load_latency,
            "VGATHER": c.load_latency * 2, "VSTORE": c.store_latency,
            "VSCATTER": c.store_latency * 2,
            "VADD": c.add_latency,  "VSUB": c.add_latency,
            "VMAX": c.add_latency,  "VMIN": c.add_latency,
            "VADDS": c.add_latency, "VSUBS": c.add_latency,
            "VMUL": c.mul_latency,  "VMULS": c.mul_latency,
            "VDIV": c.div_latency,  "VDIVS": c.div_latency,
            "VAND": c.add_latency,  "VOR": c.add_latency, "VXOR": c.add_latency,
            "VREDUCE_SUM": c.reduction_latency,
            "VREDUCE_MAX": c.reduction_latency,
            "VREDUCE_MIN": c.reduction_latency,
            "VDOT": c.mul_latency + c.add_latency,
        }
        return max(1, table.get(op, c.add_latency))

    def _resource_pool(self, fu: str) -> List[int]:
        if not hasattr(self, "_fu_ready"):
            self._fu_ready = {}
        if fu not in self._fu_ready:
            sizes = {
                "MEMORY": self.num_memory_units,
                "ALU": self.num_alu_units,
                "MUL": self.num_mul_units,
                "REDUCTION": self.num_reduction_units,
            }
            self._fu_ready[fu] = [0] * sizes.get(fu, 1)
        return self._fu_ready[fu]

    # ===== Vòng lặp chính =====

    def run(self, instructions, cycle_limit: int = 1000000):
        """Chạy mô phỏng OoO theo chu kỳ.

        Returns:
            Tuple[Dict, List, int]: (memory, timeline, total_cycles)
        """
        self.stats = {"instructions_issued": 0, "instructions_committed": 0, "stalls": 0}

        memory = Memory(dict(self.data),
                        bandwidth=self.config.memory_bandwidth,
                        latency=self.config.load_latency)
        self._fu_ready = {
            "MEMORY": [0] * self.num_memory_units,
            "ALU": [0] * self.num_alu_units,
            "MUL": [0] * self.num_mul_units,
            "REDUCTION": [0] * self.num_reduction_units,
        }
        fc = FunctionalCore(self.scalars)
        reg_values: Dict[str, List[float]] = {}
        scalars = dict(self.scalars)
        mask_registers: Dict[str, List[bool]] = {}

        # ready_at[reg] = cycle khi giá trị của reg có thể dùng được.
        # Phân biệt 3 trạng thái của một source register:
        #   - không có trong dict        → chưa từng bị ghi, dữ liệu sẵn có từ cycle 0
        #   - = PENDING (vô cực)         → đã có producer ISSUE nhưng CHƯA writeback
        #                                  → consumer phải chờ (tránh dependency giả)
        #   - = số cycle cụ thể          → producer đã writeback, sẵn sàng từ cycle đó
        PENDING = float("inf")
        ready_at: Dict[str, float] = {}

        window: List[_Entry] = []       # instruction window, thứ tự chương trình
        committed: List[_Entry] = []    # lệnh đã commit, để build timeline

        pc = 0
        total = len(instructions)
        # Trỏ tới store gần nhất đã được issue (dùng cho memory ordering)
        last_issued_store: Optional[_Entry] = None

        cycle = 0
        while (pc < total or window) and cycle < cycle_limit:

            # ── STAGE 1: COMMIT (in-order, 1 lệnh/cycle) ──────────────────
            # Chỉ commit khi lệnh đầu window đã ở trạng thái WRITEBACK
            if window and window[0].state == "WRITEBACK":
                e = window.pop(0)
                e.state = "COMMIT"
                e.commit_c = cycle
                # Áp dụng hiệu ứng chức năng theo đúng thứ tự chương trình
                self._apply(e.inst, fc, memory, reg_values, scalars, mask_registers)
                committed.append(e)
                self.stats["instructions_committed"] += 1

            # ── STAGE 2: WRITEBACK ─────────────────────────────────────────
            # Lệnh execute xong → chuyển sang WRITEBACK, báo source sẵn sàng
            for e in window:
                if e.state == "EXEC" and cycle >= e.exec_end:
                    e.state = "WRITEBACK"
                    e.wb_c = cycle
                    # Cập nhật ready_at để lệnh phụ thuộc có thể bắt đầu execute
                    if e.inst.dst_register:
                        ready_at[e.inst.dst_register] = cycle

            # ── STAGE 3: EXECUTE (out-of-order) ───────────────────────────
            # Lệnh ISSUE bắt đầu execute ngay khi:
            #   (a) tất cả source register sẵn sàng
            #   (b) memory ordering thỏa (nếu là lệnh memory)
            for e in window:
                if e.state != "ISSUE":
                    continue

                # (a) Chờ source registers
                src_ready = max(
                    (ready_at.get(s, 0) for s in e.inst.get_dependency_registers()),
                    default=0,
                )

                # (b) Chờ memory ordering:
                #     - load: chờ store trước writeback (tránh đọc dữ liệu cũ)
                #     - store: chờ store trước writeback (giữ thứ tự store)
                mem_ok = (
                    e.wait_for is None
                    or e.wait_for.state in ("WRITEBACK", "COMMIT")
                )

                # Lệnh không thể execute cùng cycle với issue (cần ít nhất 1 cycle)
                earliest = max(e.issue_c + 1, src_ready)
                if cycle >= earliest and mem_ok:
                    pool = self._resource_pool(e.fu)
                    unit_free = min(pool)
                    if unit_free > cycle:
                        continue
                    unit_idx = pool.index(unit_free)
                    e.state = "EXEC"
                    e.exec_start = cycle
                    e.exec_end = cycle + e.latency
                    pool[unit_idx] = e.exec_end

            # ── STAGE 4: ISSUE (in-order) ──────────────────────────────────
            # Đưa lệnh tiếp theo vào window nếu còn chỗ
            if pc < total and len(window) < self.window_size:
                inst = instructions[pc]
                fu = self._fu_of(inst.opcode)
                e = _Entry(
                    idx=pc, inst=inst, fu=fu,
                    latency=self._latency(inst, memory),
                    issue_c=cycle,
                )
                # Gán memory ordering tại thời điểm issue
                if inst.is_memory_instruction:
                    if inst.opcode in _STORE_OPS:
                        e.wait_for = last_issued_store   # store chờ store trước
                        last_issued_store = e            # cập nhật store mới nhất
                    else:
                        e.wait_for = last_issued_store   # load chờ store trước

                # Đánh dấu register đích là PENDING: consumer phụ thuộc phải chờ
                # tới khi lệnh này writeback (RAW hazard), không được dùng giá trị cũ.
                if inst.dst_register:
                    ready_at[inst.dst_register] = PENDING

                window.append(e)
                pc += 1
                self.stats["instructions_issued"] += 1
            elif pc < total:
                # Window đầy → stall
                self.stats["stalls"] += 1

            cycle += 1

        total_cycles = cycle

        # Xây dựng timeline theo thứ tự chương trình
        timeline = [
            {
                "instruction": e.inst.raw_text,
                "opcode":      e.inst.opcode,
                "unit":        self._fu_display(e.fu),
                "issue":       e.issue_c,
                "exec_start":  e.exec_start,
                "exec_end":    e.exec_end,
                "writeback":   e.wb_c,
                "commit":      e.commit_c,
                "start":       e.exec_start,
                "end":         e.commit_c,
                "duration":    max(1, e.commit_c - e.issue_c),
                "mode":        "ooo",
            }
            for e in sorted(committed, key=lambda x: x.idx)
        ]

        self.cycle = total_cycles
        return memory.dump(), timeline, total_cycles

    # ===== Áp dụng hiệu ứng chức năng (tuần tự, lúc commit) =====

    def _apply(self, inst, fc: FunctionalCore, memory: Memory,
            reg_values: Dict, scalars: Dict, mask_registers: Dict):
        op = inst.opcode
        fc.scalars = scalars

        if op == "VLOAD":
            mask = mask_registers.get(inst.mask) if inst.mask else None
            vals, _ = memory.load_vector(inst.src[0], mask=mask)
            reg_values[inst.dst] = vals
            return
        if op == "VLOAD_STRIDE":
            vals, _ = memory.load_stride(inst.src[0], int(inst.src[1]),
                                        self.config.vector_length)
            reg_values[inst.dst] = vals
            return
        if op == "VGATHER":
            idx = [int(i) for i in memory.get_array(inst.src[1])]
            mask = mask_registers.get(inst.mask) if inst.mask else None
            vals, _ = memory.gather(inst.src[0], idx, mask=mask)
            reg_values[inst.dst] = vals
            return
        if op == "VSTORE":
            mask = mask_registers.get(inst.mask) if inst.mask else None
            memory.store_vector(inst.dst, reg_values.get(inst.src[0], []), mask=mask)
            return
        if op == "VSCATTER":
            idx = [int(i) for i in memory.get_array(inst.src[0])]
            mask = mask_registers.get(inst.mask) if inst.mask else None
            memory.scatter(inst.dst, idx, reg_values.get(inst.src[1], []), mask=mask)
            return
        if op == "VMASK":
            mask_registers[inst.dst] = fc.mask_threshold(
                reg_values.get(inst.src[0], []), inst.src[1])
            return
        if op == "VMASK_COND":
            mask_registers[inst.dst] = fc.mask_compare(
                reg_values.get(inst.src[0], []),
                reg_values.get(inst.src[1], []), inst.src[2])
            return
        if op in ("VADD_MASKED", "VSUB_MASKED"):
            a    = reg_values.get(inst.src[0], [])
            b    = reg_values.get(inst.src[1], [])
            mask = mask_registers.get(inst.src[2], [True] * len(a))
            reg_values[inst.dst] = fc.masked_binary(op, a, b, mask)
            return
        if op == "VMOV_MASKED":
            src      = reg_values.get(inst.src[0], [])
            mask     = mask_registers.get(inst.src[1], [True] * len(src))
            existing = reg_values.get(inst.dst, [0.0] * len(src))
            reg_values[inst.dst] = fc.masked_move(src, existing, mask)
            return

        mask = mask_registers.get(inst.mask) if inst.mask else None

        if op in ("VADDS", "VSUBS", "VMULS", "VDIVS"):
            kind, res = fc.compute(op, [reg_values.get(inst.src[0], [])], inst.src[1])
        elif op in ("VREDUCE_SUM", "VREDUCE_MAX", "VREDUCE_MIN"):
            values = reg_values.get(inst.src[0], [])
            if mask is not None:
                values = [v for v, m in zip(values, mask) if m]
            kind, res = fc.compute(op, [values])
        elif op == "VDOT" and mask is not None:
            a = reg_values.get(inst.src[0], [])
            b = reg_values.get(inst.src[1], [])
            res = sum(x * y for x, y, m in zip(a, b, mask) if m)
            kind = "scalar"
        else:
            operands = [reg_values.get(s, []) for s in inst.src_registers]
            kind, res = fc.compute(op, operands)

        if mask is not None and kind == "vector":
            old = reg_values.get(inst.dst, [0.0] * len(res))
            if len(old) < len(res):
                old = old + [0.0] * (len(res) - len(old))
            res = [res[i] if i < len(mask) and mask[i] else old[i] for i in range(len(res))]

        if inst.dst is None:
            raise ValueError(f"Instruction {op} does not have a destination")

        if kind == "scalar" or is_scalar_result(op):
            scalars[inst.dst] = res
        else:
            reg_values[inst.dst] = res


# ===== TEST NHANH =====
if __name__ == "__main__":
    from config import VectorProcessorConfig
    from parser import ProgramParser

    cfg = VectorProcessorConfig(vector_length=8, num_lanes=4)
    parser = ProgramParser()
    data, scalars, instructions = parser.parse_file("examples/vector_add.txt")

    sim = OOOSimulator(cfg, data, scalars, num_rs=8)
    memory, timeline, cycles = sim.run(instructions)

    print(f"Total cycles : {cycles}")
    print(f"Committed    : {sim.stats['instructions_committed']}")
    print(f"Stalls       : {sim.stats['stalls']}")
    print(f"Result C     : {memory.get('C')}")
    print("\nTimeline:")
    print(f"  {'Instruction':<25} {'Issue':>5} {'ExecS':>6} {'ExecE':>6} {'WB':>4} {'Commit':>6}")
    print("  " + "-" * 60)
    for t in timeline:
        print(f"  {t['instruction']:<25} {t['issue']:>5} {t['exec_start']:>6} "
            f"{t['exec_end']:>6} {t['writeback']:>4} {t['commit']:>6}")
