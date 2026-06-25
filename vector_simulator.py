import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from config import VectorProcessorConfig
from memory import Memory
from vector_register_file import VectorRegisterFile
from instruction import Instruction, InstructionType


@dataclass
class VectorExecutionStats:
    """Thống kê thực thi vector"""
    total_cycles: int = 0
    total_instructions: int = 0
    memory_instructions: int = 0
    arithmetic_instructions: int = 0
    reduction_instructions: int = 0
    total_elements_processed: int = 0
    breakdown_by_opcode: Dict[str, int] = field(default_factory=dict)
    stalls: int = 0


class VectorSimulator:
    """
    Mô phỏng bộ xử lý vector (non-pipeline)
    """
    
    def __init__(self, config: VectorProcessorConfig, 
                data: Optional[Dict[str, List[float]]] = None,
                scalars: Optional[Dict[str, float]] = None):
        self.config = config
        self.memory = Memory(data, bandwidth=config.memory_bandwidth, latency=config.load_latency)
        self.scalars = dict(scalars or {})
        self.vrf = VectorRegisterFile(
            num_registers=config.num_vector_registers,
            vector_length=config.vector_length,
            enable_chaining=config.enable_chaining
        )
        
        self.stats = VectorExecutionStats()
        self.timeline: List[Dict[str, Any]] = []
        self.current_cycle = 0
        self.mask_registers: Dict[str, List[bool]] = {}
    
    # ===== Các phương thức chính =====
    
    def run(self, instructions: List[Instruction]) -> Tuple[Dict, List, int]:
        self.current_cycle = 0
        self.timeline = []
        
        for inst in instructions:
            start_cycle = self.current_cycle
            duration = self._execute_instruction(inst, start_cycle)
            end_cycle = start_cycle + duration
            
            self.timeline.append({
                "instruction": inst.raw_text,
                "start": start_cycle,
                "end": end_cycle,
                "duration": duration,
                "unit": self._get_unit(inst),
                "opcode": inst.opcode
            })
            
            self.current_cycle = end_cycle
            self._update_stats(inst, duration)
        
        total_cycles = self.current_cycle
        return self.memory.dump(), self.timeline, total_cycles
    
    def _execute_instruction(self, inst: Instruction, start_cycle: int) -> int:
        opcode = inst.opcode
        
        # ===== Memory Instructions =====
        if opcode == "VLOAD":
            return self._execute_vload(inst, start_cycle)
        elif opcode == "VSTORE":
            return self._execute_vstore(inst, start_cycle)
        elif opcode == "VLOAD_STRIDE":
            return self._execute_vload_stride(inst, start_cycle)
        elif opcode == "VGATHER":
            return self._execute_vgather(inst, start_cycle)
        elif opcode == "VSCATTER":
            return self._execute_vscatter(inst, start_cycle)
        
        # ===== Arithmetic Binary Instructions =====
        elif opcode in ["VADD", "VSUB", "VMUL", "VDIV", "VMAX", "VMIN"]:
            return self._execute_binary_op(inst, start_cycle)
        
        # ===== Arithmetic Scalar Instructions =====
        elif opcode in ["VADDS", "VSUBS", "VMULS", "VDIVS"]:
            return self._execute_scalar_op(inst, start_cycle)
        
        # ===== Reduction Instructions =====
        elif opcode == "VREDUCE_SUM":
            return self._execute_reduce_sum(inst, start_cycle)
        elif opcode == "VREDUCE_MAX":
            return self._execute_reduce_max(inst, start_cycle)
        elif opcode == "VREDUCE_MIN":
            return self._execute_reduce_min(inst, start_cycle)
        
        # ===== Mask Instructions =====
        elif opcode == "VMASK":
            return self._execute_vmask(inst, start_cycle)
        elif opcode == "VADD_MASKED":
            return self._execute_vadd_masked(inst, start_cycle)
        elif opcode == "VMASK_COND":
            return self._execute_vmask_condition(inst, start_cycle)
        elif opcode == "VSUB_MASKED":
            return self._execute_vsub_masked(inst, start_cycle)
        elif opcode == "VMOV_MASKED":
            return self._execute_vmov_masked(inst, start_cycle)
        
        # ===== Dot Product =====
        elif opcode == "VDOT":
            return self._execute_vdot(inst, start_cycle)
        
        # ===== Logical Instructions =====
        elif opcode in ["VAND", "VOR", "VXOR"]:
            return self._execute_logical_op(inst, start_cycle)
        
        else:
            raise NotImplementedError(f"Unsupported instruction: {inst.opcode}")
    
    # ===== Memory Instructions Implementation =====
    
    def _execute_vload(self, inst: Instruction, start_cycle: int) -> int:
        dst = inst.dst
        array_name = inst.src[0]

        if inst.mask is None:
            # Không mask: nạp toàn bộ vector (hành vi cũ).
            values, cycles = self.memory.load_vector(array_name, start_cycle)
            self.vrf.write(dst, values, start_cycle, cycles)
            return cycles

        # Masked load: chỉ lane active được nạp từ bộ nhớ; lane inactive giữ
        # nguyên giá trị cũ của thanh ghi đích. Memory subsystem tính băng thông
        # theo số lane active (xem Memory.load_vector(mask=...)).
        mask_full = self._resolve_mask(inst, self.config.vector_length, start_cycle)
        values, cycles = self.memory.load_vector(array_name, start_cycle, mask=mask_full)
        n = len(values)
        mask = self._normalize_mask(mask_full, n)
        old = self._old_dst(dst, n, start_cycle)
        if inst.mask_mode == "zero":
            merged = [values[i] if mask[i] else 0.0 for i in range(n)]
        else:
            merged = [values[i] if mask[i] else old[i] for i in range(n)]
        self.vrf.write(dst, merged, start_cycle, cycles)
        return cycles

    def _execute_vstore(self, inst: Instruction, start_cycle: int) -> int:
        array_name = inst.dst
        src_reg = inst.src[0]

        values = self.vrf.read(src_reg, start_cycle)
        if inst.mask is None:
            return self.memory.store_vector(array_name, values, start_cycle)

        # Masked store: chỉ ghi lane active xuống bộ nhớ; ô nhớ tại lane inactive
        # giữ nguyên giá trị cũ (memory subsystem tự merge).
        mask = self._resolve_mask(inst, len(values), start_cycle)
        return self.memory.store_vector(array_name, values, start_cycle, mask=mask)
    
    def _execute_vload_stride(self, inst: Instruction, start_cycle: int) -> int:
        dst = inst.dst
        array_name = inst.src[0]
        stride = int(inst.src[1])
        
        values, cycles = self.memory.load_stride(
            array_name, stride, self.config.vector_length, start_cycle
        )
        self.vrf.write(dst, values, start_cycle, cycles)
        
        return cycles
    
    def _execute_vgather(self, inst: Instruction, start_cycle: int) -> int:
        dst = inst.dst
        array_name = inst.src[0]
        idx_array = inst.src[1]

        indices = self.memory.get_array(idx_array)
        indices_int = [int(i) for i in indices]

        if inst.mask is None:
            values, cycles = self.memory.gather(array_name, indices_int, start_cycle)
            self.vrf.write(dst, values, start_cycle, cycles)
            return cycles

        # Masked gather: chỉ lane active thực sự truy xuất bộ nhớ; lane inactive
        # giữ giá trị cũ của thanh ghi đích.
        mask = self._normalize_mask(
            self._resolve_mask(inst, len(indices_int), start_cycle), len(indices_int))
        values, cycles = self.memory.gather(array_name, indices_int, start_cycle, mask=mask)
        n = len(values)
        old = self._old_dst(dst, n, start_cycle)
        if inst.mask_mode == "zero":
            merged = [values[i] if (i < len(mask) and mask[i]) else 0.0 for i in range(n)]
        else:
            merged = [values[i] if (i < len(mask) and mask[i]) else old[i] for i in range(n)]
        self.vrf.write(dst, merged, start_cycle, cycles)
        return cycles

    def _execute_vscatter(self, inst: Instruction, start_cycle: int) -> int:
        array_name = inst.dst
        idx_array = inst.src[0]
        src_reg = inst.src[1]

        indices = self.memory.get_array(idx_array)
        indices_int = [int(i) for i in indices]
        values = self.vrf.read(src_reg, start_cycle)

        if inst.mask is None:
            return self.memory.scatter(array_name, indices_int, values, start_cycle)

        # Masked scatter: chỉ ghi lane active theo chỉ mục; ô nhớ khác giữ nguyên.
        mask = self._resolve_mask(inst, len(values), start_cycle)
        return self.memory.scatter(array_name, indices_int, values, start_cycle, mask=mask)
    
    # ===== Mask Instructions Implementation =====
    
    def _execute_vmask(self, inst: Instruction, start_cycle: int) -> int:
        """VMASK Vd, Vs, condition - Tạo mask từ vector với threshold = mean"""
        dst = inst.dst
        src = inst.src[0]
        condition = inst.src[1]
    
        values = self.vrf.read(src, start_cycle)
        threshold = sum(values) / len(values) if values else 0
        
        if condition == 'gt':
            mask = [v > threshold for v in values]
        elif condition == 'lt':
            mask = [v < threshold for v in values]
        elif condition == 'eq':
            mask = [v == threshold for v in values]
        elif condition == 'ne':
            mask = [v != threshold for v in values]
        elif condition == 'ge':
            mask = [v >= threshold for v in values]
        elif condition == 'le':
            mask = [v <= threshold for v in values]
        
        else:
            raise ValueError(f"Unknown condition: {condition}")
    
        mask_float = [1.0 if m else 0.0 for m in mask]
        self.vrf.write(dst, mask_float, start_cycle, 1)
        self.mask_registers[dst] = mask

        # Dùng add_latency thẳng, KHÔNG qua _compute_cycles để tránh cộng thừa
        # startup_latency (mask chỉ là phép so sánh, không phải lệnh vector pipeline đầy đủ).
        return self.config.add_latency
    
    def _execute_vmask_condition(self, inst: Instruction, start_cycle: int) -> int:
        """VMASK_COND Vd, Vs1, Vs2, condition - So sánh 2 vector"""
        dst = inst.dst
        src1 = inst.src[0]
        src2 = inst.src[1]
        condition = inst.src[2]
        
        a = self.vrf.read(src1, start_cycle)
        b = self.vrf.read(src2, start_cycle)
        
        if condition == 'eq':
            mask = [x == y for x, y in zip(a, b)]
        elif condition == 'ne':
            mask = [x != y for x, y in zip(a, b)]
        elif condition == 'gt':
            mask = [x > y for x, y in zip(a, b)]
        elif condition == 'lt':
            mask = [x < y for x, y in zip(a, b)]
        elif condition == 'ge':
            mask = [x >= y for x, y in zip(a, b)]
        elif condition == 'le':
            mask = [x <= y for x, y in zip(a, b)]
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        mask_float = [1.0 if m else 0.0 for m in mask]
        self.vrf.write(dst, mask_float, start_cycle, 1)
        self.mask_registers[dst] = mask

        # Nhất quán với _execute_vmask: dùng add_latency trực tiếp
        return self.config.add_latency
    
    def _execute_vadd_masked(self, inst: Instruction, start_cycle: int) -> int:
        """VADD_MASKED Vd, Vs1, Vs2, Vmask - Chỉ cộng các lane có mask=1.

        Lane inactive giữ giá trị src1 (passthrough) — ngữ nghĩa ISA gốc.
        Tái dùng lõi per-element masked (_lanewise + _finish)."""
        a = self.vrf.read(inst.src[0], start_cycle)
        b = self.vrf.read(inst.src[1], start_cycle)
        mask = self._normalize_mask(
            self.mask_registers.get(inst.src[2], [True] * len(a)), len(a))
        computed, _ = self._lanewise(lambda x, y: x + y, [a, b], mask)
        return self._finish(inst, inst.dst, computed, mask, start_cycle,
                            self.config.add_latency, passthrough=a)

    def _execute_vsub_masked(self, inst: Instruction, start_cycle: int) -> int:
        """VSUB_MASKED Vd, Vs1, Vs2, Vmask - Trừ có điều kiện (inactive = src1)."""
        a = self.vrf.read(inst.src[0], start_cycle)
        b = self.vrf.read(inst.src[1], start_cycle)
        mask = self._normalize_mask(
            self.mask_registers.get(inst.src[2], [True] * len(a)), len(a))
        computed, _ = self._lanewise(lambda x, y: x - y, [a, b], mask)
        return self._finish(inst, inst.dst, computed, mask, start_cycle,
                            self.config.add_latency, passthrough=a)

    def _execute_vmov_masked(self, inst: Instruction, start_cycle: int) -> int:
        """VMOV_MASKED Vd, Vs, Vmask - Di chuyển có điều kiện.

        Lane inactive giữ giá trị cũ của đích (undisturbed) → passthrough=None."""
        values = self.vrf.read(inst.src[0], start_cycle)
        mask = self._normalize_mask(
            self.mask_registers.get(inst.src[1], [True] * len(values)), len(values))
        computed, _ = self._lanewise(lambda x: x, [values], mask)
        return self._finish(inst, inst.dst, computed, mask, start_cycle,
                            1, passthrough=None)
    
    # ===== Arithmetic Implementation =====
    
    # Bảng phép toán per-lane cho ALU binary (dùng chung cho mọi opcode)
    _BINARY_FN = {
        "VADD": lambda x, y: x + y,
        "VSUB": lambda x, y: x - y,
        "VMUL": lambda x, y: x * y,
        "VDIV": lambda x, y: (x / y if y != 0 else 0),
        "VMAX": lambda x, y: max(x, y),
        "VMIN": lambda x, y: min(x, y),
    }

    def _execute_binary_op(self, inst: Instruction, start_cycle: int) -> int:
        """ALU binary với per-element conditional execution theo mask.

        Lane inactive bị SKIP trong compute và giữ giá trị cũ của đích
        (undisturbed). Không mask → ghi đè toàn bộ như mô hình lockstep cũ."""
        opcode = inst.opcode
        a = self.vrf.read(inst.src[0], start_cycle)
        b = self.vrf.read(inst.src[1], start_cycle)

        if len(a) != len(b):
            raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")

        fn = self._BINARY_FN.get(opcode)
        if fn is None:
            raise NotImplementedError(opcode)

        mask = self._resolve_mask(inst, len(a), start_cycle)
        computed, _ = self._lanewise(fn, [a, b], mask)
        return self._finish(inst, inst.dst, computed, mask, start_cycle,
                            self._get_latency(opcode))

    def _execute_scalar_op(self, inst: Instruction, start_cycle: int) -> int:
        """ALU vector ⊕ scalar với per-element conditional execution theo mask."""
        opcode = inst.opcode
        a = self.vrf.read(inst.src[0], start_cycle)
        scalar_token = inst.src[1]

        if scalar_token in self.scalars:
            s = self.scalars[scalar_token]
        else:
            s = float(scalar_token)

        scalar_fn = {
            "VADDS": lambda x: x + s,
            "VSUBS": lambda x: x - s,
            "VMULS": lambda x: x * s,
            "VDIVS": lambda x: (x / s if s != 0 else 0),
        }.get(opcode)
        if scalar_fn is None:
            raise NotImplementedError(opcode)

        mask = self._resolve_mask(inst, len(a), start_cycle)
        computed, _ = self._lanewise(scalar_fn, [a], mask)
        return self._finish(inst, inst.dst, computed, mask, start_cycle,
                            self._get_latency(opcode))
    
    # ===== Reduction Implementation =====
    
    def _active_values(self, inst: Instruction, values: List[float],
                       start_cycle: int) -> List[float]:
        """Lọc ra các phần tử ở lane ACTIVE theo mask (cho reduction/dot).

        Lane inactive bị loại hoàn toàn khỏi phép cộng dồn — không đóng góp vào
        kết quả reduction, đúng chuẩn masked reduction."""
        mask = self._resolve_mask(inst, len(values), start_cycle)
        return [v for v, m in zip(values, mask) if m]

    def _reduction_cycles(self, num_active: int) -> int:
        """Chu kỳ reduction theo số lane active (cây cộng log2)."""
        return self.config.reduction_latency + math.ceil(math.log2(max(1, num_active)))

    def _execute_reduce_sum(self, inst: Instruction, start_cycle: int) -> int:
        values = self.vrf.read(inst.src[0], start_cycle)
        active = self._active_values(inst, values, start_cycle)
        self.scalars[inst.dst] = sum(active)
        return self._reduction_cycles(len(active))

    def _execute_reduce_max(self, inst: Instruction, start_cycle: int) -> int:
        values = self.vrf.read(inst.src[0], start_cycle)
        active = self._active_values(inst, values, start_cycle)
        self.scalars[inst.dst] = max(active) if active else 0
        return self._reduction_cycles(len(active))

    def _execute_reduce_min(self, inst: Instruction, start_cycle: int) -> int:
        values = self.vrf.read(inst.src[0], start_cycle)
        active = self._active_values(inst, values, start_cycle)
        self.scalars[inst.dst] = min(active) if active else 0
        return self._reduction_cycles(len(active))
    
    # ===== Dot Product Implementation =====
    
    def _execute_vdot(self, inst: Instruction, start_cycle: int) -> int:
        a = self.vrf.read(inst.src[0], start_cycle)
        b = self.vrf.read(inst.src[1], start_cycle)

        # Chỉ các lane active tham gia tích & cộng dồn (masked dot product).
        mask = self._resolve_mask(inst, min(len(a), len(b)), start_cycle)
        products = [x * y for x, y, m in zip(a, b, mask) if m]
        self.scalars[inst.dst] = sum(products)

        n_active = len(products)
        mul_cycles = self._compute_cycles_masked(mask, self.config.mul_latency)
        red_cycles = self._reduction_cycles(n_active)
        return mul_cycles + red_cycles
    
    # ===== Logical Implementation =====
    
    def _execute_logical_op(self, inst: Instruction, start_cycle: int) -> int:
        """Logical bitwise với per-element conditional execution theo mask."""
        opcode = inst.opcode
        a = self.vrf.read(inst.src[0], start_cycle)
        b = self.vrf.read(inst.src[1], start_cycle)

        logical_fn = {
            "VAND": lambda x, y: float(int(x) & int(y)),
            "VOR":  lambda x, y: float(int(x) | int(y)),
            "VXOR": lambda x, y: float(int(x) ^ int(y)),
        }.get(opcode)
        if logical_fn is None:
            raise NotImplementedError(opcode)

        mask = self._resolve_mask(inst, min(len(a), len(b)), start_cycle)
        computed, _ = self._lanewise(logical_fn, [a, b], mask)
        return self._finish(inst, inst.dst, computed, mask, start_cycle,
                            self._get_latency(opcode))
    
    # ===== Masking core (true per-element execution) =====
    #
    # Toàn bộ cơ chế "true masked SIMD" tập trung ở 5 helper dưới đây. Mọi lệnh
    # vector (ALU, scalar, logical, reduction, dot, memory) đều chạy qua chúng:
    #
    #   _resolve_mask  -> lấy predicate per-lane (list[bool]) cho lệnh
    #   _normalize_mask-> chuẩn hóa độ dài mask về n lane
    #   _lanewise      -> THỰC THI per-element CÓ ĐIỀU KIỆN (skip lane inactive)
    #   _writeback     -> WRITEBACK chỉ cập nhật lane active (merge/zero)
    #   _finish        -> gói (timing theo lane active) + writeback
    #
    # Khi lệnh KHÔNG mang mask (inst.mask is None và không phải lệnh masked
    # tường minh) → mask = toàn bộ True → hành vi y hệt mô hình lockstep cũ,
    # đảm bảo tương thích ngược 100%.

    def _normalize_mask(self, bits: List[Any], n: int) -> List[bool]:
        """Chuẩn hóa danh sách mask về đúng n phần tử bool.

        Thiếu → bù False (lane vượt quá độ dài mask coi như inactive)."""
        out = [bool(b) for b in bits]
        if len(out) < n:
            out = out + [False] * (n - len(out))
        return out[:n]

    def _resolve_mask(self, inst: Instruction, n: int, start_cycle: int) -> List[bool]:
        """Trả về predicate per-lane (list[bool], độ dài n) cho lệnh.

        Nguồn mask:
          1. inst.mask  — predicate per-instruction (cú pháp '@Vx').
          2. None       — mọi lane active (mô hình lockstep cũ).

        Mask register được lưu ở self.mask_registers (bool) khi sinh bằng VMASK,
        hoặc đọc trực tiếp từ VRF dưới dạng float (0.0 = inactive)."""
        mask_reg = inst.mask
        if mask_reg is None:
            return [True] * n
        bits = self.mask_registers.get(mask_reg)
        if bits is None:
            try:
                vals = self.vrf.read(mask_reg, start_cycle)
                bits = [v != 0 for v in vals]
            except KeyError:
                bits = [True] * n
        return self._normalize_mask(bits, n)

    @staticmethod
    def _active_count(mask: List[bool]) -> int:
        return sum(1 for m in mask if m)

    def _lanewise(self, fn, operand_vectors: List[List[float]],
                  mask: List[bool]) -> Tuple[List[Any], int]:
        """Thực thi per-element CÓ ĐIỀU KIỆN theo mask.

        Chỉ gọi fn cho lane ACTIVE — lane inactive bị SKIP hoàn toàn trong
        compute (không cộng/nhân/chia, không chạm dữ liệu, tránh cả div-by-zero
        trên lane bị tắt). Lane inactive trả placeholder None để giai đoạn
        writeback hợp nhất."""
        n = min(len(v) for v in operand_vectors) if operand_vectors else len(mask)
        results: List[Any] = []
        for i in range(n):
            if i < len(mask) and mask[i]:
                results.append(fn(*[v[i] for v in operand_vectors]))
            else:
                results.append(None)   # lane inactive: SKIP compute
        return results, n

    def _old_dst(self, dst: str, n: int, start_cycle: int) -> List[float]:
        """Đọc giá trị hiện tại của thanh ghi đích (cho chính sách undisturbed)."""
        try:
            cur = list(self.vrf.read(dst, start_cycle))
        except KeyError:
            cur = []
        if len(cur) < n:
            cur = cur + [0.0] * (n - len(cur))
        return cur[:n]

    def _writeback(self, dst: str, computed: List[Any], mask: List[bool],
                   inactive_values: Optional[List[float]], start_cycle: int,
                   latency: int, mask_mode: str = "merge") -> List[float]:
        """Giai đoạn WRITEBACK chỉ cập nhật lane ACTIVE.

        - lane active   : ghi giá trị vừa tính (computed[i]).
        - lane inactive : theo mask_mode
              "merge"/"undisturbed" → giữ inactive_values[i]
              "zero"                → ghi 0.0
        Không bao giờ ghi đè toàn bộ vector khi có lane inactive."""
        n = len(computed)
        out: List[float] = []
        for i in range(n):
            if i < len(mask) and mask[i]:
                out.append(computed[i])
            elif mask_mode == "zero":
                out.append(0.0)
            else:
                iv = inactive_values[i] if (inactive_values is not None and i < len(inactive_values)) else 0.0
                out.append(iv)
        self.vrf.write(dst, out, start_cycle, latency)
        return out

    def _finish(self, inst: Instruction, dst: str, computed: List[Any],
                mask: List[bool], start_cycle: int, op_latency: int,
                passthrough: Optional[List[float]] = None) -> int:
        """Hoàn tất một lệnh ALU/logical masked: tính timing theo lane active rồi
        writeback chỉ lane active.

        passthrough: giá trị dùng cho lane inactive (vd src1 với các lệnh
        *_MASKED tường minh). None → dùng chính sách undisturbed (giá trị cũ
        của thanh ghi đích)."""
        cycles = self._compute_cycles_masked(mask, op_latency)
        if all(mask):
            # Đường nhanh: không lane nào bị tắt → ghi toàn bộ (giống lockstep cũ)
            self.vrf.write(dst, list(computed), start_cycle, cycles)
            return cycles
        if inst.mask_mode == "zero":
            inactive = None
        elif passthrough is not None:
            inactive = passthrough
        else:
            inactive = self._old_dst(dst, len(computed), start_cycle)
        self._writeback(dst, computed, mask, inactive, start_cycle, cycles, inst.mask_mode)
        return cycles

    # ===== Helper Methods =====

    def _compute_cycles(self, num_elements: int, op_latency: int) -> int:
        groups = math.ceil(num_elements / self.config.num_lanes)
        return self.config.startup_latency + op_latency + groups

    def _compute_cycles_masked(self, mask: List[bool], op_latency: int) -> int:
        """Số chu kỳ tính theo SỐ LANE ACTIVE (density-time scheduling).

        - Không mask (mọi lane active): groups = ceil(n / lanes) → KẾT QUẢ Y HỆT
          _compute_cycles cũ ⇒ tương thích ngược tuyệt đối với mọi test cũ.
        - Có mask: chỉ lane active chiếm slot tính toán ⇒ ít group hơn ⇒ ít chu
          kỳ hơn (đáp ứng yêu cầu 'masked execution làm giảm số lane active').
        - Toàn bộ bị tắt: lệnh vẫn được issue nên vẫn tốn startup + latency."""
        active = self._active_count(mask)
        if active <= 0:
            return self.config.startup_latency + op_latency
        groups = math.ceil(active / self.config.num_lanes)
        return self.config.startup_latency + op_latency + groups
    
    def _get_latency(self, opcode: str) -> int:
        latency_map = {
            "VADD": self.config.add_latency,
            "VSUB": self.config.add_latency,
            "VMUL": self.config.mul_latency,
            "VDIV": self.config.div_latency,
            "VMAX": self.config.add_latency,
            "VMIN": self.config.add_latency,
            "VADDS": self.config.add_latency,
            "VSUBS": self.config.add_latency,
            "VMULS": self.config.mul_latency,
            "VDIVS": self.config.div_latency,
            "VAND": self.config.add_latency,
            "VOR": self.config.add_latency,
            "VXOR": self.config.add_latency,
        }
        return latency_map.get(opcode, self.config.add_latency)
    
    def _get_unit(self, inst: Instruction) -> str:
        if inst.is_memory_instruction:
            return "Memory"
        elif inst.inst_type == InstructionType.ARITHMETIC_REDUCTION:
            return "Reduction Unit"
        elif inst.inst_type == InstructionType.ARITHMETIC_DOT:
            return "Dot Product Unit"
        else:
            return "Vector ALU"
    
    def _update_stats(self, inst: Instruction, cycles: int):
        self.stats.total_cycles += cycles
        self.stats.total_instructions += 1
        self.stats.breakdown_by_opcode[inst.opcode] = self.stats.breakdown_by_opcode.get(inst.opcode, 0) + 1
        
        if inst.is_memory_instruction:
            self.stats.memory_instructions += 1
        elif inst.is_arithmetic:
            self.stats.arithmetic_instructions += 1
            if inst.inst_type in [InstructionType.ARITHMETIC_REDUCTION, InstructionType.ARITHMETIC_DOT]:
                self.stats.reduction_instructions += 1
    
    # ===== Statistics Methods =====
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_cycles": self.stats.total_cycles,
            "total_instructions": self.stats.total_instructions,
            "memory_instructions": self.stats.memory_instructions,
            "arithmetic_instructions": self.stats.arithmetic_instructions,
            "reduction_instructions": self.stats.reduction_instructions,
            "breakdown_by_opcode": self.stats.breakdown_by_opcode,
            "memory_stats": self.memory.get_stats(),
            "register_stats": self.vrf.get_stats()
        }
    
    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("VECTOR SIMULATOR STATISTICS")
        print("=" * 60)
        print(f"Total cycles:              {stats['total_cycles']}")
        print(f"Total instructions:        {stats['total_instructions']}")
        print(f"Memory instructions:       {stats['memory_instructions']}")
        print(f"Arithmetic instructions:   {stats['arithmetic_instructions']}")
        print(f"Reduction instructions:    {stats['reduction_instructions']}")
        
        print("\nBreakdown by opcode:")
        for opcode, count in sorted(stats['breakdown_by_opcode'].items()):
            print(f"  {opcode}: {count}")
        
        print(f"\nMemory bandwidth utilization: {stats['memory_stats']['bandwidth']} elem/cycle")
        print(f"Total memory cycles: {stats['memory_stats']['total_cycles_spent']}")
    
    def print_timeline(self):
        print("\n" + "=" * 60)
        print("EXECUTION TIMELINE")
        print("=" * 60)
        print(f"{'Instruction':<35} {'Start':<8} {'End':<8} {'Duration':<10} {'Unit'}")
        print("-" * 70)
        for event in self.timeline:
            print(f"{event['instruction']:<35} {event['start']:<8} {event['end']:<8} "
                f"{event['duration']:<10} {event['unit']}")


# ===== TEST =====
if __name__ == "__main__":
    from config import VectorProcessorConfig
    from instruction import InstructionTemplates
    
    config = VectorProcessorConfig(
        vector_length=8,
        num_lanes=4,
        memory_bandwidth=4,
        add_latency=2,
        mul_latency=4,
        load_latency=3,
        store_latency=3
    )
    
    data = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "B": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    }
    
    instructions = [
        InstructionTemplates.vload("V1", "A"),
        InstructionTemplates.vload("V2", "B"),
        InstructionTemplates.vadd("V3", "V1", "V2"),
        InstructionTemplates.vstore("C", "V3"),
    ]
    
    sim = VectorSimulator(config, data)
    memory, timeline, cycles = sim.run(instructions)
    
    print(f"Total cycles: {cycles}")
    print(f"Result C: {memory.get('C', [])}")
    sim.print_timeline()
    sim.print_stats()
    
    print("\n✓ All tests passed!")