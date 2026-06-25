import math
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from config import VectorProcessorConfig
from memory import Memory
from vector_register_file import VectorRegisterFile
from instruction import Instruction, InstructionType, FunctionalUnit


class HazardType(Enum):
    """Các loại hazard trong pipeline"""
    NONE = "none"
    RAW = "raw"              # Read After Write
    WAR = "war"              # Write After Read
    WAW = "waw"              # Write After Write
    STRUCTURAL = "structural"  # Xung đột tài nguyên


@dataclass
class PipelineStage:
    """Thông tin về một pipeline stage"""
    name: str
    is_busy: bool = False
    current_instruction: Optional[Instruction] = None
    start_cycle: int = 0
    end_cycle: int = 0
    functional_unit: Optional[FunctionalUnit] = None


@dataclass
class PipelineEvent:
    """Sự kiện trong pipeline"""
    cycle: int
    instruction: str
    stage: str
    event_type: str  # "issue", "execute", "complete", "stall"


class PipelineSimulator:
    """
    Mô phỏng bộ xử lý vector với pipeline
    
    Hỗ trợ:
    - Pipeline stages: Issue -> Execute -> Write Back
    - RAW dependency detection và stall
    - Structural hazard (xung đột tài nguyên)
    - Chaining (forwarding) để giảm stall
    - Memory bandwidth constraints
    - Timeline chi tiết
    """
    
    def __init__(self, config: VectorProcessorConfig,
                data: Optional[Dict[str, List[float]]] = None,
                scalars: Optional[Dict[str, float]] = None):
        """
        Khởi tạo pipeline simulator
        
        Args:
            config: Cấu hình bộ xử lý
            data: Dữ liệu bộ nhớ
            scalars: Các biến scalar
        """
        self.config = config
        self.memory = Memory(data, bandwidth=config.memory_bandwidth, latency=config.load_latency)
        self.scalars = dict(scalars or {})
        self.vrf = VectorRegisterFile(
            num_registers=config.num_vector_registers,
            vector_length=config.vector_length,
            enable_chaining=config.enable_chaining
        )
        
        # Pipeline resources
        self.resource_ready = {
            FunctionalUnit.MEMORY: 0,
            FunctionalUnit.ALU: 0,
            FunctionalUnit.REDUCTION: 0
        }
        
        # Số lượng đơn vị tài nguyên (có thể có nhiều hơn 1)
        self.num_memory_units = getattr(config, 'num_memory_units', 1)
        self.num_alu_units = getattr(config, 'num_alu_units', 1)
        
        # Theo dõi các đơn vị tài nguyên
        self.memory_units = [0] * self.num_memory_units  # cycles ready
        self.alu_units = [0] * self.num_alu_units
        
        # Register readiness tracking
        self.register_ready: Dict[str, int] = {}
        for i in range(config.num_vector_registers):
            self.register_ready[f"V{i}"] = 0
        
        # Chaining (forwarding) support
        self.forwarding_data: Dict[str, Tuple[int, List[float]]] = {}
        
        # Timeline và events
        self.timeline: List[Dict] = []
        self.stage_timeline: List[Dict] = []   # chi tiết IF/ID/EX/MEM/WB từng lệnh
        self.events: List[PipelineEvent] = []

        # Mask registers — khởi tạo ở đây để tránh AttributeError trước khi run()
        self.mask_registers: Dict[str, List[bool]] = {}
        
        # Thống kê
        self.stats = {
            "total_cycles": 0,
            "total_instructions": 0,
            "stall_cycles": 0,
            "raw_hazards": 0,
            "structural_hazards": 0,
            "forwarding_used": 0
        }
        
        self.current_cycle = 0
    
    # ===== Phương thức chính =====
    
    def run(self, instructions: List[Instruction]) -> Tuple[Dict, List, int]:
        """
        Chạy pipeline simulation 5-stage (IF → ID → EX → MEM → WB).

        Mỗi lệnh được đẩy qua đầy đủ 5 stage vật lý. Timing tôn trọng:
        - Front-end in-order (fetch/decode 1 lệnh/chu kỳ).
        - RAW hazard + chaining/forwarding (operand ready).
        - Structural hazard: số đơn vị MEMORY/ALU/REDUCTION hữu hạn (đơn vị
        không pipeline — bận trong suốt thời gian thực thi của lệnh).
        - MEM stage dùng số chu kỳ truy xuất thực tế từ memory hierarchy (gồm
        cache miss penalty).

        Returns:
            Tuple[Dict, List, int]: (memory, timeline, total_cycles)
        """
        self.current_cycle = 0
        self.timeline = []
        self.stage_timeline = []        # chi tiết từng stage cho trực quan hóa
        self.mask_registers = {}        # reset mỗi lần run
        self.events = []

        # Reset tài nguyên & trạng thái
        self.memory_units = [0] * self.num_memory_units
        self.alu_units = [0] * self.num_alu_units
        self.resource_ready = {fu: 0 for fu in self.resource_ready}
        self.register_ready = {f"V{i}": 0 for i in range(self.config.num_vector_registers)}
        # reg_avail[reg] = chu kỳ mà giá trị của reg sẵn sàng cho lệnh phụ thuộc
        reg_avail: Dict[str, int] = {f"V{i}": 0 for i in range(self.config.num_vector_registers)}

        self.stats = {k: 0 for k in self.stats}

        prev_if = -1
        prev_id = -1
        last_complete = 0

        for inst in instructions:
            unit = self._get_functional_unit(inst)
            is_mem = inst.is_memory_instruction

            # ---- Front-end: IF, ID (in-order, 1 lệnh/chu kỳ) ----
            if_c = prev_if + 1
            id_c = max(if_c, prev_id + 1)
            ex_ready = id_c           # có thể bắt đầu EX ngay sau decode

            # ---- RAW hazard: chờ operand sẵn sàng ----
            operand_ready = 0
            forwarded = 0
            for src in inst.get_dependency_registers():
                av = reg_avail.get(src, 0)
                operand_ready = max(operand_ready, av)
                # Forwarding chỉ được tính khi: (1) chaining bật, VÀ
                # (2) operand chưa sẵn sàng ngay sau decode (av > ex_ready),
                # tức là chaining thực sự đang cắt giảm stall.
                if self.config.enable_chaining and av > ex_ready:
                    forwarded += 1
            if operand_ready > ex_ready:
                self.stats["raw_hazards"] += 1
            if forwarded:
                self.stats["forwarding_used"] += forwarded

            # ---- Functional execution (program order) ----
            # Thực hiện sớm để lấy số chu kỳ MEM thực tế; kết quả không phụ
            # thuộc giá trị chu kỳ nên an toàn về mặt chức năng.
            mem_cycles_actual = self._execute_instruction(inst, max(ex_ready, operand_ready), 0)

            if is_mem:
                # Memory op: EX (address-gen) gộp vào trước MEM (0 chu kỳ timing),
                # MEM = truy xuất bộ nhớ thật (gồm cache penalty).
                ex_start = max(ex_ready, operand_ready)
                pool = self.memory_units
                unit_free = min(pool)
                idx = pool.index(unit_free)
                if unit_free > ex_start:
                    self.stats["structural_hazards"] += 1
                mem_start = max(ex_start, unit_free)
                mem_cycles = max(1, int(mem_cycles_actual or 1))
                mem_end = mem_start + mem_cycles
                pool[idx] = mem_end          # đơn vị memory bận tới hết MEM
                ex_end = mem_start           # hiển thị EX phủ addr-gen + chờ MEM
                wb_c = mem_end
                avail = mem_end              # load: dữ liệu sẵn sàng sau MEM
                stall = mem_start - ex_start
            else:
                # Compute op: EX nhiều chu kỳ, MEM passthrough (0 chu kỳ).
                latency = self._get_instruction_latency(inst)
                if unit == FunctionalUnit.ALU:
                    pool = self.alu_units
                    unit_free = min(pool)
                    idx = pool.index(unit_free)
                    ex_start = max(ex_ready, operand_ready, unit_free)
                    if unit_free > max(ex_ready, operand_ready):
                        self.stats["structural_hazards"] += 1
                    ex_end = ex_start + latency
                    pool[idx] = ex_end
                else:  # REDUCTION (đơn vị đơn)
                    unit_free = self.resource_ready[unit]
                    ex_start = max(ex_ready, operand_ready, unit_free)
                    if unit_free > max(ex_ready, operand_ready):
                        self.stats["structural_hazards"] += 1
                    ex_end = ex_start + latency
                    self.resource_ready[unit] = ex_end
                mem_start = ex_end
                mem_end = ex_end             # MEM passthrough (không tốn chu kỳ)
                wb_c = ex_end
                avail = ex_end               # chaining: forward từ cuối EX
                stall = ex_start - max(ex_ready, operand_ready)

            if not self.config.enable_chaining:
                avail = wb_c + 1             # không chaining: chờ tới sau WB

            if stall > 0:
                self.stats["stall_cycles"] += stall

            if inst.dst_register:
                reg_avail[inst.dst_register] = avail
                self.register_ready[inst.dst_register] = wb_c

            last_complete = max(last_complete, wb_c)

            # ---- Ghi timeline (giữ key cũ) + stage_timeline (mới) ----
            duration = (mem_end - ex_start) if is_mem else (ex_end - ex_start)
            self.timeline.append({
                "instruction": inst.raw_text,
                "opcode": inst.opcode,
                "start": ex_start,
                "end": wb_c,
                "duration": duration,
                "unit": unit.value,
                "issue_cycle": id_c,
                "raw_stall": max(0, operand_ready - ex_ready),
            })
            self.stage_timeline.append({
                "instruction": inst.raw_text,
                "opcode": inst.opcode,
                "unit": unit.value,
                "IF": if_c,
                "ID": id_c,
                "EX_start": ex_start,
                "EX_end": ex_end,
                "MEM_start": mem_start,
                "MEM_end": mem_end,
                "WB": wb_c,
                "stall": max(0, stall),
            })

            prev_if = if_c
            prev_id = id_c

        total_cycles = last_complete
        self.current_cycle = total_cycles
        self.stats["total_cycles"] = total_cycles
        self.stats["total_instructions"] = len(instructions)

        return self.memory.dump(), self.timeline, total_cycles


    def _execute_instruction(self, inst: Instruction, start_cycle: int, end_cycle: int):
        """
        Thực thi lệnh (tính toán kết quả).

        Returns:
            int | None: số chu kỳ truy xuất bộ nhớ (cho lệnh memory, dùng cho
            MEM stage), hoặc None với lệnh không truy xuất bộ nhớ.
        """
        opcode = inst.opcode

        try:
            # Memory instructions
            if opcode == "VLOAD":
                return self._exec_vload(inst, start_cycle)
            elif opcode == "VSTORE":
                return self._exec_vstore(inst, start_cycle)
            elif opcode == "VLOAD_STRIDE":
                return self._exec_vload_stride(inst, start_cycle)
            elif opcode == "VGATHER":
                return self._exec_vgather(inst, start_cycle)
            elif opcode == "VSCATTER":
                return self._exec_vscatter(inst, start_cycle)
            
            # Arithmetic binary
            elif opcode in ["VADD", "VSUB", "VMUL", "VDIV", "VMAX", "VMIN"]:
                self._exec_binary_op(inst, start_cycle, end_cycle)
            
            # Arithmetic scalar
            elif opcode in ["VADDS", "VSUBS", "VMULS", "VDIVS"]:
                self._exec_scalar_op(inst, start_cycle, end_cycle)
            
            # Reduction
            elif opcode == "VREDUCE_SUM":
                self._exec_reduce_sum(inst, start_cycle)
            elif opcode == "VREDUCE_MAX":
                self._exec_reduce_max(inst, start_cycle)
            elif opcode == "VREDUCE_MIN":
                self._exec_reduce_min(inst, start_cycle)
            
            # Dot product
            elif opcode == "VDOT":
                self._exec_vdot(inst, start_cycle, end_cycle)
            
            # Logical
            elif opcode in ["VAND", "VOR", "VXOR"]:
                self._exec_logical_op(inst, start_cycle, end_cycle)
            
            # Mask instructions - delegate to vector_simulator implementation
            elif opcode == "VMASK":
                self._execute_vmask(inst, start_cycle)
            elif opcode == "VADD_MASKED":
                self._execute_vadd_masked(inst, start_cycle)
            elif opcode == "VMASK_COND":
                self._execute_vmask_condition(inst, start_cycle)
            elif opcode == "VSUB_MASKED":
                self._execute_vsub_masked(inst, start_cycle)
            elif opcode == "VMOV_MASKED":
                self._execute_vmov_masked(inst, start_cycle)
            
            else:
                raise NotImplementedError(f"Unsupported: {opcode}")
        
        except Exception as e:
            print(f"Error executing {inst.raw_text}: {e}")
            raise
    
    # ===== Masking helpers (true masked SIMD trong mô hình pipeline) =====
    #
    # Ghi chú kiến trúc: trong pipeline SIMD, lane bị mask được PREDICATED — vẫn
    # đi qua đủ các stage (chiếm slot) nên latency EX/MEM KHÔNG đổi. Vì vậy ở đây
    # ta chỉ cần đảm bảo TÍNH ĐÚNG CHỨC NĂNG (writeback chỉ lane active), còn
    # timing giữ nguyên theo độ sâu pipeline. (Việc rút ngắn timing theo lane
    # active là đặc thù của mô hình density-time ở VectorSimulator.)

    def _resolve_mask(self, inst: Instruction, n: int, start_cycle: int) -> List[bool]:
        """Lấy predicate per-lane cho lệnh (None → mọi lane active)."""
        if inst.mask is None:
            return [True] * n
        bits = self.mask_registers.get(inst.mask)
        if bits is None:
            try:
                bits = [v != 0 for v in self.vrf.read(inst.mask, start_cycle)]
            except Exception:
                bits = [True] * n
        bits = [bool(b) for b in bits]
        if len(bits) < n:
            bits = bits + [False] * (n - len(bits))
        return bits[:n]

    def _merge_masked(self, dst: str, new_values: List[float], mask: List[bool],
                      start_cycle: int, mask_mode: str = "merge") -> List[float]:
        """Hợp nhất writeback: lane active lấy new_values, lane inactive giữ giá
        trị cũ của đích (merge) hoặc 0 (zero)."""
        n = len(new_values)
        try:
            old = list(self.vrf.read(dst, start_cycle))
        except Exception:
            old = []
        if len(old) < n:
            old = old + [0.0] * (n - len(old))
        out = []
        for i in range(n):
            if i < len(mask) and mask[i]:
                out.append(new_values[i])
            elif mask_mode == "zero":
                out.append(0.0)
            else:
                out.append(old[i])
        return out

    # ===== Memory Instructions Execution =====

    def _exec_vload(self, inst: Instruction, start_cycle: int) -> int:
        dst = inst.dst
        array_name = inst.src[0]
        if inst.mask is None:
            values, mem_cycles = self.memory.load_vector(array_name, start_cycle)
        else:
            mask_full = self._resolve_mask(inst, self.config.vector_length, start_cycle)
            raw, mem_cycles = self.memory.load_vector(array_name, start_cycle, mask=mask_full)
            values = self._merge_masked(dst, raw, mask_full, start_cycle, inst.mask_mode)
        self.vrf.write(dst, values, start_cycle, self.config.load_latency)

        # Thêm vào forwarding
        if self.config.enable_chaining:
            ready_cycle = start_cycle + self.config.load_latency
            self.forwarding_data[dst] = (ready_cycle, values)
        return mem_cycles

    def _exec_vstore(self, inst: Instruction, start_cycle: int) -> int:
        array_name = inst.dst
        src_reg = inst.src[0]
        values = self.vrf.read(src_reg, start_cycle)
        if inst.mask is None:
            return self.memory.store_vector(array_name, values, start_cycle)
        mask = self._resolve_mask(inst, len(values), start_cycle)
        return self.memory.store_vector(array_name, values, start_cycle, mask=mask)

    def _exec_vload_stride(self, inst: Instruction, start_cycle: int) -> int:
        dst = inst.dst
        array_name = inst.src[0]
        stride = int(inst.src[1])
        values, mem_cycles = self.memory.load_stride(array_name, stride, self.config.vector_length, start_cycle)
        self.vrf.write(dst, values, start_cycle, self.config.load_latency)

        if self.config.enable_chaining:
            ready_cycle = start_cycle + self.config.load_latency
            self.forwarding_data[dst] = (ready_cycle, values)
        return mem_cycles

    def _exec_vgather(self, inst: Instruction, start_cycle: int) -> int:
        dst = inst.dst
        array_name = inst.src[0]
        idx_array = inst.src[1]
        indices = self.memory.get_array(idx_array)
        indices_int = [int(i) for i in indices]
        if inst.mask is None:
            values, mem_cycles = self.memory.gather(array_name, indices_int, start_cycle)
        else:
            mask = self._resolve_mask(inst, len(indices_int), start_cycle)
            raw, mem_cycles = self.memory.gather(array_name, indices_int, start_cycle, mask=mask)
            values = self._merge_masked(dst, raw, mask, start_cycle, inst.mask_mode)
        self.vrf.write(dst, values, start_cycle, self.config.load_latency * 2)
        return mem_cycles

    def _exec_vscatter(self, inst: Instruction, start_cycle: int) -> int:
        array_name = inst.dst
        idx_array = inst.src[0]
        src_reg = inst.src[1]
        indices = self.memory.get_array(idx_array)
        indices_int = [int(i) for i in indices]
        values = self.vrf.read(src_reg, start_cycle)
        if inst.mask is None:
            return self.memory.scatter(array_name, indices_int, values, start_cycle)
        mask = self._resolve_mask(inst, len(values), start_cycle)
        return self.memory.scatter(array_name, indices_int, values, start_cycle, mask=mask)
    
    # ===== Arithmetic Execution =====
    
    def _exec_binary_op(self, inst: Instruction, start_cycle: int, end_cycle: int):
        opcode = inst.opcode
        dst = inst.dst
        src1 = inst.src[0]
        src2 = inst.src[1]
        
        a = self.vrf.read(src1, start_cycle)
        b = self.vrf.read(src2, start_cycle)
        
        if opcode == "VADD":
            result = [x + y for x, y in zip(a, b)]
        elif opcode == "VSUB":
            result = [x - y for x, y in zip(a, b)]
        elif opcode == "VMUL":
            result = [x * y for x, y in zip(a, b)]
        elif opcode == "VDIV":
            result = [x / y if y != 0 else 0 for x, y in zip(a, b)]
        elif opcode == "VMAX":
            result = [max(x, y) for x, y in zip(a, b)]
        elif opcode == "VMIN":
            result = [min(x, y) for x, y in zip(a, b)]
        else:
            raise NotImplementedError(opcode)

        if inst.mask is not None:
            mask = self._resolve_mask(inst, len(result), start_cycle)
            result = self._merge_masked(dst, result, mask, start_cycle, inst.mask_mode)

        latency = self._get_latency(opcode)
        self.vrf.write(dst, result, start_cycle, latency)

        if self.config.enable_chaining:
            ready_cycle = start_cycle + latency
            self.forwarding_data[dst] = (ready_cycle, result)

    def _exec_scalar_op(self, inst: Instruction, start_cycle: int, end_cycle: int):
        opcode = inst.opcode
        dst = inst.dst
        src = inst.src[0]
        scalar_token = inst.src[1]
        
        a = self.vrf.read(src, start_cycle)
        
        if scalar_token in self.scalars:
            s = self.scalars[scalar_token]
        else:
            s = float(scalar_token)
        
        if opcode == "VADDS":
            result = [x + s for x in a]
        elif opcode == "VSUBS":
            result = [x - s for x in a]
        elif opcode == "VMULS":
            result = [x * s for x in a]
        elif opcode == "VDIVS":
            result = [x / s if s != 0 else 0 for x in a]
        else:
            raise NotImplementedError(opcode)

        if inst.mask is not None:
            mask = self._resolve_mask(inst, len(result), start_cycle)
            result = self._merge_masked(dst, result, mask, start_cycle, inst.mask_mode)

        latency = self._get_latency(opcode)
        self.vrf.write(dst, result, start_cycle, latency)

        if self.config.enable_chaining:
            ready_cycle = start_cycle + latency
            self.forwarding_data[dst] = (ready_cycle, result)

    def _masked_values(self, inst: Instruction, values: List[float], start_cycle: int) -> List[float]:
        """Lọc lane active cho reduction/dot (None → toàn bộ)."""
        if inst.mask is None:
            return values
        mask = self._resolve_mask(inst, len(values), start_cycle)
        return [v for v, m in zip(values, mask) if m]

    def _exec_reduce_sum(self, inst: Instruction, start_cycle: int):
        values = self._masked_values(inst, self.vrf.read(inst.src[0], start_cycle), start_cycle)
        self.scalars[inst.dst] = sum(values)

    def _exec_reduce_max(self, inst: Instruction, start_cycle: int):
        values = self._masked_values(inst, self.vrf.read(inst.src[0], start_cycle), start_cycle)
        self.scalars[inst.dst] = max(values) if values else 0

    def _exec_reduce_min(self, inst: Instruction, start_cycle: int):
        values = self._masked_values(inst, self.vrf.read(inst.src[0], start_cycle), start_cycle)
        self.scalars[inst.dst] = min(values) if values else 0

    def _exec_vdot(self, inst: Instruction, start_cycle: int, end_cycle: int):
        a = self.vrf.read(inst.src[0], start_cycle)
        b = self.vrf.read(inst.src[1], start_cycle)
        if inst.mask is None:
            result = sum(x * y for x, y in zip(a, b))
        else:
            mask = self._resolve_mask(inst, min(len(a), len(b)), start_cycle)
            result = sum(x * y for x, y, m in zip(a, b, mask) if m)
        self.scalars[inst.dst] = result
    
    def _exec_logical_op(self, inst: Instruction, start_cycle: int, end_cycle: int):
        opcode = inst.opcode
        dst = inst.dst
        src1 = inst.src[0]
        src2 = inst.src[1]
        
        a = self.vrf.read(src1, start_cycle)
        b = self.vrf.read(src2, start_cycle)
        
        a_int = [int(x) for x in a]
        b_int = [int(x) for x in b]
        
        if opcode == "VAND":
            result = [x & y for x, y in zip(a_int, b_int)]
        elif opcode == "VOR":
            result = [x | y for x, y in zip(a_int, b_int)]
        elif opcode == "VXOR":
            result = [x ^ y for x, y in zip(a_int, b_int)]
        else:
            raise NotImplementedError(opcode)

        result_float = [float(x) for x in result]
        if inst.mask is not None:
            mask = self._resolve_mask(inst, len(result_float), start_cycle)
            result_float = self._merge_masked(dst, result_float, mask, start_cycle, inst.mask_mode)
        latency = self._get_latency(opcode)
        self.vrf.write(dst, result_float, start_cycle, latency)
    
    # ===== Mask Instructions (delegate to VectorSimulator implementation) =====
    
    def _execute_vmask(self, inst, start_cycle: int) -> int:
        """VMASK Vd, Vs, condition — Tạo mask so sánh từng phần tử với mean(vector).

        Hỗ trợ đầy đủ 6 điều kiện: gt, lt, eq, ne, ge, le.
        Kết quả lưu vào mask_registers[dst] (bool) để các lệnh masked dùng sau.
        """
        dst = inst.dst
        src_reg = inst.src[0]
        condition = inst.src[1]
        values = self.vrf.read(src_reg, start_cycle)
        threshold = sum(values) / len(values) if values else 0
        cond_map = {
            "gt": lambda v: v > threshold,
            "lt": lambda v: v < threshold,
            "eq": lambda v: v == threshold,
            "ne": lambda v: v != threshold,   # đã thêm — nhất quán với VectorSimulator
            "ge": lambda v: v >= threshold,
            "le": lambda v: v <= threshold,
        }
        if condition not in cond_map:
            raise ValueError(
                f"Unknown condition '{condition}'. "
                f"Valid: {sorted(cond_map.keys())}"
            )
        mask = [cond_map[condition](v) for v in values]
        self.mask_registers[dst] = mask
        return self.config.add_latency
    
    def _execute_vadd_masked(self, inst, start_cycle: int) -> int:
        """VADD_MASKED Vd, Vs1, Vs2, Vmask - Cong co dieu kien theo mask"""
        dst = inst.dst
        a = self.vrf.read(inst.src[0], start_cycle)
        b = self.vrf.read(inst.src[1], start_cycle)
        mask_reg = inst.src[2]
        mask = self.mask_registers.get(mask_reg, [True] * len(a))
        result = [x + y if m else x for x, y, m in zip(a, b, mask)]
        latency = self._get_latency("VADD")
        self.vrf.write(dst, result, start_cycle, latency)
        return latency
    
    def _execute_vmask_condition(self, inst, start_cycle: int) -> int:
        """VMASK_COND Vd, Vs1, Vs2, condition - So sanh 2 vector"""
        dst = inst.dst
        a = self.vrf.read(inst.src[0], start_cycle)
        b = self.vrf.read(inst.src[1], start_cycle)
        condition = inst.src[2]
        cond_map = {
            "gt": lambda x, y: x > y, "lt": lambda x, y: x < y,
            "eq": lambda x, y: x == y, "ge": lambda x, y: x >= y,
            "le": lambda x, y: x <= y, "ne": lambda x, y: x != y,
        }
        if condition not in cond_map:
            raise ValueError(f"Unknown condition: {condition}")
        fn = cond_map[condition]
        self.mask_registers[dst] = [fn(x, y) for x, y in zip(a, b)]
        return self.config.add_latency
    
    def _execute_vsub_masked(self, inst, start_cycle: int) -> int:
        """VSUB_MASKED Vd, Vs1, Vs2, Vmask"""
        dst = inst.dst
        a = self.vrf.read(inst.src[0], start_cycle)
        b = self.vrf.read(inst.src[1], start_cycle)
        mask = self.mask_registers.get(inst.src[2], [True] * len(a))
        result = [x - y if m else x for x, y, m in zip(a, b, mask)]
        latency = self._get_latency("VSUB")
        self.vrf.write(dst, result, start_cycle, latency)
        return latency
    
    def _execute_vmov_masked(self, inst, start_cycle: int) -> int:
        """VMOV_MASKED Vd, Vs, Vmask"""
        dst = inst.dst
        src = self.vrf.read(inst.src[0], start_cycle)
        mask = self.mask_registers.get(inst.src[1], [True] * len(src))
        try:
            existing = self.vrf.read(dst, start_cycle)
        except Exception:
            existing = [0.0] * len(src)
        result = [s if m else e for s, e, m in zip(src, existing, mask)]
        latency = self._get_latency("VADD")
        self.vrf.write(dst, result, start_cycle, latency)
        return latency
    
    # ===== Helper Methods =====

    def _get_functional_unit(self, inst: Instruction) -> FunctionalUnit:
        """Lấy functional unit cho lệnh"""
        if inst.is_memory_instruction:
            return FunctionalUnit.MEMORY
        elif inst.inst_type in [InstructionType.ARITHMETIC_REDUCTION, InstructionType.ARITHMETIC_DOT]:
            return FunctionalUnit.REDUCTION
        else:
            return FunctionalUnit.ALU
    
    def _get_latency(self, opcode: str) -> int:
        """Nguồn tra cứu latency duy nhất cho mọi opcode.

        Đây là bảng chân lý duy nhất — tất cả phương thức khác gọi vào đây
        thay vì duy trì bảng riêng (loại bỏ DRY violation cũ).
        """
        c = self.config
        latency_map = {
            # Memory
            "VLOAD":        c.load_latency,
            "VSTORE":       c.store_latency,
            "VLOAD_STRIDE": c.load_latency,
            "VGATHER":      c.load_latency * 2,
            "VSCATTER":     c.store_latency * 2,
            # ALU binary
            "VADD": c.add_latency,  "VSUB": c.add_latency,
            "VMUL": c.mul_latency,  "VDIV": c.div_latency,
            "VMAX": c.add_latency,  "VMIN": c.add_latency,
            # ALU scalar
            "VADDS": c.add_latency, "VSUBS": c.add_latency,
            "VMULS": c.mul_latency, "VDIVS": c.div_latency,
            # Reduction / dot
            "VREDUCE_SUM": c.reduction_latency,
            "VREDUCE_MAX": c.reduction_latency,
            "VREDUCE_MIN": c.reduction_latency,
            "VDOT": c.mul_latency + c.add_latency,
            # Logical
            "VAND": c.add_latency, "VOR": c.add_latency, "VXOR": c.add_latency,
            # Mask (tương đương ALU)
            "VMASK":        c.add_latency,
            "VMASK_COND":   c.add_latency,
            "VADD_MASKED":  c.add_latency,
            "VSUB_MASKED":  c.add_latency,
            "VMOV_MASKED":  c.add_latency,
        }
        return latency_map.get(opcode, c.add_latency)

    def _get_instruction_latency(self, inst: Instruction) -> int:
        """Wrapper tiện lợi — delegate sang _get_latency."""
        return self._get_latency(inst.opcode)
    
    # ===== Statistics Methods =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê pipeline"""
        return {
            "total_cycles": self.stats["total_cycles"],
            "total_instructions": self.stats["total_instructions"],
            "stall_cycles": self.stats["stall_cycles"],
            "raw_hazards": self.stats["raw_hazards"],
            "structural_hazards": self.stats["structural_hazards"],
            "forwarding_used": self.stats["forwarding_used"],
            "ipc": self.stats["total_instructions"] / self.stats["total_cycles"] if self.stats["total_cycles"] > 0 else 0,
            "stall_percentage": (self.stats["stall_cycles"] / self.stats["total_cycles"]) * 100 if self.stats["total_cycles"] > 0 else 0,
            "stage_stats": self.get_stage_stats(),
            "memory_stats": self.memory.get_stats(),
            "register_stats": self.vrf.get_stats()
        }

    def get_stage_stats(self) -> Dict[str, Any]:
        """Thống kê mức độ chiếm dụng từng stage (IF/ID/EX/MEM/WB)."""
        total = self.stats.get("total_cycles", 0) or 0
        busy = {"IF": 0, "ID": 0, "EX": 0, "MEM": 0, "WB": 0}
        for s in self.stage_timeline:
            busy["IF"] += 1
            busy["ID"] += 1
            busy["EX"] += max(0, s["EX_end"] - s["EX_start"])
            busy["MEM"] += max(0, s["MEM_end"] - s["MEM_start"])
            busy["WB"] += 1
        utilization = {k: (v / total if total else 0.0) for k, v in busy.items()}
        return {"busy_cycles": busy, "utilization": utilization, "total_cycles": total}
    
    def print_stats(self):
        """In thống kê pipeline"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("PIPELINE SIMULATOR STATISTICS")
        print("=" * 60)
        print(f"Total cycles:           {stats['total_cycles']}")
        print(f"Total instructions:     {stats['total_instructions']}")
        print(f"IPC:                    {stats['ipc']:.3f}")
        print(f"Stall cycles:           {stats['stall_cycles']} ({stats['stall_percentage']:.1f}%)")
        print(f"RAW hazards:            {stats['raw_hazards']}")
        print(f"Structural hazards:     {stats['structural_hazards']}")
        print(f"Forwarding used:        {stats['forwarding_used']}")
    
    def print_timeline(self):
        """In timeline"""
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION TIMELINE")
        print("=" * 70)
        print(f"{'Instruction':<35} {'Start':<8} {'End':<8} {'Dur':<6} {'Unit':<12} {'Stall'}")
        print("-" * 70)
        
        for event in self.timeline:
            stall_info = f"{event.get('raw_stall', 0)}c" if event.get('raw_stall', 0) > 0 else "-"
            print(f"{event['instruction']:<35} {event['start']:<8} {event['end']:<8} "
                f"{event['duration']:<6} {event['unit']:<12} {stall_info}")


# ===== TEST =====
if __name__ == "__main__":
    from config import VectorProcessorConfig
    from instruction import InstructionTemplates
    
    # Tạo config
    config = VectorProcessorConfig(
        vector_length=8,
        num_lanes=4,
        memory_bandwidth=4,
        add_latency=2,
        mul_latency=4,
        load_latency=3,
        store_latency=3,
        enable_chaining=True,
        num_memory_units=1,
        num_alu_units=1
    )
    
    # Tạo dữ liệu
    data = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "B": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    }
    
    # Tạo instructions
    instructions = [
        InstructionTemplates.vload("V1", "A"),
        InstructionTemplates.vload("V2", "B"),
        InstructionTemplates.vadd("V3", "V1", "V2"),
        InstructionTemplates.vstore("C", "V3"),
    ]
    
    # Chạy pipeline simulator
    sim = PipelineSimulator(config, data)
    memory, timeline, cycles = sim.run(instructions)
    
    print(f"Total cycles: {cycles}")
    print(f"Result C: {memory.get('C', [])}")
    
    # In timeline
    sim.print_timeline()
    
    # In thống kê
    sim.print_stats()
    
    print("\n✓ All tests passed!")
