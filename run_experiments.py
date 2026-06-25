#!/usr/bin/env python3
"""Research-grade experimental framework cho Vector Processor Simulator.

Chạy 7 thí nghiệm có kiểm soát, xuất CSV + metadata + biểu đồ + báo cáo, theo
hướng tái lập (reproducibility) và nhất quán về phương pháp luận.

================================================================================
PHƯƠNG PHÁP LUẬN & MÔ HÌNH HIỆU NĂNG (đọc trước khi diễn giải số liệu)
================================================================================
Đơn vị thời gian chung: CYCLE (trừu tượng). Thời gian thực = cycles / (clock).
Mọi simulator dùng CHUNG một ``VectorProcessorConfig`` ⇒ cùng định nghĩa cycle
và cùng latency semantics (load/store/add/mul/div/reduction/startup).

Mức trừu tượng của từng backend (nêu rõ để tránh hiểu nhầm dữ liệu):
- scalar   : CYCLE-APPROXIMATE (analytical) — ScalarSimulator.estimate_cycles
             ước lượng giải tích, dùng làm baseline.
- vector   : EXECUTION-DRIVEN — VectorSimulator.run thực thi từng lệnh (không
             pipeline), timing suy ra từ latency + grouping theo lanes.
- pipeline : EXECUTION-DRIVEN — PipelineSimulator.run, 5 tầng, có RAW/structural
             hazard.

Phân loại thí nghiệm (tách bạch simulation vs analytical):
- EXP1..EXP6 : SIMULATION-BASED (số liệu từ backend chạy thật).
- EXP7       : ANALYTICAL (mô hình closed-form về lane utilization của masking,
               KHÔNG phải số đo runtime).

Định nghĩa metric (một nguồn duy nhất — xem class ``Metrics``; thống nhất với
performance_analyzer.PerformanceAnalyzer):
- speedup(baseline, target)      = baseline / target
- throughput(elements, cycles)   = elements / cycles            [elem/cycle]
- efficiency_raw(speedup, ideal)  = speedup / ideal              (KHÔNG cắt trần)
- efficiency_norm                 = min(efficiency_raw, 1.0)     (chỉ để vẽ)
- bandwidth_utilization(moved, cycles, bw) = min(moved/(cycles*bw), 1.0)
Sự khác biệt giữa các thí nghiệm chỉ nằm ở OPERAND (vd speedup theo vector hay
pipeline) và được ghi tường minh trong ``assumptions`` của mỗi thí nghiệm —
KHÔNG có khác biệt ngầm về công thức.
"""

import os
import sys
import csv
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Thêm thư mục hiện tại vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VectorProcessorConfig
from parser import ProgramParser
from scalar_simulator import ScalarSimulator
from vector_simulator import VectorSimulator
from pipeline_simulator import PipelineSimulator
from performance_analyzer import PerformanceAnalyzer


# ==================================================================
# Giả định mô hình hiệu năng dùng chung (đưa vào metadata để tái lập)
# ==================================================================
CYCLE_MODEL_ASSUMPTIONS: Dict[str, str] = {
    "time_unit": "cycle (abstract); wall_time = cycles / (clock_frequency_mhz * 1e6)",
    "shared_config": "Mọi simulator dùng cùng VectorProcessorConfig (cùng latency).",
    "scalar_model": "cycle-approximate (analytical) — ScalarSimulator.estimate_cycles",
    "vector_model": "execution-driven (non-pipelined) — VectorSimulator.run",
    "pipeline_model": "execution-driven 5-stage (RAW/structural hazards) — PipelineSimulator.run",
    "latency_source": "VectorProcessorConfig: load/store/add/mul/div/reduction/startup",
}


class Metrics:
    """Định nghĩa metric DUY NHẤT cho mọi thí nghiệm (tránh khác biệt ngầm).

    Cùng công thức với performance_analyzer.PerformanceAnalyzer; gom ở đây để mọi
    experiment gọi chung một chỗ.
    """

    @staticmethod
    def speedup(baseline_cycles: float, target_cycles: float) -> float:
        return baseline_cycles / target_cycles if target_cycles > 0 else 0.0

    @staticmethod
    def throughput(num_elements: float, cycles: float) -> float:
        return num_elements / cycles if cycles > 0 else 0.0

    @staticmethod
    def efficiency_raw(speedup_value: float, ideal_speedup: float) -> float:
        # KHÔNG cắt trần: giữ thông tin khi speedup > ideal (overlap/pipeline effect).
        return speedup_value / ideal_speedup if ideal_speedup > 0 else 0.0

    @staticmethod
    def efficiency_norm(efficiency_raw_value: float) -> float:
        # Chỉ phục vụ trực quan hoá (thanh % ≤ 100%), không dùng kết luận khoa học.
        return min(efficiency_raw_value, 1.0)

    @staticmethod
    def bandwidth_utilization(elements_moved: float, cycles: float,
                              bandwidth: float) -> float:
        max_possible = cycles * bandwidth
        return min(elements_moved / max_possible, 1.0) if max_possible > 0 else 0.0


@dataclass
class ExperimentResult:
    """Đối tượng kết quả có cấu trúc theo chuẩn paper.

    Gồm: inputs (tham số quét + cố định), assumptions (giả định mô hình), rows
    (derived metrics theo từng cấu hình), và metadata phân loại. Phục vụ trực
    tiếp cho việc viết báo cáo và tái lập thí nghiệm.
    """
    id: str
    title: str
    kind: str                          # "simulation" | "analytical"
    independent_var: str               # biến độc lập được quét
    inputs: Dict[str, Any] = field(default_factory=dict)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    derived_metrics: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    def metadata(self) -> Dict[str, Any]:
        """Metadata (không gồm rows) để ghi sidecar JSON."""
        d = asdict(self)
        d.pop("rows", None)
        d["num_rows"] = len(self.rows)
        return d


class ExperimentRunner:
    """Chạy và ghi lại kết quả 7 thí nghiệm (mỗi thí nghiệm được cô lập)."""

    def __init__(self, output_dir="experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.parser = ProgramParser()
        self.analyzer = None

        # results: tương thích ngược — dict[str, list[row]] dùng cho charts/report.
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        # experiments: lớp research-grade — dict[str, ExperimentResult].
        self.experiments: Dict[str, ExperimentResult] = {}

    # ==================================================================
    # Helpers chung
    # ==================================================================

    def _build_vector_add_program(self, num_elements: int):
        """Tạo chương trình C=A+B với đúng num_elements phần tử."""
        a = list(range(1, num_elements + 1))
        b = [i * 10 for i in range(1, num_elements + 1)]
        program = f"""DATA A = {a}
DATA B = {b}

VLOAD V1, A
VLOAD V2, B
VADD V3, V1, V2
VSTORE C, V3
"""
        return self.parser.parse_text(program)

    def _run_simulation_suite(self, config, data, scalars, instructions):
        """Chạy scalar (analytical) + vector & pipeline (execution-driven).

        Trả về cycles của cả 3 backend dưới CÙNG một config ⇒ cùng cycle semantics.
        """
        scalar_sim = ScalarSimulator(config)
        scalar_cycles = scalar_sim.estimate_cycles(instructions, data)

        vector_sim = VectorSimulator(config, data.copy(), dict(scalars))
        _, _, vector_cycles = vector_sim.run(instructions)

        pipeline_sim = PipelineSimulator(config, data.copy(), dict(scalars))
        _, _, pipeline_cycles = pipeline_sim.run(instructions)

        return scalar_cycles, vector_cycles, pipeline_cycles

    def _finalize(self, exp: ExperimentResult, csv_name: str,
                  summary_columns: Optional[List[str]] = None):
        """Lưu kết quả: đăng ký structured object + rows (compat) + CSV + JSON meta."""
        self.experiments[exp.id] = exp
        self.results[exp.id] = exp.rows
        csv_rows = [r for r in exp.rows if "error" not in r]
        self._save_csv(csv_rows, csv_name)
        self._save_metadata(exp)
        if summary_columns:
            self._print_summary_table(
                [r for r in exp.rows if "error" not in r], summary_columns)
        return exp.rows

    def run_all_experiments(self):
        """Chạy tất cả 7 thí nghiệm với CÔ LẬP lỗi giữa các thí nghiệm."""
        print("=" * 70)
        print("VECTOR PROCESSOR SIMULATOR - EXPERIMENTS (research-grade)")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)

        # Mỗi thí nghiệm được bọc _safe_run: lỗi ở một EXP không làm hỏng EXP khác.
        for fn in (self.exp1_vector_length, self.exp2_num_lanes,
                   self.exp3_memory_bandwidth, self.exp4_workload_comparison,
                   self.exp5_stride_access, self.exp6_cache_size,
                   self.exp7_mask_utilization):
            self._safe_run(fn)

        self._safe_run(self.generate_summary_report)
        self._safe_run(self.generate_charts)

        print("\n" + "=" * 70)
        print("✅ All experiments completed!")
        print(f"Results saved in: {self.output_dir}")
        print("=" * 70)

    def _safe_run(self, fn: Callable):
        """Cô lập một bước: bắt mọi lỗi, in cảnh báo, không lan sang bước khác."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — chủ đích cô lặp lỗi giữa các EXP
            print(f"  ⚠ {getattr(fn, '__name__', fn)} failed (skipped): {e}")
            return None

    # ==================== THÍ NGHIỆM 1: VECTOR LENGTH ====================

    def exp1_vector_length(self):
        """EXP1 (simulation): Ảnh hưởng của vector length."""
        print("\n" + "─" * 70)
        print("📊 EXPERIMENT 1: Effect of Vector Length")
        print("─" * 70)

        vector_lengths = [4, 8, 16, 32, 64]
        lanes = 4
        memory_bw = 4

        exp = ExperimentResult(
            id="exp1", title="Effect of Vector Length", kind="simulation",
            independent_var="vector_length",
            inputs={"vector_length": vector_lengths, "lanes": lanes,
                    "memory_bandwidth": memory_bw, "workload": "vector_add (C=A+B)"},
            assumptions={**CYCLE_MODEL_ASSUMPTIONS,
                         "speedup_operands": "scalar_cycles / vector_cycles",
                         "ideal_speedup": "num_lanes"},
            derived_metrics=["speedup", "throughput", "lane_utilization", "efficiency"],
        )

        for vl in vector_lengths:
            print(f"\n  Running with Vector Length = {vl}...")
            data, scalars, instructions = self._build_vector_add_program(vl)
            num_elements = vl

            config = VectorProcessorConfig(vector_length=vl, num_lanes=lanes,
                                           memory_bandwidth=memory_bw)
            self.analyzer = PerformanceAnalyzer(config)

            scalar_cycles, vector_cycles, pipeline_cycles = \
                self._run_simulation_suite(config, data, scalars, instructions)

            speedup = Metrics.speedup(scalar_cycles, vector_cycles)
            throughput = Metrics.throughput(num_elements, vector_cycles)
            lane_util = self.analyzer.calculate_lane_utilization(num_elements)
            eff_raw = Metrics.efficiency_raw(speedup, lanes)

            exp.rows.append({
                "vector_length": vl, "lanes": lanes, "memory_bandwidth": memory_bw,
                "num_elements": num_elements,
                "scalar_cycles": scalar_cycles, "vector_cycles": vector_cycles,
                "pipeline_cycles": pipeline_cycles,
                "speedup": round(speedup, 2),
                "throughput": round(throughput, 4),
                "lane_utilization": round(lane_util, 4),
                "efficiency": round(eff_raw, 4),
                "efficiency_norm": round(Metrics.efficiency_norm(eff_raw), 4),
            })
            print(f"    Scalar: {scalar_cycles} | Pipeline: {pipeline_cycles} | "
                  f"Speedup: {speedup:.2f}x")

        return self._finalize(
            exp, "exp1_vector_length.csv",
            ["vector_length", "vector_cycles", "speedup", "lane_utilization"])

    # ==================== THÍ NGHIỆM 2: NUMBER OF LANES ====================

    def exp2_num_lanes(self):
        """EXP2 (simulation): Ảnh hưởng của số lanes."""
        print("\n" + "─" * 70)
        print("📊 EXPERIMENT 2: Effect of Number of Lanes")
        print("─" * 70)

        lanes_list = [1, 2, 4, 8, 16]
        vector_length = 64
        memory_bw = 8
        data, scalars, instructions = self._build_vector_add_program(vector_length)
        num_elements = vector_length

        exp = ExperimentResult(
            id="exp2", title="Effect of Number of Lanes", kind="simulation",
            independent_var="lanes",
            inputs={"lanes": lanes_list, "vector_length": vector_length,
                    "memory_bandwidth": memory_bw, "workload": "vector_add (C=A+B)"},
            assumptions={**CYCLE_MODEL_ASSUMPTIONS,
                         "speedup_operands": "scalar_cycles / vector_cycles",
                         "ideal_speedup": "num_lanes",
                         "efficiency": "raw = speedup/ideal (uncapped); efficiency_norm để vẽ"},
            derived_metrics=["speedup", "ideal_speedup", "efficiency",
                             "throughput", "lane_utilization"],
        )

        for lanes in lanes_list:
            print(f"\n  Running with {lanes} lanes...")
            config = VectorProcessorConfig(vector_length=vector_length,
                                           num_lanes=lanes, memory_bandwidth=memory_bw)
            self.analyzer = PerformanceAnalyzer(config)

            scalar_cycles, vector_cycles, pipeline_cycles = \
                self._run_simulation_suite(config, data, scalars, instructions)

            speedup = Metrics.speedup(scalar_cycles, vector_cycles)
            throughput = Metrics.throughput(num_elements, vector_cycles)
            lane_util = self.analyzer.calculate_lane_utilization(num_elements)
            ideal_speedup = lanes
            eff_raw = Metrics.efficiency_raw(speedup, ideal_speedup)

            exp.rows.append({
                "lanes": lanes, "vector_length": vector_length,
                "memory_bandwidth": memory_bw,
                "scalar_cycles": scalar_cycles, "vector_cycles": vector_cycles,
                "pipeline_cycles": pipeline_cycles,
                "speedup": round(speedup, 2),
                "ideal_speedup": ideal_speedup,
                "efficiency": round(eff_raw, 4),
                "efficiency_norm": round(Metrics.efficiency_norm(eff_raw), 4),
                "throughput": round(throughput, 4),
                "lane_utilization": round(lane_util, 4),
            })
            print(f"    Speedup: {speedup:.2f}x (Ideal: {ideal_speedup}x) | "
                  f"Efficiency: {eff_raw:.1%}")

        return self._finalize(
            exp, "exp2_num_lanes.csv",
            ["lanes", "speedup", "ideal_speedup", "efficiency"])

    # ==================== THÍ NGHIỆM 3: MEMORY BANDWIDTH ====================

    def exp3_memory_bandwidth(self):
        """EXP3 (simulation): Ảnh hưởng của memory bandwidth."""
        print("\n" + "─" * 70)
        print("📊 EXPERIMENT 3: Effect of Memory Bandwidth")
        print("─" * 70)

        bandwidths = [1, 2, 4, 8, 16]
        vector_length = 64
        lanes = 8
        data, scalars, instructions = self._build_vector_add_program(vector_length)
        num_elements = vector_length

        exp = ExperimentResult(
            id="exp3", title="Effect of Memory Bandwidth", kind="simulation",
            independent_var="memory_bandwidth",
            inputs={"memory_bandwidth": bandwidths, "vector_length": vector_length,
                    "lanes": lanes, "workload": "vector_add (C=A+B)"},
            assumptions={**CYCLE_MODEL_ASSUMPTIONS,
                         "speedup_operands": "scalar_cycles / vector_cycles",
                         "memory_traffic": "2×VLOAD + 1×VSTORE = 3·N phần tử"},
            derived_metrics=["speedup", "throughput", "bandwidth_utilization"],
        )

        for bw in bandwidths:
            print(f"\n  Running with Memory Bandwidth = {bw} elem/cycle...")
            config = VectorProcessorConfig(vector_length=vector_length,
                                           num_lanes=lanes, memory_bandwidth=bw)
            self.analyzer = PerformanceAnalyzer(config)

            scalar_cycles, vector_cycles, pipeline_cycles = \
                self._run_simulation_suite(config, data, scalars, instructions)

            speedup = Metrics.speedup(scalar_cycles, vector_cycles)
            throughput = Metrics.throughput(num_elements, vector_cycles)
            total_memory_elements = num_elements * 3  # 2x VLOAD + VSTORE
            bw_util = Metrics.bandwidth_utilization(total_memory_elements,
                                                    vector_cycles, bw)

            exp.rows.append({
                "memory_bandwidth": bw, "vector_length": vector_length, "lanes": lanes,
                "scalar_cycles": scalar_cycles, "vector_cycles": vector_cycles,
                "pipeline_cycles": pipeline_cycles,
                "speedup": round(speedup, 2),
                "throughput": round(throughput, 4),
                "bandwidth_utilization": round(bw_util, 4),
            })
            print(f"    Speedup: {speedup:.2f}x | BW Utilization: {bw_util:.1%}")

        return self._finalize(
            exp, "exp3_memory_bandwidth.csv",
            ["memory_bandwidth", "vector_cycles", "speedup", "bandwidth_utilization"])

    # ==================== THÍ NGHIỆM 4: WORKLOAD COMPARISON ====================

    def exp4_workload_comparison(self):
        """EXP4 (simulation): So sánh các workload (safe-skip nếu thiếu file)."""
        print("\n" + "─" * 70)
        print("📊 EXPERIMENT 4: Workload Comparison")
        print("─" * 70)

        workloads = {
            "Vector Add": "examples/vector_add.txt",
            "SAXPY": "examples/saxpy.txt",
            "Dot Product": "examples/dot_product.txt",
            "Matrix-Vector": "examples/matrix_vector.txt",
            "Stride Access": "examples/stride_access.txt",
        }

        config = VectorProcessorConfig(vector_length=16, num_lanes=4,
                                       memory_bandwidth=4)
        self.analyzer = PerformanceAnalyzer(config)

        exp = ExperimentResult(
            id="exp4", title="Workload Comparison", kind="simulation",
            independent_var="workload",
            inputs={"workloads": list(workloads.keys()),
                    "vector_length": 16, "lanes": 4, "memory_bandwidth": 4},
            assumptions={**CYCLE_MODEL_ASSUMPTIONS,
                         "speedup_operands": "scalar_cycles / pipeline_cycles",
                         "note": "Thiếu file input → safe-skip, ghi vào 'skipped'."},
            derived_metrics=["speedup", "throughput", "lane_utilization"],
        )

        for name, program_file in workloads.items():
            print(f"\n  Running: {name}...")

            if not os.path.exists(program_file):
                print(f"    ⚠ File not found (skipped): {program_file}")
                skip = {"workload": name, "error": "File not found",
                        "file": program_file}
                exp.rows.append(skip)
                exp.skipped.append(skip)
                continue

            try:
                data, scalars, instructions = self.parser.parse_file(program_file)
                num_elements = len(next(iter(data.values()))) if data else config.vector_length

                scalar_cycles, vector_cycles, pipeline_cycles = \
                    self._run_simulation_suite(config, data, scalars, instructions)

                speedup = Metrics.speedup(scalar_cycles, pipeline_cycles)
                throughput = Metrics.throughput(num_elements, pipeline_cycles)
                lane_util = self.analyzer.calculate_lane_utilization(num_elements)

                exp.rows.append({
                    "workload": name, "num_elements": num_elements,
                    "scalar_cycles": scalar_cycles, "vector_cycles": vector_cycles,
                    "pipeline_cycles": pipeline_cycles,
                    "speedup": round(speedup, 2),
                    "throughput": round(throughput, 4),
                    "lane_utilization": round(lane_util, 4),
                })
                print(f"    Scalar: {scalar_cycles} | Pipeline: {pipeline_cycles} | "
                      f"Speedup: {speedup:.2f}x")
            except Exception as e:  # cô lập lỗi từng workload
                print(f"    ❌ Error (skipped): {e}")
                skip = {"workload": name, "error": str(e)}
                exp.rows.append(skip)
                exp.skipped.append(skip)

        valid = [r for r in exp.rows if "error" not in r]
        return self._finalize(
            exp, "exp4_workload_comparison.csv",
            ["workload", "pipeline_cycles", "speedup", "throughput"] if valid else None)

    # ==================== THÍ NGHIỆM 5: STRIDE ACCESS ====================

    def exp5_stride_access(self):
        """EXP5 (simulation): Ảnh hưởng của stride access."""
        print("\n" + "─" * 70)
        print("📊 EXPERIMENT 5: Effect of Stride Access")
        print("─" * 70)

        strides = [1, 2, 4, 8, 16]
        vector_length = 32
        lanes = 4
        memory_bw = 8

        exp = ExperimentResult(
            id="exp5", title="Effect of Stride Access", kind="simulation",
            independent_var="stride",
            inputs={"stride": strides, "vector_length": vector_length,
                    "lanes": lanes, "memory_bandwidth": memory_bw},
            assumptions={**CYCLE_MODEL_ASSUMPTIONS,
                         "effective_bandwidth": "num_elements * base_bw / vector_cycles"},
            derived_metrics=["cycles", "effective_bandwidth"],
        )

        for stride in strides:
            print(f"\n  Running with Stride = {stride}...")
            array_size = vector_length * stride
            data_array = list(range(1, array_size + 1))
            program_content = f"""DATA A = {data_array}

VLOAD_STRIDE V1, A, {stride}
VSTORE B, V1
"""
            temp_file = self.output_dir / f"temp_stride_{stride}.txt"
            try:
                with open(temp_file, "w") as f:
                    f.write(program_content)
                data, scalars, instructions = self.parser.parse_file(str(temp_file))
                num_elements = vector_length

                config = VectorProcessorConfig(vector_length=vector_length,
                                               num_lanes=lanes, memory_bandwidth=memory_bw)
                self.analyzer = PerformanceAnalyzer(config)

                vector_sim = VectorSimulator(config, data, scalars)
                _, _, vector_cycles = vector_sim.run(instructions)

                effective_bw = Metrics.throughput(num_elements * memory_bw, vector_cycles)

                exp.rows.append({
                    "stride": stride, "num_elements": num_elements,
                    "cycles": vector_cycles,
                    "effective_bandwidth": round(effective_bw, 4),
                })
                print(f"    Cycles: {vector_cycles} | Effective BW: "
                      f"{effective_bw:.2f} (Base: {memory_bw})")
            except Exception as e:  # cô lập lỗi từng stride
                print(f"    ❌ Error (skipped): {e}")
                skip = {"stride": stride, "error": str(e)}
                exp.rows.append(skip)
                exp.skipped.append(skip)
            finally:
                if temp_file.exists():
                    temp_file.unlink()

        valid = [r for r in exp.rows if "error" not in r]
        return self._finalize(
            exp, "exp5_stride_access.csv",
            ["stride", "cycles", "effective_bandwidth"] if valid else None)

    # ==================== THÍ NGHIỆM 6: CACHE SIZE ====================

    def exp6_cache_size(self):
        """EXP6 (simulation): Hit rate & cycles khi thay đổi dung lượng L1."""
        print("\n" + "-" * 70)
        print("EXPERIMENT 6: Effect of Cache Size")
        print("-" * 70)

        cache_sizes = [4, 8, 16, 32, 64]
        config = VectorProcessorConfig(vector_length=64, num_lanes=8,
                                       memory_bandwidth=8)
        data, scalars, instructions = self._build_vector_add_program(64)

        exp = ExperimentResult(
            id="exp6", title="Effect of Cache Size", kind="simulation",
            independent_var="cache_lines",
            inputs={"cache_lines": cache_sizes, "line_size": 4, "associativity": 1,
                    "vector_length": 64, "lanes": 8, "memory_bandwidth": 8},
            assumptions={**CYCLE_MODEL_ASSUMPTIONS,
                         "cache": "L1 direct-mapped (assoc=1), line_size=4 phần tử",
                         "backend": "PipelineSimulator + enable_cache_simulation"},
            derived_metrics=["cycles", "cache_hits", "cache_misses", "cache_hit_rate"],
        )

        for cache_lines in cache_sizes:
            sim = PipelineSimulator(config, dict(data), dict(scalars))
            sim.memory.enable_cache_simulation(cache_size=cache_lines, line_size=4,
                                               associativity=1)
            _, _, cycles = sim.run(instructions)
            stats = sim.memory.get_stats()
            exp.rows.append({
                "cache_lines": cache_lines, "line_size": 4, "cycles": cycles,
                "cache_hits": stats.get("cache_hits", 0),
                "cache_misses": stats.get("cache_misses", 0),
                "cache_hit_rate": round(stats.get("cache_hit_rate_percent", 0) / 100.0, 4),
            })

        return self._finalize(exp, "exp6_cache_size.csv")

    # ==================== THÍ NGHIỆM 7: MASK UTILIZATION ====================

    def exp7_mask_utilization(self):
        """EXP7 (ANALYTICAL): Mô hình lane utilization theo tỉ lệ mask active.

        KHÔNG phải số đo runtime — đây là mô hình closed-form: với tỉ lệ active r,
        số lane-slot phát ra = ceil(N/lanes)*lanes (không đổi), số lane-slot hữu
        ích = số phần tử active. lane_utilization = useful / issued.
        """
        print("\n" + "-" * 70)
        print("EXPERIMENT 7: Effect of Mask Utilization (analytical)")
        print("-" * 70)

        vector_length = 64
        lanes = 8
        active_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

        exp = ExperimentResult(
            id="exp7", title="Effect of Mask Utilization", kind="analytical",
            independent_var="active_ratio",
            inputs={"active_ratio": active_ratios, "vector_length": vector_length,
                    "lanes": lanes},
            assumptions={
                "model": "closed-form (KHÔNG chạy simulator)",
                "issued_lane_slots": "ceil(N/lanes) * lanes",
                "useful_lane_slots": "active_elements",
                "lane_utilization": "useful_lane_slots / issued_lane_slots",
                "note": "Masking không giảm số lane-slot phát ra; lane inactive lãng phí.",
            },
            derived_metrics=["lane_utilization", "useful_compute_batches"],
        )

        for ratio in active_ratios:
            active = int(vector_length * ratio)
            issue_batches = math.ceil(vector_length / lanes)
            useful_batches = math.ceil(active / lanes) if active else 0
            issued_slots = issue_batches * lanes
            exp.rows.append({
                "active_ratio": ratio,
                "active_elements": active,
                "inactive_elements": vector_length - active,
                "issued_lane_slots": issued_slots,
                "useful_lane_slots": active,
                "lane_utilization": round(active / issued_slots, 4) if issued_slots else 0.0,
                "useful_compute_batches": useful_batches,
            })

        return self._finalize(exp, "exp7_mask_utilization.csv")

    # ==================================================================
    # Lưu trữ & báo cáo
    # ==================================================================

    def _save_csv(self, data, filename):
        """Lưu kết quả ra file CSV."""
        if not data:
            print(f"  ⚠ No data to save for {filename}")
            return
        filepath = self.output_dir / filename
        keys = list(data[0].keys())
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"  Saved: {filepath}")

    def _save_metadata(self, exp: ExperimentResult):
        """Ghi sidecar JSON (inputs/assumptions/derived) phục vụ tái lập & viết paper."""
        filepath = self.output_dir / f"{exp.id}_meta.json"
        payload = {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **exp.metadata(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _print_summary_table(self, data, columns):
        """In bảng tóm tắt."""
        if not data:
            return
        print("\n  " + "─" * 70)
        header = " | ".join([f"{col:>18}" for col in columns])
        print(f"  {header}")
        print("  " + "─" * 70)
        for row in data[:10]:
            values = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    values.append(f"{val:>18.2f}")
                elif isinstance(val, int):
                    values.append(f"{val:>18d}")
                else:
                    values.append(f"{str(val):>18}")
            print(f"  {' | '.join(values)}")
        if len(data) > 10:
            print(f"  ... and {len(data) - 10} more rows")

    def generate_summary_report(self):
        """Tạo báo cáo tổng hợp (text)."""
        report_path = self.output_dir / f"summary_report_{self.timestamp}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("VECTOR PROCESSOR SIMULATOR - EXPERIMENT SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write("METHODOLOGY (cycle model assumptions):\n")
            for k, v in CYCLE_MODEL_ASSUMPTIONS.items():
                f.write(f"  - {k}: {v}\n")
            f.write("\n")

            if "exp1" in self.results and self.results["exp1"]:
                f.write("EXPERIMENT 1: Effect of Vector Length\n")
                f.write("-" * 40 + "\n")
                for r in self.results["exp1"]:
                    f.write(f"  VL={r['vector_length']:3d}: {r['vector_cycles']:4d} cycles, "
                            f"Speedup={r['speedup']:5.2f}x, Util={r['lane_utilization']:.1%}\n")
                f.write("\n")

            if "exp2" in self.results and self.results["exp2"]:
                f.write("EXPERIMENT 2: Effect of Number of Lanes\n")
                f.write("-" * 40 + "\n")
                for r in self.results["exp2"]:
                    f.write(f"  Lanes={r['lanes']:2d}: Speedup={r['speedup']:5.2f}x, "
                            f"Ideal={r['ideal_speedup']:2d}x, Eff={r['efficiency']:.1%}\n")
                f.write("\n")

            if "exp3" in self.results and self.results["exp3"]:
                f.write("EXPERIMENT 3: Effect of Memory Bandwidth\n")
                f.write("-" * 40 + "\n")
                for r in self.results["exp3"]:
                    f.write(f"  BW={r['memory_bandwidth']:2d}: {r['vector_cycles']:4d} cycles, "
                            f"Speedup={r['speedup']:5.2f}x, BW Util={r['bandwidth_utilization']:.1%}\n")
                f.write("\n")

            if "exp4" in self.results and self.results["exp4"]:
                f.write("EXPERIMENT 4: Workload Comparison\n")
                f.write("-" * 40 + "\n")
                for r in self.results["exp4"]:
                    if "error" not in r:
                        f.write(f"  {r['workload']:15s}: {r['pipeline_cycles']:4d} cycles, "
                                f"Speedup={r['speedup']:5.2f}x, Throughput={r['throughput']:.3f}\n")
                f.write("\n")

            if "exp5" in self.results and self.results["exp5"]:
                f.write("EXPERIMENT 5: Effect of Stride Access\n")
                f.write("-" * 40 + "\n")
                for r in self.results["exp5"]:
                    if "error" not in r:
                        f.write(f"  Stride={r['stride']:2d}: {r['cycles']:4d} cycles, "
                                f"Effective BW={r['effective_bandwidth']:.2f}\n")
                f.write("\n")

            if "exp6" in self.results and self.results["exp6"]:
                f.write("EXPERIMENT 6: Effect of Cache Size\n")
                f.write("-" * 40 + "\n")
                for r in self.results["exp6"]:
                    f.write(f"  Lines={r['cache_lines']:2d}: {r['cycles']:4d} cycles, "
                            f"Hits={r['cache_hits']:3d}, Misses={r['cache_misses']:3d}, "
                            f"Hit Rate={r['cache_hit_rate']:.1%}\n")
                f.write("\n")

            if "exp7" in self.results and self.results["exp7"]:
                f.write("EXPERIMENT 7: Effect of Mask Utilization\n")
                f.write("-" * 40 + "\n")
                f.write("  [analytical model — không phải số đo runtime]\n")
                for r in self.results["exp7"]:
                    f.write(f"  Active={r['active_ratio']:.0%}: "
                            f"{r['active_elements']:2d} elements, "
                            f"Lane Util={r['lane_utilization']:.1%}\n")
                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
        print(f"\nSummary report saved: {report_path}")

    def generate_charts(self):
        """Vẽ biểu đồ từ kết quả thí nghiệm."""
        try:
            from visualization import PerformanceVisualizer
        except ImportError:
            print("  ⚠ visualization module not available, skipping charts")
            return

        viz = PerformanceVisualizer(output_dir=str(self.figures_dir))

        if self.results.get("exp1"):
            r = self.results["exp1"]
            viz.plot_vector_length_effect(
                [x["vector_length"] for x in r],
                [x["vector_cycles"] for x in r],
                [x["speedup"] for x in r],
                "exp1_vector_length.png",
            )

        if self.results.get("exp2"):
            r = self.results["exp2"]
            viz.plot_lanes_effect(
                [x["lanes"] for x in r],
                [x["vector_cycles"] for x in r],
                [x["speedup"] for x in r],
                [x["ideal_speedup"] for x in r],
                "exp2_num_lanes.png",
            )

        if self.results.get("exp3"):
            r = self.results["exp3"]
            viz.plot_memory_bandwidth_effect(
                [x["memory_bandwidth"] for x in r],
                [x["vector_cycles"] for x in r],
                [x["speedup"] for x in r],
                "exp3_memory_bandwidth.png",
            )

        if self.results.get("exp4"):
            r = [x for x in self.results["exp4"] if "error" not in x]
            if r:
                viz.plot_workload_comparison(
                    [x["workload"] for x in r],
                    [x["scalar_cycles"] for x in r],
                    [x["vector_cycles"] for x in r],
                    [x["pipeline_cycles"] for x in r],
                    "exp4_workload_comparison.png",
                )

        if self.results.get("exp5"):
            r = [x for x in self.results["exp5"] if "error" not in x]
            if r:
                viz.plot_stride_effect(
                    [x["stride"] for x in r],
                    [x["cycles"] for x in r],
                    [x["effective_bandwidth"] for x in r],
                    "exp5_stride_access.png",
                )

        if self.results.get("exp6"):
            r = self.results["exp6"]
            viz.plot_cache_size_effect(
                [x["cache_lines"] for x in r],
                [x["cache_hit_rate"] for x in r],
                [x["cycles"] for x in r],
                "exp6_cache_size.png",
            )

        if self.results.get("exp7"):
            r = self.results["exp7"]
            viz.plot_mask_utilization(
                [x["active_ratio"] for x in r],
                [x["lane_utilization"] for x in r],
                "exp7_mask_utilization.png",
            )

        print(f"\n📊 Charts saved to: {self.figures_dir}")

    def generate_markdown_report(self, update_latest: bool = False):
        """Tạo báo cáo Markdown đầy đủ cho nộp bài."""
        report_path = self.output_dir / f"FINAL_REPORT_{self.timestamp}.md"
        lines = [
            "# Báo cáo Vector Processor Simulator",
            "",
            f"**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Chương 1. Giới thiệu",
            "",
            "Project mô phỏng bộ xử lý vector đơn giản, so sánh scalar, vector non-pipeline",
            "và pipeline; phân tích speedup, throughput và bottleneck do memory bandwidth.",
            "",
            "## Chương 2. Cơ sở lý thuyết",
            "",
            "- **SIMD / Vector lane:** xử lý nhiều phần tử song song.",
            "- **Pipeline:** chồng lấn lệnh khi không có RAW/structural hazard.",
            "- **Memory bandwidth:** giới hạn số phần tử truyền mỗi chu kỳ.",
            "- **Speedup** = baseline_cycles / target_cycles (xem metadata mỗi thí nghiệm).",
            "",
            "## Phương pháp luận (mô hình cycle)",
            "",
        ]
        for k, v in CYCLE_MODEL_ASSUMPTIONS.items():
            lines.append(f"- **{k}:** {v}")
        lines.extend([
            "",
            "## Chương 3. Thiết kế hệ thống",
            "",
            "Kiến trúc module: parser, memory, vector_register_file, scalar/vector/pipeline",
            "simulator, performance_analyzer, visualization, Streamlit UI.",
            "",
            "## Chương 4. Hiện thực",
            "",
            "Hỗ trợ 25+ lệnh vector (VLOAD, VSTORE, VADD, VMUL, VDOT, VGATHER, VMASK, ...).",
            "Bộ unit test tự động. Giao diện `app.py` (Streamlit).",
            "",
            "## Chương 5. Thực nghiệm và đánh giá",
            "",
        ])

        def _table(headers, rows):
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("")

        if self.results.get("exp1"):
            lines.append("### Thí nghiệm 1: Vector Length (simulation)")
            lines.append("")
            lines.append("![Vector Length](figures/exp1_vector_length.png)")
            lines.append("")
            _table(
                ["VL", "Vector Cycles", "Speedup", "Lane Util"],
                [(r["vector_length"], r["vector_cycles"], r["speedup"], r["lane_utilization"])
                 for r in self.results["exp1"]],
            )
            lines.append(
                "**Nhận xét:** Vector length nhỏ bị ảnh hưởng bởi startup latency; "
                "khi VL tăng, speedup ổn định hơn nhưng memory có thể trở thành bottleneck."
            )
            lines.append("")

        if self.results.get("exp2"):
            lines.append("### Thí nghiệm 2: Số Lanes (simulation)")
            lines.append("")
            lines.append("![Lanes](figures/exp2_num_lanes.png)")
            lines.append("")
            _table(
                ["Lanes", "Speedup", "Ideal", "Efficiency"],
                [(r["lanes"], r["speedup"], r["ideal_speedup"], r["efficiency"])
                 for r in self.results["exp2"]],
            )
            lines.append(
                "**Nhận xét:** Speedup không tăng tuyến tính vô hạn vì memory bandwidth "
                "và overhead pipeline giới hạn lợi ích khi tăng lanes."
            )
            lines.append("")

        if self.results.get("exp3"):
            lines.append("### Thí nghiệm 3: Memory Bandwidth (simulation)")
            lines.append("")
            lines.append("![Bandwidth](figures/exp3_memory_bandwidth.png)")
            lines.append("")
            _table(
                ["BW", "Vector Cycles", "Speedup", "BW Util"],
                [(r["memory_bandwidth"], r["vector_cycles"], r["speedup"],
                  r["bandwidth_utilization"]) for r in self.results["exp3"]],
            )
            lines.append(
                "**Nhận xét:** Bandwidth thấp làm tăng cycles load/store; tăng lanes "
                "không giúp nhiều khi hệ thống memory-bound."
            )
            lines.append("")

        if self.results.get("exp4"):
            lines.append("### Thí nghiệm 4: So sánh Workload (simulation)")
            lines.append("")
            lines.append("![Workloads](figures/exp4_workload_comparison.png)")
            lines.append("")
            valid = [r for r in self.results["exp4"] if "error" not in r]
            _table(
                ["Workload", "Cycles", "Speedup", "Throughput"],
                [(r["workload"], r["pipeline_cycles"], r["speedup"], r["throughput"])
                 for r in valid],
            )
            lines.append(
                "**Nhận xét:** Vector Add/SAXPY tận dụng ALU tốt; Dot Product bị giới hạn "
                "bởi reduction; Stride Access chịu penalty băng thông hiệu dụng."
            )
            lines.append("")

        if self.results.get("exp5"):
            lines.append("### Thí nghiệm 5: Stride Access (simulation)")
            lines.append("")
            lines.append("![Stride](figures/exp5_stride_access.png)")
            lines.append("")
            valid = [r for r in self.results["exp5"] if "error" not in r]
            _table(
                ["Stride", "Cycles", "Effective BW"],
                [(r["stride"], r["cycles"], r["effective_bandwidth"]) for r in valid],
            )
            lines.append(
                "**Nhận xét:** Stride lớn giảm locality; cần VGATHER/VSCATTER cho truy cập "
                "không liên tục phức tạp."
            )
            lines.append("")

        if self.results.get("exp6"):
            lines.append("### Thí nghiệm 6: Cache Size (simulation)")
            lines.append("")
            lines.append("![Cache](figures/exp6_cache_size.png)")
            lines.append("")
            _table(
                ["Cache Lines", "Cycles", "Hit Rate"],
                [(r["cache_lines"], r["cycles"], r["cache_hit_rate"])
                 for r in self.results["exp6"]],
            )
            lines.append(
                "**Nhận xét:** Cache lớn hơn giảm miss và cycles tới khi đủ chứa working set."
            )
            lines.append("")

        if self.results.get("exp7"):
            lines.append("### Thí nghiệm 7: Mask Utilization (analytical)")
            lines.append("")
            lines.append("![Mask](figures/exp7_mask_utilization.png)")
            lines.append("")
            lines.append("*Mô hình closed-form — KHÔNG phải số đo runtime.*")
            lines.append("")
            _table(
                ["Active Ratio", "Active Elems", "Lane Util"],
                [(r["active_ratio"], r["active_elements"], r["lane_utilization"])
                 for r in self.results["exp7"]],
            )
            lines.append(
                "**Nhận xét:** Mask thưa làm lãng phí lane (lane inactive vẫn chiếm slot phát)."
            )
            lines.append("")

        lines.extend([
            "## Chương 6. Kết luận",
            "",
            "- Đạt yêu cầu mô phỏng scalar, vector, pipeline và 7 thí nghiệm.",
            "- Mở rộng: mask, cache, OoO, convolution, so sánh AVX/NEON/RISC-V/GPU.",
            "- Hạn chế: OoO là mô hình timing; CNN đầy đủ cần mô phỏng sâu hơn.",
            "",
            "## Liên hệ SIMD thực tế",
            "",
            "Xem `comparison.py`: AVX-512 (16 float), ARM NEON (4 float), RISC-V Vector (VLEN",
            "tùy chỉnh), NVIDIA GPU (warp 32 threads).",
            "",
        ])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        latest = report_path
        if update_latest:
            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)
            latest = docs_dir / "FINAL_REPORT.md"
            latest.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nMarkdown report: {report_path}")
        if update_latest:
            print(f"Copy at: {latest}")


# ==================== MAIN ====================

def _ensure_utf8_stdout():
    import sys
    import io
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
            return
        except Exception:
            pass
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        sys.stdout = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace")


def main():
    """Chạy tất cả thí nghiệm (hoặc một thí nghiệm cụ thể qua --exp)."""
    import argparse
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Run vector processor experiments")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Run only specific experiment (1-7)")
    parser.add_argument("--output", default="experiment_results",
                        help="Output directory (default: experiment_results)")

    args = parser.parse_args()

    runner = ExperimentRunner(output_dir=args.output)

    if args.exp:
        experiments = {
            1: runner.exp1_vector_length,
            2: runner.exp2_num_lanes,
            3: runner.exp3_memory_bandwidth,
            4: runner.exp4_workload_comparison,
            5: runner.exp5_stride_access,
            6: runner.exp6_cache_size,
            7: runner.exp7_mask_utilization,
        }
        runner._safe_run(experiments[args.exp])
        runner._safe_run(runner.generate_summary_report)
        runner._safe_run(runner.generate_charts)
    else:
        runner.run_all_experiments()


if __name__ == "__main__":
    main()
