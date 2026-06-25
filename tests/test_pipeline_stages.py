import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig
from parser import ProgramParser
from vector_simulator import VectorSimulator
from pipeline_simulator import PipelineSimulator


class TestPipelineStages(unittest.TestCase):
    """Kiểm tra pipeline 5-stage vật lý (IF/ID/EX/MEM/WB)."""

    def setUp(self):
        self.config = VectorProcessorConfig(vector_length=8, num_lanes=4,
                                            memory_bandwidth=4)
        self.parser = ProgramParser()

    def _run(self, file):
        data, scalars, ins = self.parser.parse_file(file)
        sim = PipelineSimulator(self.config, dict(data), dict(scalars))
        mem, tl, cycles = sim.run(ins)
        return sim, mem, tl, cycles

    def test_stage_timeline_per_instruction(self):
        # Không hard-code số lệnh: mỗi lệnh có đúng một bản ghi stage.
        data, scalars, ins = self.parser.parse_file("examples/vector_add.txt")
        sim = PipelineSimulator(self.config, dict(data), dict(scalars))
        sim.run(ins)
        self.assertEqual(len(sim.stage_timeline), len(ins))
        # Pipeline 5 tầng lộ các mốc thời gian tầng (đặc trưng kiến trúc cần có).
        for s in sim.stage_timeline:
            for key in ("IF", "ID", "EX_start", "EX_end", "MEM_start", "MEM_end", "WB"):
                self.assertIn(key, s)

    def test_stage_order_monotonic(self):
        sim, _, _, _ = self._run("examples/vector_add.txt")
        for s in sim.stage_timeline:
            self.assertLessEqual(s["IF"], s["ID"])
            self.assertLessEqual(s["ID"], s["EX_start"])
            self.assertLessEqual(s["EX_start"], s["EX_end"])
            self.assertLessEqual(s["EX_end"], s["MEM_end"])
            self.assertLessEqual(s["MEM_start"], s["WB"])

    def test_functional_result_correct(self):
        _, mem, _, _ = self._run("examples/vector_add.txt")
        self.assertEqual(mem["C"], [11, 22, 33, 44, 55, 66, 77, 88])

    def test_pipeline_not_slower_than_serial_vector(self):
        # Bất biến quan trọng: pipeline (overlap) <= vector (tuần tự)
        for f in ["vector_add", "saxpy", "dot_product", "stride_access"]:
            data, scalars, ins = self.parser.parse_file(f"examples/{f}.txt")
            _, _, vc = VectorSimulator(self.config, dict(data), dict(scalars)).run(ins)
            _, _, pc = PipelineSimulator(self.config, dict(data), dict(scalars)).run(ins)
            self.assertLessEqual(pc, vc, f"{f}: pipeline {pc} > vector {vc}")

    def test_stage_stats(self):
        sim, _, _, _ = self._run("examples/vector_add.txt")
        stats = sim.get_stage_stats()
        self.assertIn("utilization", stats)
        self.assertIn("EX", stats["utilization"])
        for v in stats["utilization"].values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_cache_increases_pipeline_mem_cost(self):
        data, scalars, ins = self.parser.parse_file("examples/vector_add.txt")
        base = PipelineSimulator(self.config, dict(data), dict(scalars))
        _, _, c0 = base.run(ins)
        cached = PipelineSimulator(self.config, dict(data), dict(scalars))
        cached.memory.enable_cache_simulation(cache_size=2, line_size=2)
        _, _, c1 = cached.run(ins)
        # Cache miss penalty làm MEM stage tốn hơn (hoặc bằng)
        self.assertGreaterEqual(c1, c0)

    def test_raw_dependency_creates_stall(self):
        program = """DATA A = [1, 2, 3, 4, 5, 6, 7, 8]
DATA B = [8, 7, 6, 5, 4, 3, 2, 1]
VLOAD V1, A
VLOAD V2, B
VADD V3, V1, V2
VMUL V4, V3, V2
"""
        data, scalars, ins = self.parser.parse_text(program)
        sim = PipelineSimulator(self.config, dict(data), dict(scalars))
        sim.run(ins)
        # Bất biến hành vi: có RAW dependency thì phải phát hiện hazard.
        # (Chỉ kiểm tra hệ quả kiến trúc, không phụ thuộc key nội bộ của timeline.)
        self.assertGreater(sim.stats["raw_hazards"], 0)

    def test_memory_unit_structural_hazard_is_counted(self):
        program = """DATA A = [1, 2, 3, 4, 5, 6, 7, 8]
DATA B = [8, 7, 6, 5, 4, 3, 2, 1]
VLOAD V1, A
VLOAD V2, B
"""
        data, scalars, ins = self.parser.parse_text(program)
        sim = PipelineSimulator(self.config, dict(data), dict(scalars))
        sim.run(ins)
        self.assertGreater(sim.stats["structural_hazards"], 0)


if __name__ == "__main__":
    unittest.main()
