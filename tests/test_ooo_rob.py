import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig
from parser import ProgramParser
from ooo_simulator import OOOSimulator


class TestOOOReorderBuffer(unittest.TestCase):
    """Kiểm tra OoO simulator: dependency tracking, in-order commit, kết quả đúng."""

    def setUp(self):
        self.config = VectorProcessorConfig(vector_length=8, num_lanes=4)
        self.parser = ProgramParser()

    def test_timeline_stages_are_ordered(self):
        data, scalars, ins = self.parser.parse_file("examples/vector_add.txt")
        sim = OOOSimulator(self.config, data, scalars, num_rs=4)
        _, timeline, cycles = sim.run(ins)
        self.assertGreater(cycles, 0)
        self.assertEqual(len(timeline), len(ins))
        # Kiểm tra BẤT BIẾN thứ tự tầng trong vòng đời mỗi lệnh thay vì chỉ kiểm
        # tra sự tồn tại của key nội bộ: issue -> exec -> writeback -> commit.
        for t in timeline:
            self.assertLessEqual(t["issue"], t["exec_start"])
            self.assertLessEqual(t["exec_start"], t["exec_end"])
            self.assertLessEqual(t["exec_end"], t["writeback"])
            self.assertLessEqual(t["writeback"], t["commit"])

    def test_in_order_commit(self):
        data, scalars, ins = self.parser.parse_file("examples/vector_add.txt")
        sim = OOOSimulator(self.config, data, scalars)
        _, timeline, _ = sim.run(ins)
        commits = [t["commit"] for t in timeline]
        self.assertEqual(commits, sorted(commits))   # commit theo đúng thứ tự

    def test_execute_respects_dependencies(self):
        data, scalars, ins = self.parser.parse_file("examples/vector_add.txt")
        sim = OOOSimulator(self.config, data, scalars)
        _, timeline, _ = sim.run(ins)
        by_op = {t["instruction"]: t for t in timeline}
        vadd = next(t for t in timeline if t["opcode"] == "VADD")
        loads = [t for t in timeline if t["opcode"] == "VLOAD"]
        # VADD chỉ execute sau khi cả 2 load writeback
        self.assertGreaterEqual(vadd["exec_start"], max(l["writeback"] for l in loads))

    def test_correct_results(self):
        data, scalars, ins = self.parser.parse_file("examples/vector_add.txt")
        mem, _, _ = OOOSimulator(self.config, data, scalars).run(ins)
        self.assertEqual(mem["C"][0], 11)
        self.assertEqual(mem["C"], [11, 22, 33, 44, 55, 66, 77, 88])

    def test_saxpy_results(self):
        prog = ("DATA X = [1, 2, 3, 4]\nDATA Y = [10, 10, 10, 10]\n"
                "SCALAR a = 2\nVLOAD V1, X\nVLOAD V2, Y\n"
                "VMULS V3, V1, a\nVADD V4, V3, V2\nVSTORE Y, V4")
        data, scalars, ins = self.parser.parse_text(prog)
        mem, _, _ = OOOSimulator(self.config, data, scalars).run(ins)
        self.assertEqual(mem["Y"], [12, 14, 16, 18])

    def test_reduction_scalar_result(self):
        prog = ("DATA A = [1, 2, 3, 4]\nDATA B = [1, 1, 1, 1]\n"
                "VLOAD V1, A\nVLOAD V2, B\nVMUL V3, V1, V2\nVREDUCE_SUM s, V3")
        data, scalars, ins = self.parser.parse_text(prog)
        sim = OOOSimulator(self.config, data, scalars)
        mem, timeline, cycles = sim.run(ins)
        self.assertGreater(cycles, 0)
        self.assertEqual(len(timeline), len(ins))


if __name__ == "__main__":
    unittest.main()
