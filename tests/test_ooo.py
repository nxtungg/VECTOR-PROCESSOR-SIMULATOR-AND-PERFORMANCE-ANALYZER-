import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig
from parser import ProgramParser
from ooo_simulator import OOOSimulator


class TestOOOSimulator(unittest.TestCase):

    def setUp(self):
        self.config = VectorProcessorConfig(vector_length=8, num_lanes=4)

    def test_ooo_run_returns_memory_and_cycles(self):
        parser = ProgramParser()
        data, scalars, instructions = parser.parse_file("examples/vector_add.txt")
        sim = OOOSimulator(self.config, data, scalars)
        memory, timeline, cycles = sim.run(instructions)

        self.assertGreater(cycles, 0)
        self.assertIn("C", memory)
        self.assertEqual(memory["C"][0], 11)
        self.assertGreater(len(timeline), 0)

    def test_ooo_api_matches_other_simulators(self):
        parser = ProgramParser()
        data, scalars, instructions = parser.parse_text(
            "DATA A = [1, 2]\nVLOAD V1, A\nVSTORE B, V1"
        )
        sim = OOOSimulator(self.config, data, scalars, num_rs=4)
        memory, _, cycles = sim.run(instructions)
        self.assertEqual(memory["B"], [1, 2])
        self.assertGreater(cycles, 0)

    def test_ooo_respects_single_alu_resource(self):
        parser = ProgramParser()
        data, scalars, instructions = parser.parse_text(
            """
DATA A = [1, 2, 3, 4]
DATA B = [10, 20, 30, 40]
DATA C = [5, 6, 7, 8]
DATA D = [9, 10, 11, 12]

VLOAD V1, A
VLOAD V2, B
VLOAD V3, C
VLOAD V4, D
VADD V5, V1, V2
VSUB V6, V3, V4
"""
        )
        cfg = VectorProcessorConfig(
            vector_length=4,
            num_lanes=4,
            num_alu_units=1,
            num_memory_units=4,
            load_latency=1,
            add_latency=3,
        )
        sim = OOOSimulator(cfg, data, scalars, num_rs=8)
        _, timeline, _ = sim.run(instructions)

        alu_events = [t for t in timeline if t["opcode"] in {"VADD", "VSUB"}]
        self.assertEqual(len(alu_events), 2)
        self.assertGreaterEqual(alu_events[1]["exec_start"], alu_events[0]["exec_end"])


if __name__ == "__main__":
    unittest.main()
