import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig
from parser import ProgramParser
from scalar_simulator import ScalarSimulator
from vector_simulator import VectorSimulator
from pipeline_simulator import PipelineSimulator


class TestAllExamples(unittest.TestCase):

    def setUp(self):
        self.config = VectorProcessorConfig()
        self.parser = ProgramParser()
        examples_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
        )
        self.examples = sorted(glob.glob(os.path.join(examples_dir, "*.txt")))

    def test_all_examples_parse_and_run(self):
        self.assertGreater(len(self.examples), 0)
        for example in self.examples:
            with self.subTest(example=os.path.basename(example)):
                data, scalars, instructions = self.parser.parse_file(example)
                self.assertGreater(len(instructions), 0)

                scalar_sim = ScalarSimulator(self.config)
                scalar_cycles = scalar_sim.estimate_cycles(instructions, data)
                self.assertGreater(scalar_cycles, 0)

                vector_sim = VectorSimulator(self.config, data, scalars)
                _, _, vector_cycles = vector_sim.run(instructions)
                self.assertGreater(vector_cycles, 0)

                pipeline_sim = PipelineSimulator(self.config, data, scalars)
                _, _, pipeline_cycles = pipeline_sim.run(instructions)
                self.assertGreater(pipeline_cycles, 0)


if __name__ == "__main__":
    unittest.main()
