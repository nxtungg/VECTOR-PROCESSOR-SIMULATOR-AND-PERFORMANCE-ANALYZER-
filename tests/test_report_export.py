import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VectorProcessorConfig
from parser import ProgramParser
from scalar_simulator import ScalarSimulator
from vector_simulator import VectorSimulator
from pipeline_simulator import PipelineSimulator
from report_generator import ReportGenerator


class TestReportExport(unittest.TestCase):

    def setUp(self):
        self.config = VectorProcessorConfig(vector_length=8, num_lanes=4,
                                            memory_bandwidth=4)
        parser = ProgramParser()
        data, scalars, ins = parser.parse_file("examples/vector_add.txt")
        sc = ScalarSimulator(self.config).estimate_cycles(ins, data)
        _, _, vc = VectorSimulator(self.config, dict(data), dict(scalars)).run(ins)
        ps = PipelineSimulator(self.config, dict(data), dict(scalars))
        _, tl, pc = ps.run(ins)
        self.report = ReportGenerator()
        self.program_name = "vector_add.txt"
        self.config_dict = self.config.to_dict()
        self.scalar_cycles = sc
        self.vector_result = {"cycles": vc, "throughput": 0.5, "timeline": tl}
        self.pipeline_result = {"cycles": pc, "throughput": 0.6, "timeline": tl}

    def test_save_csv_timeline(self):
        """Test saving timeline to CSV"""
        timeline = [
            {"instruction": "VLOAD V1, A", "start": 0, "end": 5, "duration": 5, "unit": "Memory"},
            {"instruction": "VADD V2, V1, V1", "start": 5, "end": 10, "duration": 5, "unit": "ALU"},
        ]
        result = self.report.save_timeline(timeline, "test_timeline.csv")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.exists())

    def test_save_markdown_report(self):
        """Test saving markdown report"""
        result = self.report.save_markdown_report(
            self.program_name, self.config_dict, self.scalar_cycles,
            self.vector_result, self.pipeline_result
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.exists())
        content = result.read_text()
        self.assertIn("Vector Processor Simulation Report", content)
        self.assertIn("Execution Results", content)

    def test_save_all_formats_csv_only(self):
        """Test save_all_formats with CSV and Markdown only"""
        files = self.report.save_all_formats(
            self.program_name, self.config_dict, self.scalar_cycles,
            self.vector_result, self.pipeline_result
        )
        self.assertIsNotNone(files)
        # Should have markdown file
        self.assertIn('markdown', files)
        # Should NOT have json, html, pdf
        self.assertNotIn('json', files)


if __name__ == "__main__":
    unittest.main()
