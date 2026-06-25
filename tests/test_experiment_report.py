import os
import tempfile
import unittest
from pathlib import Path

from run_experiments import ExperimentRunner


class TestExperimentReport(unittest.TestCase):

    def test_advanced_experiments_generate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(output_dir=tmp)
            cache_rows = runner.exp6_cache_size()
            mask_rows = runner.exp7_mask_utilization()

            self.assertEqual(len(cache_rows), 5)
            self.assertTrue(all(0.0 <= r["cache_hit_rate"] <= 1.0
                                for r in cache_rows))
            self.assertEqual(len(mask_rows), 5)
            self.assertEqual(mask_rows[0]["lane_utilization"], 0.0)
            self.assertEqual(mask_rows[-1]["lane_utilization"], 1.0)
            self.assertTrue((Path(tmp) / "exp6_cache_size.csv").exists())
            self.assertTrue((Path(tmp) / "exp7_mask_utilization.csv").exists())

            runner.generate_summary_report()
            summaries = list(Path(tmp).glob("summary_report_*.txt"))
            self.assertEqual(len(summaries), 1)
            summary = summaries[0].read_text(encoding="utf-8")
            self.assertIn("EXPERIMENT 6: Effect of Cache Size", summary)
            self.assertIn("EXPERIMENT 7: Effect of Mask Utilization", summary)

    def test_partial_report_does_not_overwrite_docs_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            latest = docs / "FINAL_REPORT.md"
            latest.write_text("full report", encoding="utf-8")

            runner = ExperimentRunner(output_dir=str(root / "results"))
            runner.results["exp5"] = [
                {"stride": 1, "cycles": 14, "effective_bandwidth": 18.29}
            ]

            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                runner.generate_markdown_report()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(latest.read_text(encoding="utf-8"), "full report")
            self.assertTrue(list((root / "results").glob("FINAL_REPORT_*.md")))


if __name__ == "__main__":
    unittest.main()
