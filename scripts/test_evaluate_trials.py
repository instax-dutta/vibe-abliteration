#!/usr/bin/env python3
"""Self-check for evaluate_trials.py. Run: python3 scripts/test_evaluate_trials.py"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("evaluate_trials.py")


def journal(rows):
    """rows: (trial_id, refusals, kl) or (trial_id, refusals, kl, params)"""
    lines = []
    for row in rows:
        tid, refusals, kl = row[:3]
        attrs = {"index": tid}
        if len(row) > 3:
            attrs["parameters"] = row[3]
        lines.append({"op_code": 8, "trial_id": tid, "user_attr": attrs})
        lines.append({"op_code": 8, "trial_id": tid, "user_attr": {"scores": [
            {"name": "Refusals", "score": {"value": refusals}},
            {"name": "KL divergence", "score": {"value": kl}},
        ]}})
    return "\n".join(json.dumps(line) for line in lines)


class EvaluateTrialsTest(unittest.TestCase):
    def run_script(self, content, *args):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "journal.jsonl").write_text(content)
            return subprocess.run([sys.executable, str(SCRIPT), tmp, *args],
                                  capture_output=True, text=True)

    def test_json_reports_pareto_and_params(self):
        result = self.run_script(journal([
            (0, 0.09, 0.010, {"direction_index": 0.0}),
            (1, 0.02, 0.050, {"direction_index": 0.1}),
            (2, 0.50, 0.900, {"direction_index": 0.2}),  # dominated
        ]), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["counts"]["trials"], 3)
        self.assertEqual(data["counts"]["pareto"], 2)
        # Heretic sorts the front by (refusals, kl)
        self.assertEqual([t["index"] for t in data["pareto"] if t["pareto"]], [1, 0])
        self.assertEqual(data["pareto"][0]["trial_index"], 0)
        self.assertEqual(data["best_by_refusals"]["index"], 1)
        self.assertEqual(data["best_by_kl"]["index"], 0)
        self.assertEqual(data["best_by_kl"]["params"], {"direction_index": 0.0})

    def test_skips_nan_and_incomplete_trials(self):
        content = journal([(0, 0.10, 0.010), (1, float("nan"), 0.001)])
        content += "\n" + json.dumps({"op_code": 8, "trial_id": 2,
                                      "user_attr": {"index": 2}})  # no scores
        result = self.run_script(content, "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["counts"]["trials"], 1)
        self.assertEqual(data["best_by_kl"]["index"], 0)

    def test_empty_dir_fails_loudly(self):
        result = self.run_script("")
        self.assertEqual(result.returncode, 1)
        self.assertIn("No complete trials found", result.stderr)


if __name__ == "__main__":
    unittest.main()
