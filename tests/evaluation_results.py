# tests/evaluation_results.py
# Collects and formats evaluation results

import json
import os
from datetime import datetime
from deepeval.test_case import LLMTestCase


class EvaluationResults:
    """
    Collects results from multiple metric evaluations
    and produces structured reports.
    """

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.results = {}
        self.timestamp = datetime.now().isoformat()

    def evaluate(self, test_case: LLMTestCase, metrics: dict) -> dict:
        """Run all metrics against a test case and collect results."""
        for metric_name, metric in metrics.items():
            try:
                metric.measure(test_case)
                self.results[metric_name] = {
                    "score":     round(metric.score, 3),
                    "threshold": metric.threshold,
                    "passed":    metric.score >= metric.threshold,
                    "reason":    metric.reason
                }
            except Exception as e:
                self.results[metric_name] = {
                    "score":     0.0,
                    "threshold": metric.threshold,
                    "passed":    False,
                    "reason":    f"Evaluation error: {str(e)}"
                }
        return self.results

    def get_overall_score(self) -> float:
        """Calculate weighted average score across all metrics."""
        if not self.results:
            return 0.0
        scores = [r["score"] for r in self.results.values()]
        return round(sum(scores) / len(scores), 3)

    def get_pass_rate(self) -> float:
        """Percentage of metrics that passed."""
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results.values() if r["passed"])
        return round(passed / len(self.results) * 100, 1)

    def print_summary(self):
        """Print a formatted summary to console."""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY: {self.test_name}")
        print(f"{'='*60}")
        for metric, result in self.results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"{metric:<15} {result['score']:.3f}  {status}")
            if result["reason"]:
                # Truncate long reasons for console display
                reason = result["reason"][:100] + "..." \
                    if len(result["reason"]) > 100 \
                    else result["reason"]
                print(f"               → {reason}")
        print(f"{'─'*60}")
        print(f"Overall Score:  {self.get_overall_score():.3f}")
        print(f"Pass Rate:      {self.get_pass_rate()}%")
        print(f"{'='*60}\n")

    def save_to_file(self, output_dir: str = "reports"):
        """Save full results to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/{self.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "test_name":     self.test_name,
            "timestamp":     self.timestamp,
            "overall_score": self.get_overall_score(),
            "pass_rate":     self.get_pass_rate(),
            "metrics":       self.results
        }
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to: {filename}")
        return filename