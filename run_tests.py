# run_tests.py
# Master test runner for ConvoQA Testing Framework
# Usage: python run_tests.py [suite] [--report]
#
# Suites:
#   all         — run everything
#   functional  — functional tests only
#   adversarial — adversarial tests only
#   regression  — regression tests only
#   smoke       — quick smoke test (fastest subset)

import subprocess
import sys
import os
import json
from datetime import datetime


# ─── Configuration ────────────────────────────────────────────

FRAMEWORK_VERSION = "1.0"
BOT_NAME = "NutriBot"
PROMPT_VERSION = "1.2"

SUITES = {
    "functional": [
        "tests/functional/test_nutribot_functional.py",
        "tests/functional/test_nutribot_deepeval.py",
    ],
    "pipeline": [
        "tests/functional/test_nutribot_pipeline.py",
    ],
    "adversarial": [
        "tests/adversarial/test_nutribot_adversarial.py",
    ],
    "regression": [
        "tests/regression/test_nutribot_regression.py",
    ],
    "smoke": [
        "tests/functional/test_nutribot_functional.py",
        "tests/regression/test_nutribot_regression.py"
        "::TestSafetyRegression",
    ],
    "all": [
        "tests/functional/test_nutribot_functional.py",
        "tests/functional/test_nutribot_deepeval.py",
        "tests/functional/test_nutribot_pipeline.py",
        "tests/adversarial/test_nutribot_adversarial.py",
        "tests/regression/test_nutribot_regression.py",
    ]
}


# ─── Runner ───────────────────────────────────────────────────

def run_suite(suite_name: str, generate_report: bool = True):
    """Run a named test suite and return results."""

    if suite_name not in SUITES:
        print(f"Unknown suite: '{suite_name}'")
        print(f"Available: {', '.join(SUITES.keys())}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/master_{suite_name}_{timestamp}.html"

    print(f"\n{'='*60}")
    print(f"ConvoQA Framework v{FRAMEWORK_VERSION}")
    print(f"Bot: {BOT_NAME} | Prompt: v{PROMPT_VERSION}")
    print(f"Suite: {suite_name.upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Build pytest command
    cmd = [
        "pytest",
        *SUITES[suite_name],
        "-v",
        "--tb=short",
    ]

    if generate_report:
        cmd += [
            f"--html={report_path}",
            "--self-contained-html"
        ]

    # Run tests
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Summary
    print(f"\n{'='*60}")
    print(f"Suite '{suite_name}' completed in {duration:.1f}s")
    print(f"Exit code: {result.returncode}")
    if generate_report:
        print(f"Report: {report_path}")
    print(f"{'='*60}\n")

    return result.returncode


def run_all_with_summary():
    """Run all suites and produce a master summary."""

    suites_to_run = ["functional", "adversarial", "regression"]
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"ConvoQA FULL RUN — {BOT_NAME} v{PROMPT_VERSION}")
    print(f"{'='*60}\n")

    for suite in suites_to_run:
        print(f"\n--- Running: {suite.upper()} ---\n")
        report_path = f"reports/{suite}_{timestamp}.html"

        cmd = [
            "pytest",
            *SUITES[suite],
            "-v",
            "--tb=short",
            f"--html={report_path}",
            "--self-contained-html"
        ]

        start = datetime.now()
        result = subprocess.run(cmd, capture_output=False)
        duration = (datetime.now() - start).total_seconds()

        results[suite] = {
            "exit_code": result.returncode,
            "passed":    result.returncode == 0,
            "duration":  round(duration, 1),
            "report":    report_path
        }

    # Master summary
    print(f"\n{'='*60}")
    print(f"MASTER SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    all_passed = True
    for suite, res in results.items():
        status = "✅ PASSED" if res["passed"] else "❌ FAILED"
        print(f"{suite:<15} {status}  ({res['duration']}s)")
        if not res["passed"]:
            all_passed = False

    print(f"{'─'*60}")
    overall = "✅ ALL SUITES PASSING" if all_passed else "❌ FAILURES DETECTED"
    print(f"Overall: {overall}")
    print(f"{'='*60}\n")

    # Save master summary JSON
    summary = {
        "framework_version": FRAMEWORK_VERSION,
        "bot":               BOT_NAME,
        "prompt_version":    PROMPT_VERSION,
        "timestamp":         timestamp,
        "overall_passed":    all_passed,
        "suites":            results
    }

    summary_path = f"reports/master_summary_{timestamp}.json"
    os.makedirs("reports", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Master summary saved: {summary_path}")

    return 0 if all_passed else 1


# ─── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "all":
        exit_code = run_all_with_summary()
    elif args[0] == "smoke":
        exit_code = run_suite("smoke")
    else:
        suite = args[0]
        report = "--no-report" not in args
        exit_code = run_suite(suite, generate_report=report)

    sys.exit(exit_code)