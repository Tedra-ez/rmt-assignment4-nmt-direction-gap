#!/usr/bin/env python3
"""Run and verify the saved final-project calculations in the current environment.

This is a technical verification. Running this with an assistant does not meet
the course condition requiring a person outside the team to reproduce results.
"""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENERATED = [
    "per_script_scores.csv", "per_language_scores.csv", "main_summary.csv",
    "main_results_long.csv", "statistical_tests.csv", "robustness.csv",
    "leave_one_out.csv", "subgroup_results.csv", "diagnostics.csv",
    "error_audit/error_summary.csv",
]


def main():
    begun = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    before = {name: (ROOT / "results" / name).read_bytes() for name in GENERATED}
    arrays = {}
    for name in ["bootstrap_draws.npz", "permutation_statistics.npz"]:
        with np.load(ROOT / "results" / name) as saved:
            arrays[name] = {key: saved[key].copy() for key in saved.files}
    report = {"started_at_utc": begun,
              "verification_type": "technical reproduction; not an outside-team human confirmation",
              "python": sys.version.split()[0], "platform": platform.platform(),
              "commands": [], "csv_comparisons": [], "array_comparisons": []}
    commands = [
        ["scripts/analyze.py", "--validate-only"],
        ["-m", "unittest", "discover", "-s", "scripts", "-p", "test_analysis.py"],
        ["scripts/analyze.py"], ["scripts/summarize_errors.py"],
        ["scripts/export_final_results.py"],
    ]
    success = True
    for args in commands:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        report["commands"].append({"command": ["python", *args], "exit_code": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr})
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode:
            success = False
            break
    if success:
        for name, old_bytes in before.items():
            actual = (ROOT / "results" / name).read_bytes()
            error = None
            try:
                pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(old_bytes)),
                                              pd.read_csv(io.BytesIO(actual)),
                                              check_exact=False, rtol=1e-9, atol=1e-9)
            except AssertionError as exc:
                error = str(exc)
                success = False
            report["csv_comparisons"].append({"file": "results/" + name,
                                               "byte_identical": old_bytes == actual,
                                               "equal_within_1e_9": error is None,
                                               "error": error})
        for name, expected in arrays.items():
            with np.load(ROOT / "results" / name) as actual:
                for key, values in expected.items():
                    if np.issubdtype(values.dtype, np.number):
                        equal = bool(np.allclose(values, actual[key], rtol=1e-9, atol=1e-9))
                    else:
                        equal = bool(np.array_equal(values, actual[key]))
                    success = success and equal
                    report["array_comparisons"].append({"file": "results/" + name, "key": key,
                                                        "equal_within_1e_9": equal})
    report.update(success=success, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(time.perf_counter() - started, 3))
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs/technical_reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Technical reproduction passed." if success else "Technical reproduction failed; see logs/technical_reproduction.json.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
