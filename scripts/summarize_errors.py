#!/usr/bin/env python3
"""Count the reviewed error-analysis cases (numbers for Section 5 and Appendix B).

Input:  results/error_audit/reviewed_round_01.csv  (60 reviewed cases)
Output: results/error_audit/error_summary.csv
Usage:  python scripts/summarize_errors.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = ROOT / "results" / "error_audit" / "reviewed_round_01.csv"
OUTPUT = ROOT / "results" / "error_audit" / "error_summary.csv"
CATEGORIES = ["meaning substitution", "omission or truncation", "named entity or number error",
              "unsupported addition or repetition", "grammar or word-order defect"]


def main() -> None:
    cases = pd.read_csv(REVIEWED, keep_default_na=False)
    errors = cases[cases["annotation_status"] == "confirmed_error"]
    rows = []

    # 1. Decision for each of the 60 cases.
    for status in ["confirmed_error", "acceptable_paraphrase", "uncertain"]:
        rows.append(("status", status, int((cases["annotation_status"] == status).sum()), None))

    # 2. Error categories (percent of confirmed errors).
    for category in CATEGORIES:
        n = int((errors["dominant_error_type"] == category).sum())
        rows.append(("error_type", category, n, round(100 * n / len(errors), 1)))

    # 3. Possible source/reference mismatch flags.
    for flag, n in cases["source_reference_flag"].value_counts().sort_index().items():
        rows.append(("source_reference_flag", flag, int(n), None))
    flagged = cases["source_reference_flag"] == "suspected"
    rows.append(("suspected_flag_by_status", "confirmed_error",
                 int((flagged & (cases["annotation_status"] == "confirmed_error")).sum()), None))
    rows.append(("suspected_flag_by_status", "uncertain",
                 int((flagged & (cases["annotation_status"] == "uncertain")).sum()), None))

    # 4. Decisions per language.
    for (language, status), n in cases.groupby(["source_language", "annotation_status"]).size().items():
        rows.append((f"language:{language}", status, int(n), None))

    table = pd.DataFrame(rows, columns=["section", "item", "count", "percent_of_errors"])
    table.to_csv(OUTPUT, index=False)
    print(table.head(12).to_string(index=False))
    print(f"\nTotal cases: {len(cases)}. Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
