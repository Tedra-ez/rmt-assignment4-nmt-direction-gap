#!/usr/bin/env python3
"""Export final-manuscript tables and figures from the existing analysis outputs.

Run after analyze.py and summarize_errors.py. This performs no new primary
inference. Values are copied at their saved precision; the figures use those
same files. Figures are exported as 600-dpi PNG and vector PDF.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "final"
FIGURES = ROOT / "figures" / "final"
MODELS = ["distilled_600m", "dense_3b", "moe_54b"]
LABELS = ["Distilled 600M", "Dense 3.3B", "MoE 54.5B"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]


def read(name):
    return pd.read_csv(RESULTS / name)


def save_figure(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = read("main_summary.csv").set_index("model").loc[MODELS].reset_index()
    stats = read("statistical_tests.csv")
    robustness = read("robustness.csv")
    subgroup = read("subgroup_results.csv")
    diagnostics = read("diagnostics.csv")
    errors = read("error_audit/error_summary.csv")
    group_columns = ["model", "label", "metric", "n_high", "n_low"]
    for value in ("high_into", "low_into", "high_from", "low_from", "gap_into", "gap_from"):
        group_columns += [value, value + "_ci_lower", value + "_ci_upper"]
    outcome_columns = ["model", "label", "metric"]
    for value in ("gap_into", "gap_from", "delta_gap"):
        outcome_columns += [value, value + "_ci_lower", value + "_ci_upper"]
    table3 = summary[outcome_columns].merge(stats, on="model", validate="one_to_one")
    table5 = subgroup.merge(diagnostics.drop(columns="mean_d"),
                           on=["model", "subgroup", "n_languages"],
                           validate="one_to_one")
    table6 = errors[errors["section"].isin(["status", "error_type", "source_reference_flag"])].copy()
    n_audited = int(errors.loc[errors["section"].eq("status"), "count"].sum())
    n_errors = int(errors.loc[errors["section"].eq("status") &
                              errors["item"].eq("confirmed_error"), "count"].iloc[0])
    table6["denominator"] = np.where(table6["section"].eq("error_type"), n_errors, n_audited)
    table6["percent_within_section"] = 100 * table6["count"] / table6["denominator"]
    tables = {
        "table2_group_means.csv": summary[group_columns],
        "table3_main_statistics.csv": table3,
        "table4_robustness.csv": robustness,
        "table5_subgroup_diagnostics.csv": table5,
        "table6_diagnostic_audit.csv": table6,
    }
    for name, table in tables.items():
        table.to_csv(TABLES / name, index=False, float_format="%.12g")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    positions = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.4, 4.0), layout="constrained")
    for offset, variable, color, label in [
        (-.18, "gap_into", COLORS[0], "Into English"),
        (.18, "gap_from", COLORS[1], "From English")]:
        points = positions + offset
        values = summary[variable].to_numpy()
        error = np.vstack([values - summary[variable + "_ci_lower"].to_numpy(),
                           summary[variable + "_ci_upper"].to_numpy() - values])
        ax.bar(points, values, width=.32, color=color, label=label,
               yerr=error, capsize=3, error_kw={"linewidth": 1})
    ax.set(xticks=positions, xticklabels=LABELS, ylabel="High − low gap (chrF++ points)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    save_figure(fig, "figure1_resource_gaps")

    fig, ax = plt.subplots(figsize=(7.4, 3.6), layout="constrained")
    for i, row in summary.iterrows():
        ax.plot([row["delta_gap_ci_lower"], row["delta_gap_ci_upper"]], [i, i],
                color=COLORS[0], linewidth=2)
        ax.scatter(row["delta_gap"], i, color=COLORS[0], s=45, zorder=3)
        ax.text(row["delta_gap_ci_upper"] + .12, i,
                f'{row["delta_gap"]:.2f} [{row["delta_gap_ci_lower"]:.2f}, {row["delta_gap_ci_upper"]:.2f}]',
                va="center", fontsize=9)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set(yticks=positions, yticklabels=LABELS, xlim=(-.25, 9.2),
           xlabel="ΔGap: from-English gap − into-English gap (chrF++ points)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.18)
    save_figure(fig, "figure2_delta_gap")

    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
    groups = ["high", "low_excluding_very_low", "very_low"]
    names = ["High\n(n = 52)", "Low excluding very low\n(n = 64)", "Very low\n(n = 79)"]
    for m, model in enumerate(MODELS):
        ordered = subgroup[subgroup["model"].eq(model)].set_index("subgroup").loc[groups]
        ax.bar(positions + (m - 1) * .24, ordered["mean_direction_penalty"],
               width=.22, color=COLORS[m], label=LABELS[m])
    ax.set(xticks=positions, xticklabels=names,
           ylabel="Mean into-English − from-English score\n(chrF++ points)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    save_figure(fig, "figure3_subgroup_penalties")

    fig, ax = plt.subplots(figsize=(7.4, 3.2), layout="constrained")
    statuses = errors[errors["section"].eq("status")].set_index("item")
    status_ids = ["confirmed_error", "acceptable_paraphrase", "uncertain"]
    status_labels = ["Recorded discrepancy", "Acceptable paraphrase", "Uncertain"]
    values = statuses.loc[status_ids, "count"].to_numpy()
    bars = ax.barh(status_labels, values, color=[COLORS[1], COLORS[2], "#888888"])
    ax.bar_label(bars, padding=5, fontsize=12)
    ax.invert_yaxis()
    ax.set_title(f"Recorded audit statuses (n = {n_audited})", fontsize=12)
    ax.set_xlabel("Selected items", fontsize=12)
    ax.set_xlim(0, 44)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    save_figure(fig, "figure4_diagnostic_audit")

    manifest = {
        "description": "Final-manuscript artifacts, generated from saved numerical outputs; no new primary inference.",
        "table_numbers": {"2": "group means and direction gaps", "3": "main inference", "4": "sensitivity checks",
                          "5": "resource subgroups", "6": "purposive diagnostic audit"},
        "figure3_note": "Three disjoint groups: high, low excluding very low, very low. No uncertainty intervals are inferred for subgroup means.",
        "figure4_note": "Purposive low-scoring sample; these counts do not estimate model error prevalence. Labels reproduce the inherited review file.",
        "files": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for folder in (TABLES, FIGURES) for p in sorted(folder.iterdir())
                  if p.suffix in (".csv", ".png", ".pdf")],
    }
    (RESULTS / "final_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(tables)} manuscript tables and 4 figures (PNG 600 dpi + vector PDF).")


if __name__ == "__main__":
    main()
