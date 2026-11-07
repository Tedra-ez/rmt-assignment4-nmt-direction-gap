#!/usr/bin/env python3
"""Main analysis for Assignment 4.

Steps:
  1. Check that the input files match the frozen protocol (SHA-256 hashes).
  2. Load chrF++ scores for three NLLB-200 models and keep only the
     into-English and from-English directions (402 per model).
  3. Average script variants within each language (201 units -> 195 languages).
  4. Compute group means, Gap and DeltaGap for each model.
  5. Bootstrap 95% confidence intervals and permutation p-values (Holm).
  6. Robustness checks, subgroup results and figures.

Usage:
    python scripts/analyze.py                 # full analysis
    python scripts/analyze.py --validate-only # only check the inputs
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol" / "protocol_v1.json"

MODEL_IDS = ("distilled_600m", "dense_3b", "moe_54b")
SHORT_LABELS = ("Distilled 600M", "Dense 3.3B", "MoE 54.5B")
DIRECTIONS = ("into_english", "from_english")
OUTCOMES = ("high_into", "low_into", "high_from", "low_from",
            "gap_into", "gap_from", "delta_gap")


# ---------------------------------------------------------------- helpers

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# ------------------------------------------------------- 1. input checks

def read_protocol() -> dict:
    """Load the frozen protocol and verify every input file against its hash."""
    protocol = json.loads(PROTOCOL.read_text())
    check(tuple(m["id"] for m in protocol["models"]) == MODEL_IDS, "Unexpected model list.")
    check(protocol["metric"] == "chrF++", "Unexpected metric.")
    for entry in protocol["data_manifest"]:
        path = ROOT / entry["file"]
        check(path.is_file(), f"Missing input file: {entry['file']}")
        check(sha256(path) == entry["sha256"], f"Input file changed (hash mismatch): {entry['file']}")
    return protocol


# ------------------------------------------------------ 2. load the data

def load_scores(protocol: dict) -> pd.DataFrame:
    """Return 1,206 rows: model x language-script x direction, with resource labels."""
    groups = pd.read_csv(ROOT / "data/raw/resource_groups.csv", keep_default_na=False)
    counts = pd.to_numeric(groups["public_training_examples"])
    # Resource rule from the protocol: High >= 1,000,000 examples; Very low <= 100,000.
    check((groups["resource_group"] == np.where(counts >= 1_000_000, "high", "low")).all(),
          "Resource labels disagree with the 1,000,000 threshold.")
    groups["very_low_nested"] = groups["very_low_nested"].astype(str).str.lower().eq("true")
    check((groups["very_low_nested"] == counts.le(100_000)).all(),
          "Very-low labels disagree with the 100,000 threshold.")
    groups = groups[groups["language_script"] != "eng_Latn"].copy()
    groups["language"] = groups["language_script"].str.split("_").str[0]
    check(len(groups) == 201, "Expected 201 non-English language-script units.")

    frames = []
    for model in protocol["models"]:
        raw = pd.read_csv(ROOT / model["file"], keep_default_na=False)
        check(not raw["direction"].duplicated().any(), f"{model['id']}: duplicate directions.")
        parts = raw["direction"].str.split("-", expand=True)
        source, target = parts[0], parts[1]
        into = target.eq("eng_Latn") & source.ne("eng_Latn")    # xx -> English
        out = source.eq("eng_Latn") & target.ne("eng_Latn")     # English -> xx
        chosen = raw.loc[into | out, ["direction", "chrf++"]].copy()
        rows = chosen.index
        chosen["raw_csv_row"] = rows + 2                         # 1-based row number incl. header
        chosen["raw_file"] = model["file"]
        chosen["source_direction"] = chosen["direction"]
        chosen["language_script"] = np.where(into.loc[rows], source.loc[rows], target.loc[rows])
        chosen["direction"] = np.where(into.loc[rows], DIRECTIONS[0], DIRECTIONS[1])
        chosen["score"] = pd.to_numeric(chosen.pop("chrf++"))
        check(len(chosen) == 402, f"{model['id']}: expected 402 English-centric directions.")
        check(chosen["score"].between(0, 100).all(), f"{model['id']}: invalid chrF++ values.")
        chosen = chosen.merge(groups, on="language_script", how="left", validate="many_to_one")
        check(chosen["resource_group"].notna().all(), "Language without a resource label.")
        chosen["model"] = model["id"]
        frames.append(chosen)
    records = pd.concat(frames, ignore_index=True)
    check(len(records) == 1206, "Expected 1,206 scores in total.")
    return records


# ------------------------------------------ 3. one score per base language

def aggregate_languages(records: pd.DataFrame) -> pd.DataFrame:
    """Average script variants (e.g. two Acehnese scripts) into one language score."""
    keys = ["model", "language", "resource_group", "very_low_nested", "direction"]
    return records.groupby(keys, sort=True, as_index=False).agg(
        score=("score", "mean"), script_count=("language_script", "nunique"),
        script_ids=("language_script", lambda x: ";".join(sorted(x))),
        raw_csv_rows=("raw_csv_row", lambda x: ";".join(map(str, sorted(x)))),
        raw_file=("raw_file", "first"),
    )


def to_cube(languages: pd.DataFrame):
    """Array of shape (language, model, direction) plus High / Very-low masks."""
    ids = sorted(languages["language"].unique())
    cube = np.empty((len(ids), len(MODEL_IDS), 2))
    for m, model in enumerate(MODEL_IDS):
        subset = languages[languages["model"] == model]
        for d, direction in enumerate(DIRECTIONS):
            ordered = subset[subset["direction"] == direction].set_index("language").reindex(ids)
            cube[:, m, d] = ordered["score"].to_numpy()
    info = languages.drop_duplicates("language").set_index("language").reindex(ids)
    high = info["resource_group"].eq("high").to_numpy()
    very_low = info["very_low_nested"].to_numpy(dtype=bool)
    check((int(high.sum()), int((~high).sum())) == (52, 143), "Expected 52 High and 143 Low languages.")
    return cube, high, ids, very_low


# ------------------------------------------------ 4. Gap and DeltaGap

def outcomes_from_means(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    gap = high - low                                   # Gap = High mean - Low mean
    return np.stack((high[..., 0], low[..., 0], high[..., 1], low[..., 1],
                     gap[..., 0], gap[..., 1],
                     gap[..., 1] - gap[..., 0]),       # DeltaGap = Gap_from - Gap_into
                    axis=-1)


def group_outcomes(cube: np.ndarray, high: np.ndarray) -> np.ndarray:
    return outcomes_from_means(cube[high].mean(axis=0), cube[~high].mean(axis=0))


# ------------------------------------------------- 5. statistics

def bootstrap(cube, high, resamples, rng, batch_size=256):
    """Stratified bootstrap: resample languages within High and within Low.
    A sampled language keeps both directions and all three model scores."""
    hi, lo = cube[high], cube[~high]
    result = np.empty((resamples, cube.shape[1], len(OUTCOMES)))
    for begin in range(0, resamples, batch_size):
        size = min(batch_size, resamples - begin)
        hi_idx = rng.integers(len(hi), size=(size, len(hi)))
        lo_idx = rng.integers(len(lo), size=(size, len(lo)))
        result[begin:begin + size] = outcomes_from_means(hi[hi_idx].mean(axis=1), lo[lo_idx].mean(axis=1))
    return result


def permutation_test(difference, high, resamples, rng, batch_size=512):
    """Two-sided permutation test.

    difference = into-English score - from-English score for each language.
    Statistic T = mean(difference, Low) - mean(difference, High), which equals DeltaGap.
    Resource labels are shuffled with group sizes fixed (52 / 143).
    p = (1 + number of |T_perm| >= |T_obs|) / (resamples + 1).
    """
    n_high, n_low = int(high.sum()), int((~high).sum())
    observed = difference[~high].mean(axis=0) - difference[high].mean(axis=0)
    total = difference.sum(axis=0)
    draws = np.empty((resamples, difference.shape[1]))
    for begin in range(0, resamples, batch_size):
        size = min(batch_size, resamples - begin)
        labels = np.argpartition(rng.random((size, len(high))), n_high - 1, axis=1)[:, :n_high]
        high_sum = difference[labels].sum(axis=1)
        draws[begin:begin + size] = (total - high_sum) / n_low - high_sum / n_high
    tolerance = 100 * np.finfo(float).eps * np.maximum(1.0, np.abs(observed))
    extreme = (np.abs(draws) >= np.abs(observed) - tolerance).sum(axis=0)
    p = (extreme + 1) / (resamples + 1)
    return p, extreme, draws


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm correction: sort p-values, multiply by 3, 2, 1 and keep them non-decreasing."""
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values, dtype=float)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p_values[order] * np.arange(len(order), 0, -1)))
    return adjusted


# ----------------------------------------- 6. robustness and subgroups

def additional_analyses(records, cube, high, identifiers, very_low):
    observed = group_outcomes(cube, high)[:, -1]
    d = cube[:, :, 0] - cube[:, :, 1]
    full_hi, full_lo = d[high].sum(axis=0), d[~high].sum(axis=0)

    # Leave one language out (195 recalculations per model).
    loo_rows = []
    for i, language in enumerate(identifiers):
        n_hi = int(high.sum()) - int(high[i])
        n_lo = int((~high).sum()) - int(not high[i])
        hi_sum = full_hi - (d[i] if high[i] else 0)
        lo_sum = full_lo - (d[i] if not high[i] else 0)
        value = lo_sum / n_lo - hi_sum / n_hi
        for m, model in enumerate(MODEL_IDS):
            loo_rows.append({"model": model, "excluded_language": language,
                             "excluded_resource_group": "high" if high[i] else "low",
                             "n_high": n_hi, "n_low": n_lo, "delta_gap": value[m],
                             "change_from_full": value[m] - observed[m],
                             "same_sign_as_full": bool(np.sign(value[m]) == np.sign(observed[m]))})
    loo = pd.DataFrame(loo_rows)

    # Equal weight per script unit, and medians instead of means.
    robustness = []
    for m, model in enumerate(MODEL_IDS):
        means = records[records["model"] == model].groupby(["resource_group", "direction"])["score"].mean()
        script_delta = (means["high", "from_english"] - means["low", "from_english"]
                        - means["high", "into_english"] + means["low", "into_english"])
        median_diff = np.median(d[~high, m]) - np.median(d[high, m])
        part = loo[loo["model"] == model]
        robustness.append({"model": model, "primary_delta_gap": observed[m],
                           "script_weighted_delta_gap": script_delta,
                           "median_difference": median_diff,
                           "leave_one_out_min": part["delta_gap"].min(),
                           "leave_one_out_max": part["delta_gap"].max(),
                           "leave_one_out_sign_stable_count": int(part["same_sign_as_full"].sum()),
                           "leave_one_out_total": len(part)})

    # Subgroups: High, Low, Very low, Low without Very low.
    subgroups, diagnostics = [], []
    for label, mask in (("high", high), ("low", ~high), ("very_low", very_low),
                        ("low_excluding_very_low", ~high & ~very_low)):
        for m, model in enumerate(MODEL_IDS):
            values = d[mask, m]
            q1, q3 = np.quantile(values, [.25, .75])
            subgroups.append({"model": model, "subgroup": label, "n_languages": int(mask.sum()),
                              "mean_into": cube[mask, m, 0].mean(), "mean_from": cube[mask, m, 1].mean(),
                              "mean_direction_penalty": values.mean()})
            diagnostics.append({"model": model, "subgroup": label, "n_languages": len(values),
                                "mean_d": values.mean(), "sd_d": values.std(ddof=1),
                                "median_d": np.median(values), "q1_d": q1, "q3_d": q3,
                                "iqr_d": q3 - q1, "min_d": values.min(), "max_d": values.max(),
                                "positive_d_count": int((values > 0).sum()),
                                "zero_d_count": int((values == 0).sum()),
                                "negative_d_count": int((values < 0).sum())})
    return pd.DataFrame(robustness), loo, pd.DataFrame(subgroups), pd.DataFrame(diagnostics)


# ------------------------------------------------------------- figures

def make_figures(directory, observed, intervals, cube, high, robustness, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})
    directory.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(MODEL_IDS))

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Figure 1: gap into vs from English, with 95% CIs.
    fig, ax = plt.subplots(figsize=(8.2, 4.7), layout="constrained")
    for offset, idx, color, label in ((-.18, 4, "#0072B2", "Into English"), (.18, 5, "#D55E00", "From English")):
        pos = x + offset
        ax.bar(pos, observed[:, idx], width=.32, color=color, label=label)
        ax.vlines(pos, intervals[0, :, idx], intervals[1, :, idx], color="#222222", linewidth=1)
        ax.plot(pos, intervals[0, :, idx], "_", color="#222222", markersize=8)
        ax.plot(pos, intervals[1, :, idx], "_", color="#222222", markersize=8)
    ax.set(xticks=x, xticklabels=SHORT_LABELS, ylabel="High − low gap (chrF++ points)",
           title="Resource gaps by translation direction")
    ax.axhline(0, color="gray", linewidth=.7)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.18)
    save(fig, "figure_1_resource_gaps")

    # Figure 2: DeltaGap with 95% CIs.
    fig, ax = plt.subplots(figsize=(8.2, 4.1), layout="constrained")
    ax.hlines(x, intervals[0, :, 6], intervals[1, :, 6], color="#0072B2", linewidth=2)
    ax.scatter(observed[:, 6], x, color="#0072B2", s=45, zorder=3)
    ax.plot(intervals[0, :, 6], x, "|", color="#0072B2", markersize=10)
    ax.plot(intervals[1, :, 6], x, "|", color="#0072B2", markersize=10)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set(yticks=x, yticklabels=SHORT_LABELS,
           xlabel="ΔGap: from-English gap − into-English gap (chrF++ points)",
           title="Directional gap difference with 95% language-bootstrap intervals")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.18)
    save(fig, "figure_2_delta_gap")

    # Figure 3: per-language direction differences.
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 4.5), sharey=True, layout="constrained")
    difference = cube[:, :, 0] - cube[:, :, 1]
    rng = np.random.default_rng(seed)
    for m, ax in enumerate(axes):
        for group_pos, mask, color in ((0, high, "#0072B2"), (1, ~high, "#D55E00")):
            jitter = rng.uniform(-.15, .15, size=int(mask.sum()))
            ax.scatter(group_pos + jitter, difference[mask, m], s=12, color=color, alpha=.55, linewidths=0)
            median = np.median(difference[mask, m])
            ax.plot([group_pos - .22, group_pos + .22], [median, median], color="#111111", linewidth=2)
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set(title=SHORT_LABELS[m], xticks=[0, 1],
               xticklabels=[f"High\n(n={high.sum()})", f"Low\n(n={(~high).sum()})"])
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Into-English score − from-English score\n(chrF++ points)")
    fig.suptitle("Direction differences for all base-language units; black marks show medians")
    save(fig, "figure_3_language_differences")

    # Figure 4: robustness checks.
    fig, ax = plt.subplots(figsize=(8.2, 4.3), layout="constrained")
    ordered = robustness.set_index("model").loc[list(MODEL_IDS)]
    ax.hlines(x, ordered["leave_one_out_min"], ordered["leave_one_out_max"], color="#555555",
              linewidth=3, label="Leave-one-language-out range")
    ax.scatter(ordered["primary_delta_gap"], x, marker="o", s=40, label="Primary mean difference", zorder=3)
    ax.scatter(ordered["script_weighted_delta_gap"], x + .12, marker="s", s=35,
               label="Equal script-unit weights", zorder=3)
    ax.scatter(ordered["median_difference"], x - .12, marker="D", s=30,
               label="Difference of group medians", zorder=3)
    ax.axvline(0, color="gray", linestyle="--", linewidth=.8)
    ax.set(yticks=x, yticklabels=SHORT_LABELS, xlabel="Directional contrast (chrF++ points)",
           title="Predeclared robustness checks")
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.grid(axis="x", alpha=.18)
    ax.legend(frameon=False, loc="best", fontsize=8)
    save(fig, "figure_4_robustness")


# ------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description="Assignment 4 main analysis")
    parser.add_argument("--validate-only", action="store_true", help="Only check the input files.")
    args = parser.parse_args()

    started = time.perf_counter()
    protocol = read_protocol()
    records = load_scores(protocol)
    if args.validate_only:
        print("Input check passed: 3 models x 402 directions = 1,206 scores; hashes match the protocol.")
        return 0

    settings = protocol["primary_analysis"]       # seed 20261108, 20,000 bootstrap, 100,000 permutations
    languages = aggregate_languages(records)
    cube, high, identifiers, very_low = to_cube(languages)
    observed = group_outcomes(cube, high)
    difference = cube[:, :, 0] - cube[:, :, 1]

    # One seed -> three independent random streams (bootstrap, permutation, figure jitter).
    seeds = np.random.SeedSequence(settings["seed"]).spawn(3)
    boot = bootstrap(cube, high, settings["bootstrap_resamples"], np.random.default_rng(seeds[0]))
    intervals = np.quantile(boot, [.025, .975], axis=0, method="linear")
    raw_p, extreme, permutations = permutation_test(
        difference, high, settings["permutation_resamples"], np.random.default_rng(seeds[1]))
    adjusted = holm_adjust(raw_p)

    long_rows, summary_rows, tests = [], [], []
    for m, model in enumerate(MODEL_IDS):
        summary = {"model": model, "label": protocol["models"][m]["label"], "role": protocol["models"][m]["role"],
                   "metric": "chrF++", "n_high": int(high.sum()), "n_low": int((~high).sum())}
        for k, outcome in enumerate(OUTCOMES):
            row = {"model": model, "outcome": outcome, "estimate": observed[m, k],
                   "ci_lower": intervals[0, m, k], "ci_upper": intervals[1, m, k],
                   "metric": "chrF++", "bootstrap_resamples": len(boot), "ci_level": .95}
            long_rows.append(row)
            summary[outcome] = row["estimate"]
            summary[outcome + "_ci_lower"] = row["ci_lower"]
            summary[outcome + "_ci_upper"] = row["ci_upper"]
        # Decision rule from the protocol: Holm p < .05 AND the 95% CI excludes zero.
        excludes_zero = bool(intervals[0, m, -1] > 0 or intervals[1, m, -1] < 0)
        status = ("supported within benchmark assumptions"
                  if adjusted[m] < .05 and excludes_zero else "not supported")
        tests.append({"model": model, "statistic": observed[m, -1],
                      "test": "two-sided resource-label permutation association",
                      "extreme_permutations": int(extreme[m]), "permutations": len(permutations),
                      "p_unadjusted": raw_p[m], "p_holm": adjusted[m], "holm_family_size": len(MODEL_IDS),
                      "ci_excludes_zero": excludes_zero, "hypothesis_status": status,
                      "association_direction": "larger_from_English" if observed[m, -1] > 0
                      else "larger_into_English" if observed[m, -1] < 0 else "equal"})
        summary.update(p_unadjusted=raw_p[m], p_holm=adjusted[m], hypothesis_status=status)
        summary_rows.append(summary)

    robustness, loo, subgroups, diagnostics = additional_analyses(records, cube, high, identifiers, very_low)

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    tables = {"per_script_scores.csv": records, "per_language_scores.csv": languages,
              "main_summary.csv": pd.DataFrame(summary_rows), "main_results_long.csv": pd.DataFrame(long_rows),
              "statistical_tests.csv": pd.DataFrame(tests), "robustness.csv": robustness,
              "leave_one_out.csv": loo, "subgroup_results.csv": subgroups, "diagnostics.csv": diagnostics}
    for name, table in tables.items():
        table.to_csv(results / name, index=False, float_format="%.12g")
    np.savez_compressed(results / "bootstrap_draws.npz", values=boot,
                        models=np.array(MODEL_IDS), outcomes=np.array(OUTCOMES))
    np.savez_compressed(results / "permutation_statistics.npz", values=permutations,
                        models=np.array(MODEL_IDS))
    make_figures(ROOT / "figures", observed, intervals, cube, high, robustness, seeds[2])

    run_log = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib")},
        "seed": settings["seed"],
        "bootstrap_resamples": settings["bootstrap_resamples"],
        "permutation_resamples": settings["permutation_resamples"],
        "protocol_sha256": sha256(PROTOCOL),
    }
    (results / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")

    moe = summary_rows[-1]
    print(f"Done in {run_log['elapsed_seconds']} s. Main table: results/main_summary.csv")
    print(f"MoE 54.5B: DeltaGap = {moe['delta_gap']:.2f} "
          f"[{moe['delta_gap_ci_lower']:.2f}, {moe['delta_gap_ci_upper']:.2f}], Holm p = {moe['p_holm']:.5f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        sys.exit(2)
