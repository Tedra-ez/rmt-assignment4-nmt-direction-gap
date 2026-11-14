# Directional resource gaps in NLLB-200: a matched FLORES-200 secondary analysis

Final-project reproducibility candidate, Research Methods and Tools, SE-2418.
The author list and CRediT contributions require author confirmation. Preparation
with AI assistance is declared in the accompanying manuscript.

This repository performs a quantitative secondary analysis of published NLLB-200
scores. It does not train models, generate new translations, or use participant
data. Its contribution is a transparent matched reanalysis with language-level
uncertainty and sensitivity checks. See `FINAL_STATUS.md` for remaining checks.

## Research question and main result

How does the high-minus-low resource translation-quality gap differ between
translation into English and translation from English in the selected NLLB-200
evaluations?

We retain 1,206 published chrF++ scores: 201 non-English language-script units,
two directions, and three models. Averaging script variants within each base
language yields 195 languages (52 high resource and 143 low resource). The
public-resource proxy uses at least 1,000,000 examples for high resource; low
resource includes 79 very-low-resource languages with at most 100,000 examples.
The counts do not represent complete NLLB training exposure.

`Gap = mean(high-resource scores) − mean(low-resource scores)`.
`DeltaGap = Gap_from_English − Gap_into_English`.

| Model | Gap into English | Gap from English | DeltaGap [95% interval] | Holm p |
|---|---:|---:|---:|---:|
| Distilled 600M | 9.67 | 12.47 | 2.80 [0.75, 4.68] | 0.00714 |
| Dense 3.3B | 9.53 | 13.23 | 3.70 [1.55, 5.68] | 0.00154 |
| MoE 54.5B | 8.79 | 13.07 | 4.28 [2.14, 6.23] | 0.00048 |

All score differences are in chrF++ points, not percentages of correct
translations. Unrounded values are saved in `results/main_summary.csv`.

## Setup

Unzip the final course package, then enter its `reproducibility` directory.
Python 3.12 is recommended. No GPU, model weights, API keys, or credentials are
needed. Internet access is needed to install packages and, optionally, retrieve
the sentence-audit inputs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` records all package versions from the fresh preparation
environment; `requirements.txt` lists the four direct dependencies.
On Windows, activate with `.venv\Scripts\Activate.ps1`.
Release scope: local tag `final`; not published remotely. The local tag is
created after all final-package files are assembled. No public release URL is
asserted by this package.

## Reproduce every Results table and figure

Run one command from this directory:

```bash
python scripts/reproduce_final.py
```

It validates the source hashes, runs the five statistical unit tests, recomputes
the main analysis, recounts recorded audit labels, and exports manuscript
Tables 2–6 and Figures 1–4. It compares all ten recomputed numerical CSV files
and the saved bootstrap/permutation arrays with their previous values at
absolute/relative tolerance 1e-9. It records actual execution timestamps in
`logs/technical_reproduction.json` and `results/run_log.json`.

Equivalent individual commands are:

```bash
python scripts/analyze.py --validate-only
python -m unittest discover -s scripts -p "test_analysis.py"
python scripts/analyze.py
python scripts/summarize_errors.py
python scripts/export_final_results.py
```

The unit tests use small synthetic arrays to check formulas and statistical
functions. They are separate from rerunning the study on published data. Figure
rendering may vary by operating system; numerical equality is the criterion.

| Manuscript item | Export | Numerical source |
|---|---|---|
| Table 2: group means, direction gaps and intervals | `results/final/table2_group_means.csv` | `results/main_summary.csv` |
| Table 3: gaps, contrasts, intervals and tests | `results/final/table3_main_statistics.csv` | `results/main_summary.csv`, `results/statistical_tests.csv` |
| Table 4: sensitivity checks | `results/final/table4_robustness.csv` | `results/robustness.csv`, `results/leave_one_out.csv` |
| Table 5: subgroup means and distribution diagnostics | `results/final/table5_subgroup_diagnostics.csv` | `results/subgroup_results.csv`, `results/diagnostics.csv` |
| Table 6: recorded diagnostic counts | `results/final/table6_diagnostic_audit.csv` | `results/error_audit/error_summary.csv`, `results/error_audit/reviewed_round_01.csv` |
| Figure 1: resource gaps by direction | `figures/final/figure1_resource_gaps.png` | `results/main_summary.csv` |
| Figure 2: DeltaGap and language-bootstrap intervals | `figures/final/figure2_delta_gap.png` | `results/main_summary.csv` |
| Figure 3: disjoint resource-subgroup penalties | `figures/final/figure3_subgroup_penalties.png` | `results/subgroup_results.csv` |
| Figure 4: selected diagnostic cases | `figures/final/figure4_diagnostic_audit.png` | `results/error_audit/error_summary.csv` |

Every final figure is also exported as a vector PDF; PNG exports are 600 dpi.
The related-work table is a literature synthesis rather than an empirical
result. Older figures in `figures/` are retained supplementary diagnostics.
`results/final_export_manifest.json` hashes the generated manuscript artifacts.
For a per-language source location, inspect `results/per_language_scores.csv`,
including its `raw_file`, `raw_csv_rows`, and `script_ids` columns.

## Analysis settings and interpretation

`protocol/protocol_v1.json` is the current configuration, prepared after the
Assignment 4 results were known. It retains seed 20261108, 20,000 stratified
language bootstrap resamples, 100,000 resource-label permutations, and Holm
adjustment across three model-specific tests. Resampling keeps both directions
and all model scores paired within each sampled language. Permutation group
sizes remain 52 and 143; two-sided p-values use the plus-one correction.

This is **not a preregistration**. The historical Assignment 4 configuration is
retained under `protocol/historical/` for provenance only; its dates do not
independently establish when analytical decisions were made.

Language-bootstrap intervals describe resampling uncertainty across this
benchmark's languages. They do not capture training-seed or sentence-level
variation. Related languages may be dependent, and the permutation test assumes
exchangeability under its null. Resource level was not randomly assigned; no
causal effect of data volume, architecture, or parameter count is established.
The three systems share the NLLB family. Their matched coverage makes them
useful comparison conditions but does not isolate the effect of model size.

The Assignment 3 pilot used separate Nature development-set chrF++ aggregates
and Cui et al. spBLEU values. Those pilot scales and resource taxonomies are not
pooled into the current devtest analysis.

## Reproduce diagnostic selection from public inputs

```bash
python scripts/fetch_error_data.py
python scripts/select_error_cases.py
python scripts/summarize_errors.py
python scripts/export_final_results.py
```

The downloader retrieves six selected members of the upstream MoE translation
archive using HTTP byte ranges and the FLORES reference archive. It verifies
hashes and line counts. The scoring script evaluates 6,072 sentences with
SacreBLEU 2.5.1 chrF++ and selects the ten lowest scores per language, breaking
ties by sentence ID. It does not generate manual review labels.

The inherited review file records 37 discrepancies, 8 acceptable paraphrases,
and 15 uncertain cases among 60 selected items. Of the 37 recorded discrepancies,
25 are meaning substitutions and 10 omissions/truncations. These are diagnostic
counts in a deliberately difficult sample, not population error rates. The
audit covers only translation into English and cannot directly explain the
directional contrast. There is no qualified bilingual validation or measured
inter-rater agreement. Source/reference mismatch is suspected in 23 items,
including 10 recorded discrepancies. Review labels and author involvement
require author confirmation for publication.

## Availability, licences, and reproducibility status

All main score inputs and metadata are included, with upstream URLs and SHA-256
hashes in `data/source_manifest.json`. Sentence-audit provenance is in
`data/error_audit/provenance.json`; full original sentence files can be retrieved
by the optional commands and are ignored by Git. Selected text is present in
the review/candidate CSV files and remains third-party content.

The inherited MIT licence applies to original project code/documentation only.
Copied fairseq code retains its MIT notice. FLORES text is CC BY-SA 4.0. NLLB
models are distributed under CC BY-NC 4.0; that model notice is retained, but
an explicit standalone licence for the published score CSVs and generated
translations has not been independently confirmed. No MIT relicensing of those
files is claimed. See `THIRD_PARTY_NOTICES.md`.

A fresh assistant technical reproduction is recorded in `logs/`. It is not an
outside-team human reproduction. The earlier Aizada Appendix C concerned a
team member and an earlier repository version, so it does not complete the
final publication-readiness requirement. No final author approval, similarity
check, journal submission, or published `final` release is claimed here.
