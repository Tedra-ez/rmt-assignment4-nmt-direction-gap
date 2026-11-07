# Translation quality gaps by resource level and translation direction

**Research Methods and Tools, Assignment 4**
Fariza Ismagambetova, Aizada Sembayeva, Zhanibek Yskak (Group SE-2418)

## Research question

How does the translation-quality gap between low- and high-resource languages differ between translation **into** English and translation **from** English in multilingual NMT evaluations?

- **H1:** the gap differs between the two directions.
- **H0:** the gap does not differ.

## What we did

This is a **quantitative secondary analysis**. We did not train models or translate anything ourselves. We used Meta's published chrF++ scores for three NLLB-200 models on the FLORES-200 devtest set (1,012 sentences per direction):

| Model | Role |
|---|---|
| NLLB-200 Distilled 600M | simple baseline (small existing system) |
| NLLB-200 Dense 3.3B | strong baseline |
| NLLB-200 MoE 54.5B | main model |

Steps:

1. Keep the into-English and from-English directions: 201 language-script units × 2 directions × 3 models = **1,206 scores**.
2. Average script variants of the same language, which gives **195 languages: 52 high-resource** (≥ 1,000,000 training examples) and **143 low-resource**.
3. For each direction compute `Gap = mean(High) − mean(Low)`.
4. Primary outcome: `DeltaGap = Gap_from_English − Gap_into_English`. A positive value means the gap is larger from English.
5. Estimate uncertainty with a 95% bootstrap CI (20,000 resamples of languages) and test it with a permutation test (100,000 label shuffles), using Holm correction across the 3 models. Seed `20261108`.
6. Additional analysis: sensitivity checks, subgroups (Very low ≤ 100,000 examples), and an error analysis of 60 translations.

**Main result (MoE 54.5B):** the gap is 8.79 chrF++ points into English and 13.07 from English, so DeltaGap = **4.28**, 95% CI [2.14, 6.23], Holm p = 0.00048. All three models show a positive DeltaGap, so H1 is supported.

## Repository structure

```
├── README.md
├── requirements.txt            Python package versions
├── analysis.ipynb              notebook with the saved tables and figures
├── protocol/                   frozen protocol (Appendix A) and its record
├── data/
│   ├── raw/                    published NLLB score files and resource labels
│   ├── source_manifest.json    download links and SHA-256 hashes of the inputs
│   └── error_audit/            information about the error-analysis input files
├── scripts/
│   ├── analyze.py              main analysis: all tables and figures
│   ├── test_analysis.py        unit tests on small synthetic data
│   ├── fetch_error_data.py     downloads translations for the error analysis
│   ├── select_error_cases.py   selects the 60 error-analysis sentences
│   └── summarize_errors.py     counts error types (Appendix B)
├── results/                    output tables (CSV), bootstrap/permutation arrays, logs
│   └── error_audit/            60 reviewed cases and the error summary
├── figures/                    figures 1–4 (PNG)
├── report/                     Assignment 4 report (DOCX)
└── docs/                       Appendix C reproduction record and third-party licences
```

## How to reproduce the main results

Requirements: Python 3.12 and internet access to install packages. No GPU, model weights or API keys are needed.

```bash
git clone https://github.com/Tedra-ez/rmt-assignment4-nmt-direction-gap.git
cd rmt-assignment4-nmt-direction-gap
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the steps:

```bash
# 1. Check that the input files match the frozen protocol (hashes, 1,206 scores)
python scripts/analyze.py --validate-only

# 2. Unit tests of the statistical functions
python -m unittest discover -s scripts -p "test_analysis.py"

# 3. Full analysis (takes a few seconds)
python scripts/analyze.py

# 4. Error-analysis counts for Appendix B
python scripts/summarize_errors.py
```

Expected output of step 3:

```
MoE 54.5B: DeltaGap = 4.28 [2.14, 6.23], Holm p = 0.00048
```

Step 3 rewrites the CSV files in `results/` and the figures in `figures/`. The numbers should be identical to the committed files; check with `git diff results/`. Each run also writes `results/run_log.json` with the date, environment and run time.

## Error analysis (optional, needs internet)

The 60 selected and reviewed sentences are already in `results/error_audit/`. To repeat the selection from the original files:

```bash
python scripts/fetch_error_data.py      # downloads 6 translation files + FLORES references
python scripts/select_error_cases.py    # scores every sentence and selects the 60 cases
```

Selection rule: model MoE 54.5B, direction into English, six languages (`ace_Latn`, `asm_Beng`, `khm_Khmr`, `lao_Laoo`, `mya_Mymr`, `som_Latn`), the 10 lowest sentence-level chrF++ scores per language. The manual decisions (error, acceptable paraphrase, or uncertain, plus the error type and evidence) are recorded in `results/error_audit/reviewed_round_01.csv`.

## Where each number in the report comes from

| Report | File |
|---|---|
| Table 2: group means and gaps | `results/main_summary.csv` |
| Table 3: DeltaGap, 95% CI, p-values | `results/main_summary.csv`, `results/statistical_tests.csv` |
| CIs for every outcome | `results/main_results_long.csv` (raw draws in `bootstrap_draws.npz`) |
| Table 4: sensitivity checks | `results/robustness.csv`, `results/leave_one_out.csv` |
| Subgroups, medians, IQR | `results/subgroup_results.csv`, `results/diagnostics.csv` |
| Score of one language and its row in the raw file | `results/per_language_scores.csv` (`raw_csv_rows`) |
| Error analysis (Section 5, Appendix B) | `results/error_audit/error_summary.csv`, `reviewed_round_01.csv` |
| Run time, Python version, hardware | `results/run_log_2026-11-07.json` |
| Figures 1–2 | `figures/figure_1_resource_gaps.png`, `figures/figure_2_delta_gap.png` |

`results/run_log_2026-11-07.json` is the log of the run reported in the paper (macOS ARM64, Python 3.12.14, 3.52 s). The code was tidied afterwards without changing any calculation. Running `analyze.py` again reproduces the same tables.

## Protocol

The frozen decisions (models, metric, seed, number of resamples, tests, decision rule, error-analysis rule) are in `protocol/protocol_v1.json` and the Appendix A form in `protocol/`. They were fixed on 7 November 2026 before the final analysis (`protocol/local_freeze_record.json`). `analyze.py` reads the seed and the number of resamples from this file and checks the input hashes against it.

## Reproduction test (Appendix C)

Aizada Sembayeva, who did not write the analysis code, reproduced the main results on Windows 11 in a new virtual environment on 7 November 2026. The result was **Yes**: all numerical values matched. The record is in `docs/appendix_c_reproduction.md`, and the original record is `docs/AppendixC_reproduction_record.json`.

## Release

The submitted version is tagged `assignment-4`.

## Data sources and licences

- NLLB-200 published scores and translations: Meta AI, CC BY-NC 4.0 ([fairseq NLLB](https://github.com/facebookresearch/fairseq/tree/nllb)).
- FLORES-200 benchmark: CC BY-SA 4.0 ([flores200](https://github.com/facebookresearch/flores/tree/main/flores200)).
- `data/raw/flores200_public.py` and `data/raw/nllb_grouping_original.py` are copied from fairseq (MIT) and contain the training-example counts used for the resource groups.

Our own code is under the MIT licence (`LICENSE`). Third-party licences are in `docs/licenses/` and `THIRD_PARTY_NOTICES.md`.
