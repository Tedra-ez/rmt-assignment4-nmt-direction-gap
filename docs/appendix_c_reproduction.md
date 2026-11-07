# Appendix C. Reproduction Test Record

Readable summary of `docs/AppendixC_reproduction_record.json` (the original record of the run).

| Item | Record |
|---|---|
| Person who reproduced the results | Aizada Sembayeva (team member who did not write the analysis code) |
| Date, machine, operating system | 7 November 2026, 23:30 (UTC+05:00); AMD64; Windows 11 (10.0.26200) |
| Python version | 3.12.5 |
| Commit / release tag used | `assignment-4` → commit `ee03474` |
| Commands executed | `git clone https://github.com/Tedra-ez/rmt-assignment4-nmt-direction-gap.git`; `git checkout assignment-4`; `python3.12 scripts/reproduce_check.py` (new virtual environment, `pip install -r requirements.txt`, unit tests, input validation, full analysis, comparison with the published results) |
| Did the main results table reproduce? | **Yes** |
| Differences observed and explanation | All numerical values matched within 1e-9 (10/10 result files, 5/5 saved arrays). Byte-level differences only: Windows line endings (CRLF), last-digit floating-point differences between CPU architectures, and figure rendering on a different OS. |
| Problems in the README fixed after the test | None; the README commands worked as written. |

**Note.** This test used commit `ee03474`. The repository was simplified afterwards: the README was rewritten, helper records were removed, and `analyze.py` was shortened. The calculations did not change. Running `python scripts/analyze.py` on the simplified version produces byte-identical result CSV files (check with `git diff --stat results/`).
