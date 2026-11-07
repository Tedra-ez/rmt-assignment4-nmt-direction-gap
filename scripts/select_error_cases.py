#!/usr/bin/env python3
"""Select the 60 sentences for the error analysis (Appendix B).

Rule from the frozen protocol:
  * model MoE 54.5B, direction into English, six fixed low-resource languages;
  * score every devtest sentence with sentence-level chrF++ (SacreBLEU 2.5.1);
  * take the 10 lowest-scoring sentences per language (ties -> lower sentence ID);
  * list them in round-robin language order (60 candidates).

A low score is only a reason to look at a sentence, not proof of an error.
The manual review of these 60 cases is stored in results/error_audit/reviewed_round_01.csv.

Run scripts/fetch_error_data.py first to download the translations and references.
Usage:  python scripts/select_error_cases.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data" / "error_audit"
OUT_DIR = ROOT / "results" / "error_audit"
LANGUAGES = ("ace_Latn", "asm_Beng", "khm_Khmr", "lao_Laoo", "mya_Mymr", "som_Latn")
TARGET = "eng_Latn"
SENTENCES = 1012
PER_LANGUAGE = 10

SCORE_FIELDS = ["record_id", "model", "source_language", "target_language", "sentence_id",
                "sentence_chrfpp", "rank_within_language", "source_file", "reference_file", "hypothesis_file"]
CANDIDATE_FIELDS = ["screening_round", "screening_order", *SCORE_FIELDS,
                    "source_sentence_id", "reference_sentence_id", "hypothesis_sentence_id",
                    "source_text", "reference_text", "hypothesis_text", "annotation_status",
                    "dominant_error_type", "evidence", "possible_cause", "uncertainty",
                    "rejection_reason", "reviewer", "student_validation_status"]


def load_texts() -> dict:
    """Read the downloaded files and check them against data/error_audit/provenance.json."""
    provenance = json.loads((AUDIT_DIR / "provenance.json").read_text())
    texts = {}
    for row in provenance["files"]:
        path = AUDIT_DIR / row["path"]
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run: python scripts/fetch_error_data.py")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SystemExit(f"Hash mismatch for {row['path']}")
        lines = data.decode("utf-8").splitlines()
        if len(lines) != SENTENCES:
            raise SystemExit(f"Expected {SENTENCES} lines in {row['path']}")
        texts[row["path"]] = lines
    return texts


def score_sentences(texts: dict):
    from sacrebleu.metrics import CHRF
    scorer = CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False, eps_smoothing=False)
    rel = str(AUDIT_DIR.relative_to(ROOT))
    references = texts[f"references/{TARGET}.devtest"]
    ranked, all_rows = {}, []
    for language in LANGUAGES:
        hyp_path = f"hypotheses/flores200-{language}-{TARGET}-devtest.hyp"
        rows = []
        for sentence_id, hypothesis in enumerate(texts[hyp_path], start=1):
            score = scorer.sentence_score(hypothesis, [references[sentence_id - 1]]).score
            rows.append({"record_id": f"{language}-{TARGET}:{sentence_id:04d}", "model": "moe_54b",
                         "source_language": language, "target_language": TARGET,
                         "sentence_id": sentence_id, "sentence_chrfpp": score,
                         "source_file": f"{rel}/references/{language}.devtest",
                         "reference_file": f"{rel}/references/{TARGET}.devtest",
                         "hypothesis_file": f"{rel}/{hyp_path}"})
        rows.sort(key=lambda r: (r["sentence_chrfpp"], r["sentence_id"]))   # lowest first, tie -> ID
        for rank, row in enumerate(rows, start=1):
            row["rank_within_language"] = rank
            row["sentence_chrfpp"] = format(row["sentence_chrfpp"], ".17g")
        ranked[language] = rows
        all_rows.extend(sorted(rows, key=lambda r: r["sentence_id"]))
    return all_rows, ranked


def select_candidates(ranked: dict, texts: dict) -> list[dict]:
    candidates = []
    for index in range(PER_LANGUAGE):            # rank 1 for all languages, then rank 2, ...
        for language in LANGUAGES:
            item = ranked[language][index]
            sid = item["sentence_id"]
            candidates.append({
                "screening_round": 1, "screening_order": len(candidates) + 1, **item,
                "source_sentence_id": f"{language}.devtest:{sid}",
                "reference_sentence_id": f"{TARGET}.devtest:{sid}",
                "hypothesis_sentence_id": f"flores200-{language}-{TARGET}-devtest.hyp:{sid}",
                "source_text": texts[f"references/{language}.devtest"][sid - 1],
                "reference_text": texts[f"references/{TARGET}.devtest"][sid - 1],
                "hypothesis_text": texts[f"hypotheses/flores200-{language}-{TARGET}-devtest.hyp"][sid - 1],
                "annotation_status": "unreviewed", "dominant_error_type": "", "evidence": "",
                "possible_cause": "", "uncertainty": "", "rejection_reason": "", "reviewer": "",
                "student_validation_status": "pending",
            })
    return candidates


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stream.getvalue().encode("utf-8"))


def main() -> None:
    texts = load_texts()
    all_rows, ranked = score_sentences(texts)
    candidates = select_candidates(ranked, texts)
    write_csv(OUT_DIR / "all_sentence_scores.csv", all_rows, SCORE_FIELDS)
    write_csv(OUT_DIR / "candidates_round_01.csv", candidates, CANDIDATE_FIELDS)
    print(f"Scored {len(all_rows)} sentences; selected {len(candidates)} candidates "
          f"-> results/error_audit/candidates_round_01.csv")


if __name__ == "__main__":
    main()
