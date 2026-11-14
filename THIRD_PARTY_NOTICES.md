# Third-party notices and licence scope

`LICENSE` retains the original project's MIT notice. It covers original
analysis code and project documentation only; the named historical copyright
holders are retained as a licence notice, not as verification of current
manuscript authorship or of each person's contribution.

| Material | Origin | Licence/status |
|---|---|---|
| Copied `flores200_public.py` and `nllb_grouping_original.py` | Meta fairseq commit `ec0ae71a24fbc2f42252f37735f59dd6547eba38` | MIT; copied notice retained in `docs/licenses/LICENSE_FAIRSEQ_CODE_MIT.txt` |
| FLORES-200 source/reference sentences, including excerpts in candidate/review CSVs | [Official FLORES repository](https://github.com/facebookresearch/flores/tree/main/flores200) | CC BY-SA 4.0; notice in `docs/licenses/LICENSE_FLORES_CC_BY_SA_4.txt` |
| Published NLLB score CSVs and translations used for secondary analysis | [Official NLLB fairseq project](https://github.com/facebookresearch/fairseq/tree/nllb), exact download links in the manifests | The NLLB model release carries CC BY-NC 4.0, retained in `docs/licenses/LICENSE_NLLB_MODEL_CC_BY_NC_4.txt`. A separate explicit licence applying specifically to the score CSVs and generated outputs has not been independently confirmed. The project does not relicense these files under MIT. |
| `resource_groups.csv` | Derived from the pinned public-example counts | Derived metadata; source attribution and hashes retained |
| Python dependencies | Their respective upstream projects | Their respective licences; not replaced by this project's MIT licence |

The final manuscript's data statement should preserve these distinctions.
Before public reuse requiring explicit output rights, check the upstream
artifact terms or obtain clarification from the rights holder.

Core sources:

- NLLB Team (2024), *Scaling neural machine translation to 200 languages*.
  https://doi.org/10.1038/s41586-024-07335-x
- NLLB Team et al. (2022), *No Language Left Behind: Scaling Human-Centered
  Machine Translation*. https://arxiv.org/abs/2207.04672
- Popović (2017), *chrF++: words helping character n-grams*.
  https://aclanthology.org/W17-4770/
