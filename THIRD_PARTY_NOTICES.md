# Third-party notices

This repository uses material produced by other researchers. Our MIT licence covers only our own code.

| Material | Source | Licence |
|---|---|---|
| NLLB-200 published scores (`data/raw/nllb_200_*_metrics.csv`) and translations used in the error analysis | Meta AI, NLLB Team, https://github.com/facebookresearch/fairseq/tree/nllb | CC BY-NC 4.0 (`docs/licenses/LICENSE_NLLB_MODEL_CC_BY_NC_4.txt`) |
| `data/raw/flores200_public.py`, `data/raw/nllb_grouping_original.py` | fairseq, commit `ec0ae71a24fbc2f42252f37735f59dd6547eba38` | MIT (`docs/licenses/LICENSE_FAIRSEQ_CODE_MIT.txt`) |
| FLORES-200 sentences (in `results/error_audit/`) | https://github.com/facebookresearch/flores/tree/main/flores200 | CC BY-SA 4.0 (`docs/licenses/LICENSE_FLORES_CC_BY_SA_4.txt`) |

`data/raw/resource_groups.csv` was created by us from the training-example counts in the fairseq files above.

References:
- NLLB Team et al. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. https://arxiv.org/abs/2207.04672
- NLLB Team (2024). *Scaling neural machine translation to 200 languages*. Nature, 630, 841–846. https://doi.org/10.1038/s41586-024-07335-x
- Popović, M. (2017). *chrF++: words helping character n-grams*. https://aclanthology.org/W17-4770/
