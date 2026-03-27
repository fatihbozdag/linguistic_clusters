# ConstructionMiner: Linguistic Clusters Framework

Pipeline for identifying stable distributional patterns in learner English passive constructions using multi-metric filtering and validation.

## Overview

The Linguistic Clusters (LC) Framework implements a three-layer architecture for detecting formulaic patterns in learner corpus data:

- **Layer 1 (Instance):** Transitional probability (ATP), directional contingency (ΔP), boundary entropy (Hr)
- **Layer 2 (Schema):** Dual-lane acceptance via NPMI (fixed collocations) or Hslot (productive templates)
- **Layer 3 (Validation):** Permutation testing, sensitivity analysis, ablation analysis

## Key Results

Applied to 9,529 ICLE texts (57,090 valid passive constructions):
- 8 validated schemas from 183 formulaic instances across 85 TAM×complement patterns
- Permutation testing: p = .023 (TAM-shuffling confirms non-random structure)
- 75% of schemas globally stable across threshold variations
- 94.6% false-positive reduction from instance-level filtering

## Project Structure

```
CMCU/
├── src/                    # Core pipeline modules (7 files)
│   ├── lc_metrics.py       # LC Framework metrics (ATP, ΔP, Hr, NPMI, Hslot, G²)
│   ├── passive_extractor.py
│   ├── passive_filter.py
│   ├── prep_normalizer.py
│   ├── tam_comp_classifier.py
│   ├── validation_tests.py
│   └── verb_validator.py
├── scripts/                # Analysis and validation scripts
├── tests/                  # Unit tests
├── results/
│   ├── icle/               # ICLE results (8 schemas, FRESH run)
│   └── locness/            # LOCNESS native comparison
├── data/                   # Corpus data (not included; see data/README.md)
├── run_full_corpus_analysis.py
├── run_complete_validation.py
└── requirements.txt
```

## Requirements

```
pip install -r requirements.txt
```

Requires Python 3.9+ and spaCy with `en_core_web_trf` model.

## Citation

Bozdağ, F. U. (under review). Modeling distributional stability in learner English passives: The Linguistic Clusters Framework. *Corpus Linguistics and Linguistic Theory*.

## License

MIT
