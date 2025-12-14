# DataNova  
### Gaia DR3 Stellar Analysis, Visualization & Classification Pipeline

**Author:** Emīls Locmelis-Lunovs  
**Institution:** Rīgas Zolitūdes ģimnāzija  
**Academic Level:** 12th Grade  
**License:** MIT  
**Python Version:** 3.11+

---

## DOI

[![DOI](https://zenodo.org/badge/1047612190.svg)](https://doi.org/10.5281/zenodo.17537782)

---

## Overview

**DataNova** is an interactive, research-grade Python pipeline for **exploratory analysis, visualization, and supervised classification of Gaia DR3 stellar data**.

Built around a Streamlit interface, DataNova enables researchers and students to safely query and analyze Gaia DR3 samples, generate publication-ready Hertzsprung–Russell (HR) diagrams, study correlations between astrophysical and variability parameters, train and interpret machine-learning classifiers on stellar populations, and export fully reproducible datasets and metadata.

The project is designed with **scientific transparency, reproducibility, and physical interpretability** as first-class goals.

This repository accompanies the research work:

> *Locmelis-Lunovs, E. (2025). DataNova: A Reproducible Pipeline for Correlation-Based Stellar Classification.*

---

## Key Capabilities

### Data Acquisition
- Safe Gaia DR3 querying via `astroquery` (bounded to ≤ 50 000 sources)
- Local CSV ingestion for offline or pre-filtered datasets
- Optional enrichment via:
  - `gaiadr3.vari_summary`
  - SOS variability tables (RR Lyrae, Cepheids, LPVs)

### Preprocessing & Physics
- Absolute magnitude computation with parallax S/N control
- Optional Bailer–Jones distance support
- RUWE astrometric quality filtering
- Gaia AP-based dereddening (when available)
- Physically motivated derived quantities (bolometric correction placeholder, luminosity ratios)

### Visualization
- Publication-ready HR diagrams:
  - colored by variability class
  - colored by radius or luminosity
  - classifier-confidence visualization
- Robust axis scaling and correct astrophysical orientation
- Pearson correlation matrices with annotated coefficients
- Compact layouts suitable for academic papers and posters

### Machine Learning
- RandomForest classification with:
  - class weighting
  - optional SMOTE / ADASYN balancing
- GridSearchCV (optional Optuna support if installed)
- Feature-importance and permutation-importance tools
- Explicit separation of observational quantities and model-derived outputs

### Reproducibility & Export
- Deterministic random seeds
- Metadata-rich CSV exports (original and filtered datasets)
- Feature-score and correlation artifacts
- Environment snapshot (Python and package versions)
- Optional prediction and confidence export

---

## Design Philosophy

DataNova intentionally:
- avoids silent refetching or retraining,
- keeps model-dependent quantities separate from physical filters,
- prioritizes interpretability over black-box optimization,
- respects Gaia archive service constraints.

The default configuration is suitable for **student research, exploratory astrophysics, and small-to-medium-scale ML studies** without HPC resources.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<yourusername>/DataNova.git
cd DataNova
```

Install dependencies:

```bash
pip install -r install_requirements.txt
```

### Optional Dependencies

- `astroquery` — required for live Gaia fetching
- `imblearn` — enables SMOTE / ADASYN
- `optuna` — enables Bayesian hyperparameter optimization

---

## Usage

Run the Streamlit app:

```bash
streamlit run C:\Users\{location of datanova.py}
```

Typical workflow:

1. Fetch a Gaia DR3 sample **or** upload a local CSV
2. Inspect correlations and HR diagrams
3. (Optional) Train a classifier
4. Filter stars by absolute magnitude and class selection
5. Export reproducible datasets and metadata

### Note on Streamlit Reruns

Streamlit reruns parts of the script when UI controls change (e.g., filtering values).
This is expected behavior and does not refetch Gaia data or retrain models unless you press the corresponding buttons.

---

## Limitations

- Gaia queries are capped at 50 000 rows for stability and responsiveness.
- Not intended for full-catalog processing without batching.
- Bolometric corrections are heuristic placeholders and should be replaced for precision work.
- Classifier predictions are model-dependent and must be interpreted cautiously.

---

## License

# MIT License

Copyright (c) 2025 Emīls Locmelis-Lunovs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

---

## Citation

If you use DataNova in academic work, please cite:

```
Locmelis-Lunovs, E. (2025). DataNova: A Reproducible Pipeline for Correlation-Based Stellar Classification. Zenodo. https://doi.org/10.5281/zenodo.17537782
```
