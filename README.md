# Computational Methods for Analysing T Cell Spatial Organisation in Ovarian Cancer

## A KDTree-Based Spatial Transcriptomics Pipeline with Machine Learning Validation

**Author:** Ojo Gideon Mayowa  
**Supervisors:** Professor Annalisa Occhipinti | Professor Claudio Angione  
**Institution:** Teesside University, School of Computing, Engineering & Digital Technologies  
**Degree:** MSc Applied Data Science, 2026

---

## Overview

This repository contains the full analysis pipeline, machine learning models, and interactive Streamlit dashboard for the study of T cell spatial organisation in ovarian cancer.

The study investigates how the physical positioning of CD8+ T cells within ovarian cancer tissue relates to their functional exhaustion state, using spatial transcriptomics data from Yeh et al. (2024) comprising **491,792 cells across 58 patients** measured using the MERSCOPE platform.

---

## Key Findings

- T cells physically closer to cancer cells are significantly more exhausted (Spearman r = −0.1795, p = 4.51 × 10⁻⁶²)
- A quantitative exhaustion threshold exists at 5–9 cancer neighbours overall, dropping to 1–2 in untreated and Stage IV patients — a novel finding not previously reported
- Treatment significantly repositions T cells from the tumour (median 58.6 vs 174.7 µm, p = 8.98 × 10⁻¹⁶²)
- THBS1 acts as a genuine infiltration barrier while CD47 is reactively upregulated by tumour proximity, providing spatial evidence supporting the TSP-1/CD47 exhaustion pathway
- Random Forest trained on spatial features achieved AUC = 0.5992 (5-fold CV) and AUC = 0.5852 (LOPO)
- Group-specific LOPO revealed stronger spatial exhaustion signal in untreated patients (AUC = 0.5939) vs treated patients (AUC = 0.5766)

---

## Repository Structure

```
├── spatial_analysis_reordered.ipynb   # Main analysis notebook
├── dashboard.py                        # Interactive Streamlit dashboard
├── dashboard_model.pkl                 # Trained Random Forest model
├── dashboard_data.csv                  # Processed cell-level data
├── dashboard_cells.csv                 # Cell metadata
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Dataset

The dataset is publicly available at:  
🔗 [https://zenodo.org/records/12613839](https://zenodo.org/records/12613839)

Clinical metadata is available as supplementary material to:  
> Yeh, C. et al. (2024). Spatial organisation of the tumour microenvironment in ovarian cancer. *Nature Immunology*, 25. [https://doi.org/10.1038/s41590-024-01943-5](https://doi.org/10.1038/s41590-024-01943-5)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mayor01/tcell-spatial-ovarian-cancer.git
cd tcell-spatial-ovarian-cancer

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser. Load your `.h5ad` file using the file path input on the **Load Data** page.

**Dashboard pages:**
1. **Home** — Project overview and key findings
2. **Load Data** — Load any compatible `.h5ad` file
3. **Run Pipeline** — Compute KDTree spatial features
4. **Predict Cell State** — Predict T cell exhaustion state
5. **Spatial Plot** — Visualise cells on tissue
6. **Feature Importance** — Random Forest feature importance
7. **Patient Explorer** — Patient-level analysis
8. **Exhaustion Threshold** — Threshold analysis by clinical group

---

## Running the Analysis Notebook

```bash
jupyter notebook spatial_analysis_reordered.ipynb
```

Update the file path at the top of the notebook to point to your local copy of `ST_Discovery.h5ad`.

---

## Methods Summary

| Component | Details |
|---|---|
| Spatial pipeline | KDTree-based (SciPy) |
| Spatial features | Nearest tumour distance, radius neighbour count (r=150µm), KNN tumour fraction (k=20) |
| Exhaustion score | Z-scored composite of PDCD1, LAG3, HAVCR2, CTLA4 |
| ML model | Random Forest (100 estimators, scikit-learn) |
| Validation | 5-fold CV, LOPO, Group-specific LOPO |
| CNN | ResNet50 (PyTorch), 256px patches, PCA to 50 components |
| Statistics | Spearman correlation, Mann-Whitney U (SciPy) |

---

## Software

All analysis was implemented in **Python 3.11** using:

| Package | Version |
|---|---|
| Scanpy | 1.9 |
| scikit-learn | 1.3 |
| SciPy | 1.11 |
| NumPy | 1.24 |
| Pandas | 2.0 |
| PyTorch | 2.0 |
| Streamlit | 1.28 |

---

## Citation

If you use this code or pipeline in your research, please cite:

> Ojo, G.M., Occhipinti, A. & Angione, C. (2026). Computational Methods for Analysing T Cell Spatial Organisation in Ovarian Cancer: A KDTree-Based Spatial Transcriptomics Pipeline with Machine Learning Validation. *Manuscript in preparation.*

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Contact

**Ojo Gideon Mayowa**  
MSc Applied Data Science, Teesside University  
GitHub: [@mayor01](https://github.com/mayor01)

**Professor Annalisa Occhipinti** (Corresponding Author)  
Email: A.Occhipinti@tees.ac.uk  
Webpage: [https://research.tees.ac.uk/en/persons/annalisa-occhipinti](https://research.tees.ac.uk/en/persons/annalisa-occhipinti)

**Professor Claudio Angione** (Co-author)  
Email: C.Angione@tees.ac.uk
Webpage: https://sites.google.com/view/angionelab/
