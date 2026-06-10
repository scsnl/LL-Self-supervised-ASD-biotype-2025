# Self-Supervised ASD Subtype Discovery

**Self-supervised learning reveals robust, reproducible autism subtypes with distinct clinical, imaging and genetic signatures**

Li L, Zhang Y, Strock A, Ryali S, Menon V
Stanford University, Department of Psychiatry & Behavioral Sciences
*Correspondence: hpleili@stanford.edu; menon@stanford.edu*

---

## Overview

This repository contains the analysis code for identifying and characterizing reproducible neurobiological subtypes of Autism Spectrum Disorder (ASD) using a scanner-invariant, self-supervised deep learning framework.

**Key innovations:**
1. **VICReg-stDNN** — Variance–Invariance–Covariance Regularization applied to a spatiotemporal Deep Neural Network trained on raw region-by-time fMRI sequences (Brainnetome Atlas parcellation).
2. **Orthogonal site-signal projection** — Scanner/site signals are explicitly isolated and removed, ensuring subtype assignments are biologically driven rather than acquisition artifacts.
3. **Data-driven cluster selection** — The number of subtypes is chosen via a composite stability–separation–site-independence score across a hyperparameter sweep.
4. **Cross-cohort reproducibility** — Subtypes are validated across multiple independent datasets (ABIDE I&II, CMI-HBN, GENDAAR, Stanford) spanning many acquisition sites.

---

## Datasets

| Cohort | Role |
|--------|------|
| ABIDE I & II | Discovery |
| CMI-HBN | Validation |
| GENDAAR | Validation |
| Stanford | Validation |

Data must be requested from the respective repositories (this repository contains **code only**, plus a fully synthetic smoke-test dataset; no real participant data is included):
- ABIDE: [http://fcon_1000.projects.nitrc.org/indi/abide/](http://fcon_1000.projects.nitrc.org/indi/abide/)
- CMI-HBN: [http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)
- GENDAAR: via the National Institute of Mental Health Data Archive (NDA)
- Stanford: via application to the authors

---

## Installation

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
```

GPU (CUDA) is recommended for the model-training steps (01–02). CPU-only is feasible but slower.

---

## Quick Start (Smoke Test)

A single script runs the **entire pipeline** end-to-end. A fully synthetic dataset (`smoke_test_abide100.pkl`) is included so the smoke test runs out of the box with no real data:

```bash
# Run on the bundled synthetic dataset:
bash run_smoke_test.sh
```

The script runs Steps 02–09 sequentially. All results are written to `smoke_test_results/`.

Once you have verified the outputs, remove them to keep a clean repository:

```bash
bash clean_smoke_test.sh
```

### Regenerating the smoke-test dataset

The bundled `smoke_test_abide100.pkl` is **fully synthetic** (random time series, randomized clinical values, fake `SUB_xxxx` identifiers). It contains no real participant data and exists only to exercise the pipeline. To (re)generate it:

```bash
python scripts/utils/make_synthetic_smoke.py \
  --out-pklz    smoke_test_abide100.pkl \
  --n-per-group 50 \
  --seed        42
```

If you have access to the real ABIDE pickle and want a smoke set drawn from it, use `create_smoke_test.py` instead. That script samples a stratified subset, **anonymizes identifiers** (replacing them with sequential `SUB_xxxx` IDs) and drops free-text fields, so the output carries no real subject identifiers:

```bash
python scripts/utils/create_smoke_test.py \
  --abide-pklz  path/to/combined_ABIDE_information_with_fMRI.pklz \
  --out-pklz    smoke_test_abide100.pkl \
  --n-per-group 50
```

---

## Analysis Pipeline

```
scripts/
├── 01_hyperparameter_search.py    # Hyperparameter optimization
├── 02_train_vicreg.py             # VICReg-stDNN training & embedding export
├── 03_cluster_visualize.py        # K-means cluster selection + UMAP/PCA
├── 04_behavior_enrichment.py      # Behavioral domain enrichment
├── 05_comorbidity_abide.py        # ADHD comorbidity – ABIDE
├── 05b_comorbidity_cmi.py         # Broad psychiatric comorbidity – CMI-HBN
├── 06_integrated_gradients.py     # Explainable AI: IG ROI importance maps
├── 07_brain_connectivity.py       # GBC + network-level FC statistics
├── 08_gradient_axis.py            # Neurobiological gradient axis (MDS + Cohen's d)
├── 09_receptor_gene_pls.py        # Receptor/gene spatial correlation (PLS)
├── supplementary/                 # Supplementary figure scripts (S1–S7)
└── utils/
    ├── make_synthetic_smoke.py    # Generate the synthetic smoke-test dataset
    ├── create_smoke_test.py       # Create an anonymized smoke set from real ABIDE
    ├── step7_abide_network_metrics.py
    ├── step7_brain_metrics_abide.py
    ├── step7_abide_alff_violin_only.py
    ├── step7_gbc_load_utils.py
    ├── step7_network_fc_stats_three_subtypes_lib.py
    ├── step7_outlier_utils.py
    └── step7_violin_three_subtypes_lib.py
```

### Step-by-step

#### Step 01 – Hyperparameter Search (optional)
Runs a grid/random search over model configurations to identify the architecture and training hyperparameters.

```bash
python scripts/01_hyperparameter_search.py \
  --abide-path DATA/combined_ABIDE_information_with_fMRI.pklz \
  --cmi-path   CMI-DATA/combined_asd_td_rest_run1_data.pklz \
  --tag        hypersearch
```

#### Step 02 – VICReg-stDNN Training
Trains the self-supervised model on ABIDE-ASD only. Exports embeddings for all cohorts.

```bash
python scripts/02_train_vicreg.py \
  --abide-path DATA/combined_ABIDE_information_with_fMRI.pklz \
  --cmi-path   CMI-DATA/combined_asd_td_rest_run1_data.pklz \
  --tag        asd_subtype_main \
  --final-train-on-all
```

**Site-removal interface (`--site-removal`).** Two interchangeable strategies for removing scanner/site variance are available:
- `orthogonal` (default) — a scanner-proxy branch yields a site vector and the biological embedding is orthogonalized against it (orthogonal projection + orthogonal penalty). This is the strategy used throughout the study.
- `adversarial` — a site-prediction head attached to the embedding through a Gradient Reversal Layer (DANN-style); the encoder is trained to make the embedding non-predictive of site (site labels use `SITE_KEY_BASE`). Enable with `--site-removal adversarial --adv-w 1.0 --grl-lambda 1.0`.

Both interfaces share the same biological encoder and export embeddings in the same format, so all downstream steps (03–09) are unchanged regardless of choice.

**Key outputs** (`asd_unsup_runs/<tag>/`):
- `final_all_abide_asd_model.pth` — trained model weights
- `best_overall.json` — selected configuration
- `abide_asd_emb.npy`, `abide_td_emb.npy` — ABIDE embeddings
- `cmi_asd_emb.npy`, `cmi_td_emb.npy` — CMI embeddings (when CMI path provided)

#### Step 03 – Cluster Selection & Visualization
Re-evaluates the number of clusters on ABIDE-ASD embeddings and generates UMAP/PCA plots.

```bash
python scripts/03_cluster_visualize.py \
  --emb-root asd_unsup_runs/<tag>/ \
  --outdir   results/step3/ \
  --kmin 2 --kmax 6
```

**Key outputs:**
- `final_centroids.npy` — cluster centroids fitted on ABIDE-ASD
- `abide_asd_labels.npy` — subtype labels for ABIDE-ASD
- `umap_abide_cmi.png` — 2-D visualization

#### Step 04 – Behavioral Domain Enrichment
Maps phenotypic measures to harmonized domains; computes signed fold-change enrichment per subtype.

```bash
python scripts/04_behavior_enrichment.py \
  --abide-pklz DATA/combined_ABIDE_information_with_fMRI.pklz \
  --step2-outdir results/step3/ \
  --outdir     results/step4/
```

#### Step 05 – Comorbidity Analysis
```bash
# ABIDE: ADHD comorbidity
python scripts/05_comorbidity_abide.py \
  --abide-pklz   DATA/combined_ABIDE_information_with_fMRI.pklz \
  --step2-outdir results/step3/ \
  --outdir       results/step5_abide/

# CMI-HBN: broad psychiatric comorbidity
python scripts/05b_comorbidity_cmi.py \
  --cmi-pklz     CMI-DATA/combined_asd_td_rest_run1_data.pklz \
  --behavior-csv CMI-DATA/labeled_asd_td_adhd.csv \
  --step2-outdir results/step3/ \
  --emb-root     asd_unsup_runs/<tag>/ \
  --outdir       results/step5_cmi/
```

#### Step 06 – Integrated Gradients (Explainable AI)
Computes prototype-based IG attribution maps per subtype using the trained stDNN.

```bash
python scripts/06_integrated_gradients.py \
  --model-path   asd_unsup_runs/<tag>/final_all_abide_asd_model.pth \
  --abide-pklz   DATA/combined_ABIDE_information_with_fMRI.pklz \
  --step2-outdir results/step3/ \
  --dataset      ABIDE \
  --labels-file  abide_asd_labels.npy \
  --network-map-csv    atlas/subregion_func_network_Yeo_updated.csv \
  --devatlas24-map-csv atlas/devatlas/BN246_to_DevAtlas24_mapping.csv \
  --devatlas24-labels  atlas/devatlas/DevAtlas_24RSNS_Labels.txt \
  --outdir       results/step6/
```

#### Step 07 – Brain Connectivity (GBC + Network FC)
Computes Global Brain Connectivity (GBC) and network-level FC statistics.

```bash
python scripts/07_brain_connectivity.py \
  --abide-pklz   DATA/combined_ABIDE_information_with_fMRI.pklz \
  --step2-outdir results/step3/ \
  --net-map      atlas/subregion_func_network_Yeo_updated.csv \
  --method       anova \
  --outdir       results/step7/
```

#### Step 08 – Neurobiological Gradient Axis
Constructs the 1-D gradient axis across metrics via MDS on Cohen's d matrices.

```bash
python scripts/08_gradient_axis.py \
  --abide-step7-dir results/step7/ \
  --abide-step3-dir results/step3/ \
  --outdir          results/step8/
```

#### Step 09 – Receptor / Gene PLS
Correlates subtype-specific connectivity profiles with neurotransmitter receptor maps and gene expression.

```bash
python scripts/09_receptor_gene_pls.py \
  --step7_dir    results/step7/ \
  --step2_dir    results/step3/ \
  --receptor_csv atlas/brainnetome/receptor_data_bn246.csv \
  --gene_csv     atlas/brainnetome/BN246_geneexpression.csv.gz \
  --out_dir      results/step9/
```

---

## Supplementary Scripts

| Script | Description |
|--------|-------------|
| `S1_k_selection_plots.py` | Cluster-number selection composite score visualization |
| `S2_k_sensitivity.py` | Cluster-number sensitivity analysis |
| `S3_external_robustness.py` | External cohort replication robustness |
| `S4_algo_comparison.py` | Clustering algorithm comparison (spectral/hierarchical vs K-means) |
| `S5_baseline_comparison.py` | Comparison to conventional feature baselines |
| `S6_demographics.py` | Demographic table generation |
| `S7_motion_control.py` | Network FC with framewise-displacement covariate control |

---

## Atlas & Reference Files

```
atlas/
├── subregion_func_network_Yeo_updated.csv        # ROI → Yeo-7 functional network mapping (Steps 07, S7)
├── brainnetome/
│   ├── BN_Atlas_246_1mm.nii.gz                      # Brainnetome Atlas NIfTI (1 mm)
│   ├── BN_Atlas_246_LUT.txt                      # Look-up table for ROI labels
│   ├── BN246_geneexpression.csv.gz                  # AHBA gene expression (Step 09)
│   ├── receptor_data_bn246.csv                   # PET neurotransmitter receptor densities (Step 09)
│   └── receptor_names_pet.npy                    # Receptor name list
└── devatlas/
    ├── BN246_to_DevAtlas24_mapping.csv           # BN → DevAtlas RSN overlap mapping (Steps 06, 08)
    ├── BN246_to_DevAtlas24_overlap_percent.csv   # Overlap percentage table
    ├── DevAtlas4D_NoOverlap_24RSNS.nii.gz           # DevAtlas parcellation
    └── DevAtlas_24RSNS_Labels.txt                # RSN labels and parent-system groupings
```

**Brainnetome Atlas** (Fan et al. 2016) — [http://atlas.brainnetome.org](http://atlas.brainnetome.org)

**DevAtlas** (developmental network parcellation) — Doucet et al. 2025, *Dev. Cogn. Neurosci.*

**Receptor density maps** — Hansen et al. 2022 [https://github.com/netneurolab/hansen_receptors](https://github.com/netneurolab/hansen_receptors)

**Gene expression** — Allen Human Brain Atlas [https://human.brain-map.org](https://human.brain-map.org)

---

## Data Format

All cohort data are expected as Python pickle files (`.pkl` / `.pklz`) containing a `pandas.DataFrame` with at minimum:

| Column | Description |
|--------|-------------|
| `data` | `np.ndarray` of shape `[T, ROIs]` — mean ROI time series |
| `DX_GROUP` | `1` = ASD, `2` = TDC (ABIDE convention) |
| `SEX` | Sex code |
| `AGE_AT_SCAN` | Age in years (float) |
| `SITE_ID` | Acquisition site identifier |
| `mean_fd` | Mean framewise displacement (QC) |
| `percentofvolsrepaired` | Scrubbed volume fraction (QC) |

QC thresholds (mean FD and repaired-volume fraction) are applied at load time; see the scripts for the exact values used.

---

## Three ASD Subtypes

| Subtype | Label | Profile |
|---------|-------|---------|
| **Subtype 1** | Mild-symptom | Reduced SCL and RRB burden; near-typical connectivity |
| **Subtype 2** | Behaviorally dysregulated | Elevated RRB and BEH-EMO; highest GBC; high ADHD comorbidity |
| **Subtype 3** | Social-deficit | Elevated SCL; lowest GBC; strongest network segregation alterations |

The three subtypes align along a reproducible neurobiological gradient axis (S3 → TDC → S1 → S2) in the stDNN embedding space and across brain connectivity metrics.

---

## Citation

If you use this code, please cite:

> Li L, Zhang Y, Strock A, Ryali S, Menon V. Self-supervised learning reveals robust, reproducible autism subtypes with distinct clinical, imaging and genetic signatures. *In preparation* (2026).

---

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

Preprocessing used SPM12 (Wellcome Centre for Human Neuroimaging). The VICReg objective follows Bardes et al. (2021). DevAtlas parcellation from Doucet et al. (2025). Receptor density maps from Hansen et al. (2022).
