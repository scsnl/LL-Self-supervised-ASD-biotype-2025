# Self-supervised deep learning stratification of autism into neural biotypes

Code for **"Self-supervised deep learning stratifies autism into reproducible neural biotypes with distinct clinical, circuit, and polygenic-risk signatures."**

Li L, Zhang Y, Strock A, Ryali S, Menon V
Department of Psychiatry and Behavioral Sciences, Stanford University
Correspondence: hpleili@stanford.edu; menon@stanford.edu

## What this repository does

The pipeline takes parcellated resting-state fMRI time series, learns participant-level embeddings with a self-supervised spatiotemporal network, clusters those embeddings into three neural biotypes, and then characterizes the biotypes across behavior, comorbidity, functional connectivity, gene expression, and receptor density.

1. **VICReg-stDNN.** Variance-Invariance-Covariance Regularization applied to a spatiotemporal 1D-CNN trained directly on region-by-time sequences parcellated with the Brainnetome atlas, rather than on static connectivity matrices.
2. **Site-variance suppression.** A lightweight scanner branch produces a site vector, and the biological embedding is orthogonalized against it before clustering. The projection removes site-related variance that is linear in this vector; it does not guarantee that all scanner information is gone, and the paper reports sensitivity analyses showing that biotype assignment does not track acquisition site.
3. **Cluster selection.** The number of biotypes comes from a composite score combining bootstrap and leave-one-site-out stability, silhouette separation, and independence from site, evaluated across a hyperparameter traversal.
4. **External evaluation.** Biotypes are carried into independent cohorts two ways: nearest-centroid projection using the discovery centroids, and de novo clustering with Hungarian alignment.

### Naming

The paper calls the three groups **biotypes B1, B2 and B3**. Filenames, variables and output columns here use the earlier term *subtype*, so `Subtype 1`, `Subtype 2` and `Subtype 3` are B1, B2 and B3 respectively. The code was left unrenamed so that the released version matches the version that produced the published analyses.

| Code label | Paper | Profile |
|---|---|---|
| `Subtype 1` | B1, mild-symptom | Lowest symptom burden, connectivity closest to typically developing controls |
| `Subtype 2` | B2, behaviorally dysregulated | Elevated restricted and repetitive behavior and behavioral-emotional problems, heaviest comorbidity burden, hyperconnectivity |
| `Subtype 3` | B3, social-deficit | Most pronounced social-communication difficulty, hypoconnectivity and reduced network segregation |

The three sit along a reproducible gradient ordered B3, controls, B1, B2 in embedding space and across connectivity metrics.

## System requirements

**Operating system.** Any Linux or macOS system with Python 3.9 or later. The demo below was last verified on Ubuntu 24.04 with Python 3.12.3.

**Python packages.** Minimum versions are listed in `requirements.txt`. The demo was verified with torch 2.13.0, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, scikit-learn 1.8.0, statsmodels 0.14.6, matplotlib 3.10.8, seaborn 0.13.2, umap-learn 0.5.12, networkx 3.6.1, bctpy 0.6.0, nilearn 0.14.0, nibabel 5.4.2, brainspace 0.2.1 and gseapy 1.3.1. Nothing in the pipeline depends on a pinned patch release.

**Hardware.** Training on full cohort data uses a single CUDA-capable GPU; the hyperparameter traversal is the expensive part and is the only reason a GPU is needed. Every other step, and the entire demo, runs on CPU. The demo was verified on a single CPU core with 3 GB of RAM and no GPU.

**Internet access.** Step 09 queries Enrichr through gseapy for gene-set enrichment. Without outbound access those queries fail and the step logs the failure and continues, so the PLS and correlation outputs are still written.

**Software outside this repository.** Preprocessing was carried out with SPM12 and custom MATLAB scripts, and is not part of this repository. The pipeline here begins from parcellated ROI time series stored as described under *Input format*.

## Installation

```bash
git clone https://github.com/scsnl/LL-Self-supervised-ASD-biotype-2025.git
cd LL-Self-supervised-ASD-biotype-2025
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

There is no build or compilation step. Install time is dominated by downloading the PyTorch wheel, which is the largest dependency by an order of magnitude; with a warm pip cache the remaining packages install in under a minute.

## Demo

`smoke_test_abide100.pkl` is bundled and fully synthetic: random time series, randomized clinical values, sequential `SUB_xxxx` identifiers, `SITE_xx` site codes. It contains no participant data and exists so that the pipeline can be exercised without applying for cohort access.

```bash
bash run_smoke_test.sh
```

The script runs steps 02, 03, 04, 05, 06, 07, 08 and 09 in reduced-cost mode: three training epochs, two candidate configurations, K swept over 2 to 3, and 100 permutations in step 09. Step 01, step 05b and the supplementary scripts are excluded because they need cohort files that are not bundled.

**Run time.** 137 seconds end to end on the single-core CPU machine described above.

**Expected output.** Model artifacts go to `asd_unsup_runs/smoke_test_<timestamp>/`:

```
abide_asd_emb.npy            (50, 512) ASD embeddings
abide_td_emb.npy             (50, 512) control embeddings
abide_asd_labels.npy         cluster assignment
abide_asd_sites.npy          site codes used by the scanner branch
final_centroids.npy          centroids from the discovery fit
final_all_abide_asd_model.pth
best_overall.json            selected configuration
cv_overview.csv              per-fold selection summary
fold_1_grouped/              per-fold checkpoint, centroids, selection table
```

Analysis output goes to `smoke_test_results/`:

```
step03/cmi/   k_sweep.csv, summary.json, final_centroids.npy,
              abide_asd_labels.npy, umap_abide_cmi.png
step04/       domain_composite_scores.csv, enrichment_fold.csv,
              enrichment_continuous.csv, fold_enrichment.png,
              domain_matched_columns.txt
step05/       abide_adhd_counts.csv, abide_subtype_adhd_rate.png,
              abide_adhd_only_subtype_distribution.{csv,png},
              abide_asd_pca_adhd.png
step06/       proto_ig_roi_maps.{npy,csv}, meta.json
step07/       fcs_{asd,td}.npy, gbcs_{asd,td}_gsr.npy,
              alffs_{asd,td}_gsr.npy, labels_asd.npy,
              net_stats/ (per-network statistics and effect sizes),
              figures_net/ (network FC matrices, hotspot maps, GBC violins)
step08/       gradient_axis_01..05_*.png,
              gradient_axis_all_metrics_stats.csv
step09/       step9_summary_report.txt, subtype<k>_fc_sum.npy,
              subtype<k>/gene/ (VIP scores, PLS performance, scatter plots),
              subtype<k>/receptor/ (same for receptor maps),
              subtype<k>/enrichment/significant_genes.txt
```

Two things about the demo output are worth anticipating. The input is noise, so cluster selection settles on K = 2 rather than 3, and step 09 therefore writes `subtype1` and `subtype2` only. Effect sizes and p values from the demo carry no meaning; the point is that each step executes and writes output with the expected shapes.

```bash
bash clean_smoke_test.sh    # removes smoke_test_results/ and run artifacts
```

To regenerate the synthetic dataset:

```bash
python scripts/utils/make_synthetic_smoke.py \
  --out-pklz smoke_test_abide100.pkl --n-per-group 50 --seed 42
```

`scripts/utils/create_smoke_test.py` instead draws a stratified subset from a real ABIDE pickle, replacing identifiers with sequential `SUB_xxxx` codes and dropping free-text fields, so its output carries no real identifiers either.

## Running on your own data

### Input format

Each cohort is a pickle (`.pkl` or `.pklz`) holding a `pandas.DataFrame` with one row per participant:

| Column | Content |
|---|---|
| `data` | `np.ndarray` of shape `[T, 246]`, mean time series per Brainnetome region |
| `subject_id` | Participant identifier |
| `DX_GROUP` | 1 for ASD, 2 for controls, following the ABIDE convention |
| `SEX` | Sex code |
| `AGE_AT_SCAN` | Age in years |
| `SITE_ID` | Acquisition site |
| `mean_fd` | Mean framewise displacement |
| `percentofvolsrepaired` | Percentage of volumes repaired during scrubbing |

Participants are dropped at load time when mean framewise displacement exceeds 0.5 mm or more than 10 percent of volumes required repair, and when the run is shorter than the temporal cropping window.

### Order of execution

Steps 02 and 03 must run first because everything downstream reads `abide_asd_labels.npy` from the step 03 output directory.

```bash
python scripts/02_train_vicreg.py \
  --abide-path DATA/cohort.pklz \
  --tag asd_biotype_main \
  --final-train-on-all

python scripts/03_cluster_visualize.py \
  --emb-root asd_unsup_runs/asd_biotype_main/ \
  --outdir results/step3/ \
  --kmin 2 --kmax 6
```

Steps 04 through 09 then take `--step2-outdir results/step3/` together with the cohort pickle.

### Site removal

`--site-removal orthogonal` is the default and is the strategy used throughout the study: the scanner branch yields a site vector and the biological embedding is orthogonalized against it, with an additional orthogonality penalty in the loss.

`--site-removal adversarial` attaches a site-prediction head through a gradient reversal layer instead, using `SITE_KEY_BASE` as the site label. It is provided as an alternative interface and was not used for the reported results. Enable it with `--site-removal adversarial --adv-w 1.0 --grl-lambda 1.0`.

Both share the same biological encoder and write embeddings in the same format, so steps 03 through 09 are unaffected by the choice.

## Pipeline

```
scripts/
  01_hyperparameter_search.py   hyperparameter traversal
  02_train_vicreg.py            VICReg-stDNN training, embedding export
  03_cluster_visualize.py       K selection, K-means fit, PCA and UMAP plots
  04_behavior_enrichment.py     behavioral domain fold enrichment
  05_comorbidity_abide.py       ADHD comorbidity in ABIDE
  05b_comorbidity_cmi.py        broad psychiatric comorbidity in CMI-HBN
  06_integrated_gradients.py    prototype-based Integrated Gradients attribution
  07_brain_connectivity.py      GBC decomposition and network-level FC statistics
  08_gradient_axis.py           1D gradient axis by MDS on normalized mean differences
  09_receptor_gene_pls.py       receptor and gene spatial association by PLS
  supplementary/                S1 to S7, supplementary analyses
  utils/                        shared helpers, synthetic data generation
```

### Step 01, hyperparameter search

```bash
python scripts/01_hyperparameter_search.py \
  --abide-path DATA/combined_ABIDE_information_with_fMRI.pklz \
  --cmi-path CMI-DATA/combined_asd_td_rest_run1_data.pklz \
  --tag hypersearch
```

### Step 02, training

Trains on ASD participants from the discovery cohort only, then exports embeddings for every cohort supplied.

```bash
python scripts/02_train_vicreg.py \
  --abide-path DATA/combined_ABIDE_information_with_fMRI.pklz \
  --cmi-path CMI-DATA/combined_asd_td_rest_run1_data.pklz \
  --tag asd_biotype_main \
  --final-train-on-all
```

### Step 03, cluster selection

```bash
python scripts/03_cluster_visualize.py \
  --emb-root asd_unsup_runs/<tag>/ \
  --outdir results/step3/ \
  --kmin 2 --kmax 6
```

### Step 04, behavioral enrichment

Maps phenotypic measures onto six harmonized domains, standardizes them with robust Z-scores, then computes fold enrichment of elevated domains within each biotype against the overall ASD prevalence. Fold enrichment is a ratio plotted on a log axis, so values above 1 are enrichment and values below 1 are depletion; significance uses a directional binomial test.

```bash
python scripts/04_behavior_enrichment.py \
  --abide-pklz DATA/combined_ABIDE_information_with_fMRI.pklz \
  --step2-outdir results/step3/ \
  --outdir results/step4/
```

### Step 05, comorbidity

```bash
python scripts/05_comorbidity_abide.py \
  --abide-pklz DATA/combined_ABIDE_information_with_fMRI.pklz \
  --step2-outdir results/step3/ \
  --outdir results/step5_abide/

python scripts/05b_comorbidity_cmi.py \
  --cmi-pklz CMI-DATA/combined_asd_td_rest_run1_data.pklz \
  --behavior-csv CMI-DATA/labeled_asd_td_adhd.csv \
  --step2-outdir results/step3/ \
  --emb-root asd_unsup_runs/<tag>/ \
  --outdir results/step5_cmi/
```

### Step 06, Integrated Gradients

Attribution is aggregated to networks before per-biotype standardization, in that order, so that the radar plots compare standardized network-level attribution rather than standardized ROI values.

```bash
python scripts/06_integrated_gradients.py \
  --model-path asd_unsup_runs/<tag>/final_all_abide_asd_model.pth \
  --abide-pklz DATA/combined_ABIDE_information_with_fMRI.pklz \
  --step2-outdir results/step3/ \
  --dataset ABIDE \
  --labels-file abide_asd_labels.npy \
  --network-map-csv atlas/subregion_func_network_Yeo_updated.csv \
  --devatlas24-map-csv atlas/devatlas/BN246_to_DevAtlas24_mapping.csv \
  --devatlas24-labels atlas/devatlas/DevAtlas_24RSNS_Labels.txt \
  --outdir results/step6/
```

### Step 07, connectivity

```bash
python scripts/07_brain_connectivity.py \
  --abide-pklz DATA/combined_ABIDE_information_with_fMRI.pklz \
  --step2-outdir results/step3/ \
  --net-map atlas/subregion_func_network_Yeo_updated.csv \
  --method anova \
  --outdir results/step7/
```

Network aggregation and the statistics built on it infer the number of networks from the mapping file, so `--net-map` accepts any ROI-to-network assignment; the command above passes a Yeo-7 mapping as an illustration. Network-level results in the paper use the DevAtlas24 mapping in `atlas/devatlas/BN246_to_DevAtlas24_mapping.csv`. The plotting helpers in this script label their axes for a seven-network layout, so a mapping with a different number of networks needs those labels adjusted.

### Step 08, gradient axis

The axis comes from one-dimensional MDS on pairwise normalized mean differences between groups, where the difference in group means is divided by the standard deviation of the combined sample. This is not Cohen's d: the denominator is not a pooled within-group standard deviation, which keeps the metric stable in the smaller validation cohorts where pooled estimates are noisy. Exported columns are named `d_*` for backward compatibility and hold this normalized mean difference.

```bash
python scripts/08_gradient_axis.py \
  --abide-step7-dir results/step7/ \
  --abide-step3-dir results/step3/ \
  --outdir results/step8/
```

### Step 09, receptor and gene association

```bash
python scripts/09_receptor_gene_pls.py \
  --step7_dir results/step7/ \
  --step2_dir results/step3/ \
  --receptor_csv atlas/brainnetome/receptor_data_bn246.csv \
  --gene_csv atlas/brainnetome/BN246_geneexpression.csv.gz \
  --out_dir results/step9/
```

### Supplementary scripts

| Script | Analysis |
|---|---|
| `S1_k_selection_plots.py` | Composite score across candidate K |
| `S2_k_sensitivity.py` | Sensitivity of the solution to K |
| `S3_external_robustness.py` | Replication robustness in external cohorts |
| `S4_algo_comparison.py` | Spectral and hierarchical clustering against K-means |
| `S5_baseline_comparison.py` | Raw ALFF and static FC baselines against the learned embedding |
| `S6_demographics.py` | Demographic tables |
| `S7_motion_control.py` | Network FC with framewise displacement as a covariate |

## Data

| Cohort | Role | Access |
|---|---|---|
| ABIDE I and II | Discovery | http://fcon_1000.projects.nitrc.org/indi/abide/ |
| CMI-HBN | Validation | http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/, data use agreement required |
| GENDAAR | Validation | NIMH Data Archive, authorized access required |
| Stanford | Validation | Request to the corresponding author |

This repository holds code, atlas files and the synthetic demo dataset. No participant data are included.

## Atlas and reference files

```
atlas/
  subregion_func_network_Yeo_updated.csv       ROI to Yeo-7 mapping, demo default for steps 07 and S7
  brainnetome/
    BN_Atlas_246_1mm.nii.gz                    Brainnetome parcellation, 1 mm
    BN_Atlas_246_LUT.txt                       ROI label lookup
    BN246_geneexpression.csv.gz                AHBA expression, 15,633 genes, step 09
    receptor_data_bn246.csv                    PET receptor densities, step 09
    receptor_names_pet.npy                     Receptor names
  devatlas/
    BN246_to_DevAtlas24_mapping.csv            BN246 to DevAtlas24 assignment, steps 06 to 08
    BN246_to_DevAtlas24_overlap_percent.csv    Overlap percentages behind that assignment
    DevAtlas4D_NoOverlap_24RSNS.nii.gz         DevAtlas parcellation
    DevAtlas_24RSNS_Labels.txt                 Network labels and parent systems
```

Brainnetome atlas: Fan et al. 2016, http://atlas.brainnetome.org
DevAtlas 24-network developmental parcellation: Doucet et al. 2025, Developmental Cognitive Neuroscience
Receptor density maps: Hansen et al. 2022, https://github.com/netneurolab/hansen_receptors
Gene expression: Allen Human Brain Atlas, https://human.brain-map.org

## Citation

Li L, Zhang Y, Strock A, Ryali S, Menon V. Self-supervised deep learning stratifies autism into reproducible neural biotypes with distinct clinical, circuit, and polygenic-risk signatures. Under review, 2026.

## License

MIT. See `LICENSE`.

## Acknowledgements

Preprocessing used SPM12 from the Wellcome Centre for Human Neuroimaging. The VICReg objective follows Bardes et al. 2021. The developmental parcellation is from Doucet et al. 2025 and the receptor maps from Hansen et al. 2022. Gene expression data are from the Allen Human Brain Atlas.
