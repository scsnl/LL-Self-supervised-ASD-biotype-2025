#!/usr/bin/env bash
# run_smoke_test.sh
#
# End-to-end smoke test for the ASD subtype pipeline.
# Runs the full analysis (Steps 02-09) on a 100-participant ABIDE subset
# (50 ASD / 50 TDC) to verify the pipeline works on your system.
#
# Prerequisites:
#   pip install -r requirements.txt
#
# Usage (first time):
#   bash run_smoke_test.sh <path/to/combined_ABIDE.pklz>
#
# Usage (if smoke_test_abide100.pkl already exists):
#   bash run_smoke_test.sh
#
# After verifying the results:
#   bash clean_smoke_test.sh     # removes all generated test output

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SMOKE_PKL="smoke_test_abide100.pkl"
OUT="smoke_test_results"

# ── [0] Generate smoke-test dataset ─────────────────────────────────────────
if [[ -n "$1" ]]; then
    echo "[0/9] Generating smoke-test dataset from: $1"
    python scripts/utils/create_smoke_test.py \
        --abide-pklz "$1" \
        --out-pklz   "$SMOKE_PKL" \
        --n-per-group 50 --seed 42
fi

if [[ ! -f "$SMOKE_PKL" ]]; then
    echo "ERROR: $SMOKE_PKL not found."
    echo "Usage: bash run_smoke_test.sh <path/to/combined_ABIDE.pklz>"
    exit 1
fi

echo ""
echo "=================================================="
echo "  ASD Subtype Pipeline  -  Smoke Test"
echo "  Dataset : $SMOKE_PKL (100 participants)"
echo "  Output  : $OUT/"
echo "=================================================="

# ── [2] VICReg-stDNN training (5 epochs) ────────────────────────────────────
echo ""
echo "[2/9] Training VICReg-stDNN (3 epochs, smoke mode)..."
python scripts/02_train_vicreg.py \
    --abide-path smoke_test_abide100.pkl \
    --tag        smoke_test \
    --epochs     3 \
    --kmin 2 --kmax 3 \
    --num-configs 2 \
    --final-train-on-all

EMB_ROOT=$(ls -td asd_unsup_runs/smoke_test_* 2>/dev/null | head -1)
echo "  Embedding root: $EMB_ROOT"

# ── [3] Cluster selection + UMAP ────────────────────────────────────────────
echo ""
echo "[3/9] Cluster selection + UMAP visualization..."
python scripts/03_cluster_visualize.py \
    --emb-root "$EMB_ROOT" \
    --outdir   "$OUT/step03" \
    --kmin 2 --kmax 3

STEP3_DIR="$OUT/step03/cmi"

# ── [4] Behavioral domain enrichment ────────────────────────────────────────
echo ""
echo "[4/9] Behavioral domain enrichment..."
python scripts/04_behavior_enrichment.py \
    --abide-pklz "$SMOKE_PKL" \
    --step2-outdir "$STEP3_DIR" \
    --outdir "$OUT/step04"

# ── [5] ADHD comorbidity ─────────────────────────────────────────────────────
echo ""
echo "[5/9] ADHD comorbidity analysis..."
python scripts/05_comorbidity_abide.py \
    --abide-pklz "$SMOKE_PKL" \
    --step2-outdir "$STEP3_DIR" \
    --outdir "$OUT/step05"

# ── [6] Integrated Gradients ─────────────────────────────────────────────────
echo ""
echo "[6/9] Integrated Gradients attribution maps..."
MODEL_PTH=$(find "$EMB_ROOT" -name "*.pth" | head -1)
python scripts/06_integrated_gradients.py \
    --model-path  "$MODEL_PTH" \
    --abide-pklz  "$SMOKE_PKL" \
    --step2-outdir "$STEP3_DIR" \
    --dataset ABIDE \
    --labels-file "abide_asd_labels.npy" \
    --network-map-csv     "atlas/subregion_func_network_Yeo_updated.csv" \
    --devatlas24-map-csv  "atlas/devatlas/BN246_to_DevAtlas24_mapping.csv" \
    --devatlas24-labels   "atlas/devatlas/DevAtlas_24RSNS_Labels.txt" \
    --outdir "$OUT/step06"

# ── [7] Brain connectivity (FC + GBC + ALFF) ────────────────────────────────
echo ""
echo "[7/9] Brain connectivity (FC + GBC + ALFF)..."
python scripts/07_brain_connectivity.py \
    --abide-pklz   "$SMOKE_PKL" \
    --step2-outdir "$STEP3_DIR" \
    --net-map      "atlas/subregion_func_network_Yeo_updated.csv" \
    --method kruskal \
    --outdir "$OUT/step07"

# ── [8] Gradient axis ────────────────────────────────────────────────────────
echo ""
echo "[8/9] Neurobiological gradient axis..."
python scripts/08_gradient_axis.py \
    --abide-step7-dir "$OUT/step07" \
    --abide-step3-dir "$STEP3_DIR" \
    --outdir "$OUT/step08"

# ── [9] Receptor / gene PLS (optional) ──────────────────────────────────────
echo ""
echo "[9/9] Receptor/gene PLS..."
if [[ -f "atlas/brainnetome/receptor_data_bn246.csv" ]]; then
    python scripts/09_receptor_gene_pls.py \
        --step7_dir  "$OUT/step07" \
        --step2_dir  "$STEP3_DIR" \
        --receptor_csv atlas/brainnetome/receptor_data_bn246.csv \
        --gene_csv     atlas/brainnetome/BN246_geneexpression.csv.gz \
        --out_dir "$OUT/step09" \
        --n_perm 100
else
    echo "  (skipped – atlas/brainnetome/receptor_data_bn246.csv not found)"
fi

echo ""
echo "=================================================="
echo "  Smoke test COMPLETE."
echo "  Check results in: $OUT/"
echo ""
echo "  To clean up test output, run:"
echo "    bash clean_smoke_test.sh"
echo "=================================================="
ls "$OUT/"
