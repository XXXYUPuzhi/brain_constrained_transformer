#!/bin/bash
# run_experiments.sh — Launch all resource-limitation experiments
#
# Usage: bash run_experiments.sh
#
# Structure:
#   Phase 1: Baseline reproduction
#   Phase 2: L1 lambda sweep
#   Phase 3: Top-K activation sparsity sweep
#   Phase 4: Post-training analysis
#
# Author: Puzhi YU
# Date:   January 2026

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/train_resource_limited.py"
ANALYZE="$SCRIPT_DIR/analyze_resource_limited.py"
DATA_DIR="$SCRIPT_DIR/junction_predict_resortind"
OUT_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "========================================"
echo "  Resource Limitation Experiments"
echo "  $(date)"
echo "========================================"

# Phase 1: Baseline (2 layers, 12d1h — the reference model)
echo ""
echo "── Phase 1: Baseline reproduction ──"
$PYTHON "$SCRIPT" \
    --reg none \
    --layers 2 --dim 12 --heads 1 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    2>&1 | tee "$LOG_DIR/baseline_2L_12d1h.log"

# Phase 2: L1 sweep
echo ""
echo "── Phase 2: L1 regularization sweep ──"
for LAMBDA in 1e-5 5e-5 1e-4 5e-4 1e-3 1e-2; do
    echo ""
    echo "  lambda = $LAMBDA"
    $PYTHON "$SCRIPT" \
        --reg l1 --lambda_l1 $LAMBDA \
        --layers 2 --dim 12 --heads 1 \
        --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
        --snapshot_every 100 \
        2>&1 | tee "$LOG_DIR/l1_lambda${LAMBDA}_2L_12d1h.log"
done

# Phase 3: Top-K sweep (MLP hidden dim = 4 * 12 = 48 neurons)
echo ""
echo "── Phase 3: Top-K activation sparsity sweep ──"
for K in 4 8 16 24 32; do
    echo ""
    echo "  k = $K"
    $PYTHON "$SCRIPT" \
        --reg topk --topk_k $K \
        --layers 2 --dim 12 --heads 1 \
        --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
        2>&1 | tee "$LOG_DIR/topk_k${K}_2L_12d1h.log"
done

echo ""
echo "========================================"
echo "  All training done! $(date)"
echo "========================================"

# Phase 4: Post-training analysis
echo ""
echo "── Phase 4: Post-training analysis ──"
$PYTHON "$ANALYZE" \
    --model_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR" \
    --out_dir "$SCRIPT_DIR/analysis_results" \
    --no_umap \
    2>&1 | tee "$LOG_DIR/analysis.log"

echo ""
echo "All done! $(date)"
