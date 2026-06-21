#!/bin/bash
# Exp 4 Step 1 — Garden baseline: B_total (Arm A) + σ (Arm B)
#
# Dependency: Arm A must finish first. It writes BTOTAL_GARDEN.
# Arm B reads BTOTAL_GARDEN for its --B_total.
#
# Output: /root/autodl-tmp/exp4/garden_baseline/
#   arm_a/seed_0..4  — Octree-GS native, measure B_total
#   arm_b/seed_0..4  — A-only controller, measure σ

set -e

SRC=/root/autodl-tmp/m360/garden
DST=/root/autodl-tmp/exp4/garden_baseline
BTOTAL_FILE="$DST/BTOTAL_GARDEN"
ITERS=30000
UPDATE_UNTIL=25000

echo "=== Exp 4 Step 1: Garden baseline ==="
echo "Source: $SRC"
echo "Output: $DST"

mkdir -p "$DST"

# ── Arm A: Octree-GS native (no controller) → B_total ──
echo ""
echo "--- Arm A: Octree-GS native × 5 seeds ---"
for seed in 0 1 2 3 4; do
    echo "  seed=$seed"
    python train.py \
        -s "$SRC" --ds 8 \
        -m "$DST/arm_a/seed_$seed" \
        --fork 2 --base_layer 10 --visible_threshold 0.0 \
        --dist2level round --update_ratio 0.2 \
        --iterations $ITERS --update_until $UPDATE_UNTIL \
        --test_iterations 7000 15000 $UPDATE_UNTIL $ITERS \
        --save_iterations $UPDATE_UNTIL $ITERS \
        --seed $seed --no_controller &
    sleep 30s
done
wait

# Extract B_total from TensorBoard total_points @ update_until
# Uses the last seed as representative (5-seed mean computed later)
B_TOTAL_SEED0=$(python -c "
import os, glob, struct
# Read TB event file for arm_a/seed_0, find total_points @ $UPDATE_UNTIL
print('PLEASE_READ_MANUALLY')
" 2>/dev/null || echo "MANUAL")

echo ""
echo "=== Arm A complete ==="
echo "Extract B_total from TensorBoard scalar total_points @ $UPDATE_UNTIL."
echo "For each seed: tensorboard --logdir $DST/arm_a/seed_<N>/events*"
echo "Then compute 5-seed mean and write to $BTOTAL_FILE."
echo ""
echo "After BTOTAL_GARDEN is written, re-run with ARM_B=1:"
echo "  ARM_B=1 bash scripts/exp4_garden_baseline.sh"

# ── Arm B: A-only controller → σ (only if BTOTAL_GARDEN exists) ──
if [ "${ARM_B:-0}" != "1" ]; then
    echo ""
    echo "Skipping Arm B. Set ARM_B=1 and ensure BTOTAL_GARDEN exists."
    exit 0
fi

B_TOTAL=$(cat "$BTOTAL_FILE")
echo ""
echo "--- Arm B: A-only controller × 5 seeds (B_total=$B_TOTAL) ---"
for seed in 0 1 2 3 4; do
    echo "  seed=$seed"
    python train.py \
        -s "$SRC" --ds 8 \
        -m "$DST/arm_b/seed_$seed" \
        --fork 2 --base_layer 10 --visible_threshold 0.0 \
        --dist2level round --update_ratio 0.2 \
        --iterations $ITERS --update_until $UPDATE_UNTIL \
        --test_iterations 7000 15000 $UPDATE_UNTIL $ITERS \
        --save_iterations $UPDATE_UNTIL $ITERS \
        --seed $seed --B_total $B_TOTAL &
    sleep 30s
done
wait

echo ""
echo "=== Arm B complete ==="
echo "Compute σ_PSNR from results.json across 5 seeds."
echo "Per-checkpoint Δ from TB total_points."
echo "Output dir: $DST/arm_b/"
