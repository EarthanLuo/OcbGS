#!/bin/bash
# Exp 4 Step 2 — Garden no-harm control: A+B vs A-only
#
# Prerequisite: BTOTAL_GARDEN from Step 1 Arm A.
# Verifies A+B does not degrade quality on near-uniform scene.
#
# Output: /root/autodl-tmp/exp4/garden_no_harm/
#   a_only/seed_0..4   — A-only controller (continue from Step 1 Arm B,
#                        or re-run for fresh comparison)
#   a_plus_b/seed_0..4 — A+B (λ=1, M=10, K=4)
#
# If Step 1 Arm B already ran, the A-only arm here is a re-measurement
# for matched-noise comparison against A+B. Use same seed set.

set -e

SRC=/root/autodl-tmp/m360/garden
DST=/root/autodl-tmp/exp4/garden_no_harm
BTOTAL_FILE=/root/autodl-tmp/exp4/garden_baseline/BTOTAL_GARDEN
ITERS=30000
UPDATE_UNTIL=25000

if [ ! -f "$BTOTAL_FILE" ]; then
    echo "ERROR: BTOTAL_GARDEN not found. Run scripts/exp4_garden_baseline.sh first."
    exit 1
fi

B_TOTAL=$(cat "$BTOTAL_FILE")
echo "=== Exp 4 Step 2: Garden no-harm control (B_total=$B_TOTAL) ==="
mkdir -p "$DST"

# ── A-only ──
echo ""
echo "--- A-only controller × 5 seeds ---"
for seed in 0 1 2 3 4; do
    echo "  seed=$seed"
    python train.py \
        -s "$SRC" --ds 8 \
        -m "$DST/a_only/seed_$seed" \
        --fork 2 --base_layer 10 --visible_threshold 0.0 \
        --dist2level round --update_ratio 0.2 \
        --iterations $ITERS --update_until $UPDATE_UNTIL \
        --test_iterations 7000 15000 $UPDATE_UNTIL $ITERS \
        --save_iterations $UPDATE_UNTIL $ITERS \
        --seed $seed --B_total $B_TOTAL &
    sleep 30s
done
wait

# ── A+B (λ=1, M=10, K=4 — lowest cost, liveliest signal) ──
echo ""
echo "--- A+B (λ=1, M=10, K=4) × 5 seeds ---"
for seed in 0 1 2 3 4; do
    echo "  seed=$seed"
    python train.py \
        -s "$SRC" --ds 8 \
        -m "$DST/a_plus_b/seed_$seed" \
        --fork 2 --base_layer 10 --visible_threshold 0.0 \
        --dist2level round --update_ratio 0.2 \
        --iterations $ITERS --update_until $UPDATE_UNTIL \
        --test_iterations 7000 15000 $UPDATE_UNTIL $ITERS \
        --save_iterations $UPDATE_UNTIL $ITERS \
        --seed $seed --B_total $B_TOTAL \
        --b_enabled --fusion_lambda 1.0 \
        --b_camlist_size 4 --b_refresh_period 10 &
    sleep 30s
done
wait

echo ""
echo "=== Step 2 complete ==="
echo "Compare results.json across arms:"
echo "  A-only:   $DST/a_only/seed_*/results.json"
echo "  A+B:      $DST/a_plus_b/seed_*/results.json"
echo "Assert |ΔPSNR| ≤ 2σ_garden (from Step 1 Arm B) at all checkpoints."
