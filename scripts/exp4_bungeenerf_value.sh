#!/bin/bash
# Exp 4 Step 4 — BungeeNeRF value judgment: A+B vs A-only
#
# Prerequisite: BTOTAL_<SCENE> from Step 3.
# A+B (λ=1, M=10, K=16 — strongest B signal) vs A-only baseline.
# This is the KEEP/DROP decision point.
#
# Output: /root/autodl-tmp/exp4/bungeenerf_value/<scene>/
#   a_only/seed_0..4   — A-only controller
#   a_plus_b/seed_0..4 — A+B

set -e

BUNGEE_ROOT=/root/autodl-tmp/bungeenerf
DST=/root/autodl-tmp/exp4/bungeenerf_value
BASELINE_DST=/root/autodl-tmp/exp4/bungeenerf_baseline
ITERS=30000
UPDATE_UNTIL=25000
SEEDS=(0 1 2 3 4)

if [ $# -eq 0 ]; then
    SCENES="amsterdam quebec rome"
else
    SCENES="$@"
fi

echo "=== Exp 4 Step 4: BungeeNeRF value judgment ==="
echo "Scenes: $SCENES"
echo "Output: $DST"
mkdir -p "$DST"

for SCENE in $SCENES; do
    SRC="$BUNGEE_ROOT/$SCENE"
    BTOTAL_FILE="$BASELINE_DST/BTOTAL_$SCENE"

    if [ ! -d "$SRC" ]; then
        echo "WARNING: $SRC not found — skipping $SCENE"
        continue
    fi
    if [ ! -f "$BTOTAL_FILE" ]; then
        echo "ERROR: BTOTAL_$SCENE not found. Run scripts/exp4_bungeenerf_baseline.sh first."
        exit 1
    fi

    B_TOTAL=$(cat "$BTOTAL_FILE")
    echo ""
    echo "=== Scene: $SCENE (B_total=$B_TOTAL) ==="

    # ── A-only ──
    echo "--- A-only × 5 seeds ---"
    for seed in "${SEEDS[@]}"; do
        echo "  seed=$seed"
        python train.py \
            -s "$SRC" \
            -m "$DST/$SCENE/a_only/seed_$seed" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --progressive True --levels -1 --dist_ratio 0.99 \
            --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations 7000 15000 $UPDATE_UNTIL $ITERS \
            --save_iterations $UPDATE_UNTIL $ITERS \
            --seed $seed --B_total $B_TOTAL &
        sleep 30s
    done
    wait

    # ── A+B (λ=1, M=10, K=16) ──
    echo "--- A+B (λ=1, M=10, K=16) × 5 seeds ---"
    for seed in "${SEEDS[@]}"; do
        echo "  seed=$seed"
        python train.py \
            -s "$SRC" \
            -m "$DST/$SCENE/a_plus_b/seed_$seed" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --progressive True --levels -1 --dist_ratio 0.99 \
            --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations 7000 15000 $UPDATE_UNTIL $ITERS \
            --save_iterations $UPDATE_UNTIL $ITERS \
            --seed $seed --B_total $B_TOTAL \
            --b_enabled --fusion_lambda 1.0 \
            --b_camlist_size 16 --b_refresh_period 10 &
        sleep 30s
    done
    wait

    echo ""
    echo "=== $SCENE complete ==="
done

echo ""
echo "=== Step 4 complete ==="
echo "Compare results.json across arms:"
echo "  A-only:   $DST/<scene>/a_only/seed_*/results.json"
echo "  A+B:      $DST/<scene>/a_plus_b/seed_*/results.json"
echo ""
echo "Decision: if |ΔPSNR(A+B − A)| > 2σ_garden at any checkpoint → KEEP B."
echo "          Otherwise → DROP B (fallback per ADR-0002)."
