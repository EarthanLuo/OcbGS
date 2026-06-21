#!/bin/bash
# Exp 4 — BungeeNeRF: value judgment (σ + A+B vs A-only per scene)
#
# Usage: bash scripts/exp4_bungeenerf.sh [amsterdam] [quebec] [rome] ...
#
# Phase 1: Quick B_total measurement (seed 0 only, Arm A)
# Phase 2: A-only (Arm B) + A+B (Arm C, λ=1,M=10,K=16) × 5 seeds
# Phase 3: Auto-compare A+B vs A-only vs σ → KEEP/DROP
#
# Output: /root/autodl-tmp/exp4/bungeenerf/<scene>/

set -e

BUNGEE_ROOT=/root/autodl-tmp/bungeenerf
DST_ROOT=/root/autodl-tmp/exp4/bungeenerf
ITERS=30000
UPDATE_UNTIL=25000
CHECKPOINTS=(7000 15000 25000 30000)
SEEDS=(0 1 2 3 4)

if [ $# -eq 0 ]; then
    SCENES="amsterdam quebec rome"
else
    SCENES="$@"
fi

echo "=== Exp 4 BungeeNeRF ==="
echo "Scenes: $SCENES"

for SCENE in $SCENES; do
    SRC="$BUNGEE_ROOT/$SCENE"
    DST="$DST_ROOT/$SCENE"
    BTOTAL_FILE="$DST/BTOTAL_$SCENE"

    if [ ! -d "$SRC" ]; then
        echo "WARNING: $SRC not found — skipping $SCENE"
        continue
    fi

    mkdir -p "$DST"
    echo ""
    echo "=========================================="
    echo "=== Scene: $SCENE ==="
    echo "=========================================="

    # ── Phase 1: Quick B_total (seed 0 only) ──
    if [ ! -f "$BTOTAL_FILE" ]; then
        echo ""
        echo "--- Phase 1: Quick B_total (seed=0) ---"
        python train.py \
            -s "$SRC" \
            -m "$DST/arm_a/seed_0" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --progressive True --levels -1 --dist_ratio 0.99 \
            --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations $UPDATE_UNTIL $ITERS \
            --save_iterations $UPDATE_UNTIL $ITERS \
            --seed 0 --no_controller

        python scripts/collect_results.py total_points \
            --glob "$DST/arm_a/seed_0" \
            --step $UPDATE_UNTIL \
            --aggregate mean \
            --output-btotal "$BTOTAL_FILE"
        echo "B_total written to $BTOTAL_FILE"
    else
        echo ""
        echo "--- Phase 1: SKIP (BTOTAL exists) ---"
    fi

    B_TOTAL=$(cat "$BTOTAL_FILE")
    echo "B_total=$B_TOTAL"

    # ── Phase 2: A-only (Arm B) + A+B (Arm C) × 5 seeds ──
    echo ""
    echo "--- Phase 2: A-only + A+B × 5 seeds ---"

    # Arm B — A-only
    for seed in "${SEEDS[@]}"; do
        echo "  arm_b seed=$seed"
        python train.py \
            -s "$SRC" \
            -m "$DST/arm_b/seed_$seed" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --progressive True --levels -1 --dist_ratio 0.99 \
            --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations "${CHECKPOINTS[@]}" $ITERS \
            --save_iterations $UPDATE_UNTIL $ITERS \
            --seed $seed --B_total $B_TOTAL &
        sleep 30s
    done

    # Arm C — A+B (λ=1, M=10, K=16)
    for seed in "${SEEDS[@]}"; do
        echo "  arm_c seed=$seed"
        python train.py \
            -s "$SRC" \
            -m "$DST/arm_c/seed_$seed" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --progressive True --levels -1 --dist_ratio 0.99 \
            --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations "${CHECKPOINTS[@]}" $ITERS \
            --save_iterations $UPDATE_UNTIL $ITERS \
            --seed $seed --B_total $B_TOTAL \
            --b_enabled --fusion_lambda 1.0 \
            --b_camlist_size 16 --b_refresh_period 10 &
        sleep 30s
    done
    wait

    # ── Phase 3: Collect + Compare ──
    echo ""
    echo "--- Phase 3: Collect & Compare ---"

    python scripts/collect_results.py metrics \
        --glob "$DST/arm_b/seed_*" \
        --checkpoints "${CHECKPOINTS[@]}" \
        --output "$DST/sigma_$SCENE.json"

    python scripts/collect_results.py metrics \
        --glob "$DST/arm_c/seed_*" \
        --checkpoints "${CHECKPOINTS[@]}" \
        --output "$DST/summary_a_plus_b.json"

    echo ""
    python scripts/collect_results.py compare \
        --a-only "$DST/sigma_$SCENE.json" \
        --a-plus-b "$DST/summary_a_plus_b.json" \
        --sigma "$DST/sigma_$SCENE.json"

    echo ""
    echo "=== $SCENE done ==="
    echo "Summary:  $DST/sigma_$SCENE.json"
    echo "A+B:      $DST/summary_a_plus_b.json"
done

echo ""
echo "=== All BungeeNeRF scenes done ==="
