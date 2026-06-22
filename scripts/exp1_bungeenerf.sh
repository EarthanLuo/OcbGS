#!/bin/bash
# Exp 1 — BungeeNeRF: matched vs natural (non-uniform value judgment)
#
# Usage: bash scripts/exp1_bungeenerf.sh [amsterdam] [quebec] [rome] ...
#
# Phase 1: Quick B_total measurement (seed 0 only, sequential)
# Phase 2: Matched (--no_plateau) + Natural interleaved, A-only, 5 seeds
# Phase 3: Collect + table compare + verify matched Σn ≡ B_total + reached_phase2
#
# Output: /root/autodl-tmp/exp1/bungeenerf/<scene>/

set -e

export PYTHONWARNINGS=ignore

BUNGEE_ROOT=/root/autodl-tmp/bungeenerf
DST_ROOT=/root/autodl-tmp/exp1/bungeenerf
BTOTAL_ROOT=/root/autodl-tmp/btotal
ITERS=30000
UPDATE_UNTIL=25000
CHECKPOINTS=(30000)
SAVE_CHECKPOINTS=(25000 30000)
SEEDS=(0 1 2 3 4)
MAX_JOBS=${MAX_JOBS:-3}

if [ $# -eq 0 ]; then
    SCENES="amsterdam quebec rome"
else
    SCENES="$@"
fi

mkdir -p "$BTOTAL_ROOT"

echo "=== Exp 1 BungeeNeRF ==="
echo "Scenes: $SCENES"

for SCENE in $SCENES; do
    SRC="$BUNGEE_ROOT/$SCENE"
    DST="$DST_ROOT/$SCENE"
    BTOTAL_FILE="$BTOTAL_ROOT/BTOTAL_$SCENE"

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
        PORT=$((6009 + RANDOM % 1000))
        python ocbgs/train.py \
            -s "$SRC" \
            -m "$DST/arm_a/seed_0" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --progressive --levels -1 --dist_ratio 0.99 \
            --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
            --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
            --seed 0 --port $PORT --no_controller

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

    # ── Phase 2: Matched + Natural interleaved ──
    echo ""
    echo "--- Phase 2: Matched (--no_plateau) + Natural (A-only, plateau ON) ---"
    echo "  max_jobs=$MAX_JOBS"

    _running=0
    for seed in "${SEEDS[@]}"; do
        # Matched arm
        ARMDIR="$DST/arm_matched"
        mkdir -p "$ARMDIR"
        if [ -f "$ARMDIR/seed_$seed/results.json" ]; then
            echo "  matched seed=$seed — DONE (skip)"
        else
            while (( _running >= MAX_JOBS )); do
                wait -n 2>/dev/null || true
                (( _running-- )) || true
            done
            echo "  matched seed=$seed"
            PORT=$((6009 + RANDOM % 1000))
            python ocbgs/train.py \
                -s "$SRC" \
                -m "$ARMDIR/seed_$seed" \
                --fork 2 --base_layer 10 --visible_threshold 0.0 \
                --dist2level round --update_ratio 0.2 \
                --progressive --levels -1 --dist_ratio 0.99 \
                --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
                --iterations $ITERS --update_until $UPDATE_UNTIL \
                --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --seed $seed --port $PORT \
                --B_total $B_TOTAL --no_plateau &
            (( _running++ )) || true
            sleep 10s
        fi

        # Natural arm (plateau default ON)
        ARMDIR="$DST/arm_natural"
        mkdir -p "$ARMDIR"
        if [ -f "$ARMDIR/seed_$seed/results.json" ]; then
            echo "  natural seed=$seed — DONE (skip)"
        else
            while (( _running >= MAX_JOBS )); do
                wait -n 2>/dev/null || true
                (( _running-- )) || true
            done
            echo "  natural seed=$seed"
            PORT=$((6009 + RANDOM % 1000))
            python ocbgs/train.py \
                -s "$SRC" \
                -m "$ARMDIR/seed_$seed" \
                --fork 2 --base_layer 10 --visible_threshold 0.0 \
                --dist2level round --update_ratio 0.2 \
                --progressive --levels -1 --dist_ratio 0.99 \
                --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
                --iterations $ITERS --update_until $UPDATE_UNTIL \
                --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --seed $seed --port $PORT \
                --B_total $B_TOTAL &
            (( _running++ )) || true
            sleep 10s
        fi
    done
    wait

    # ── Phase 3: Collect + Compare ──
    echo ""
    echo "--- Phase 3: Collect & Compare ---"

    ARM_BASELINE="$DST/arm_baseline.json"
    ARM_MATCHED="$DST/arm_matched.json"
    ARM_NATURAL="$DST/arm_natural.json"

    python scripts/collect_results.py metrics \
        --glob "$DST/arm_a/seed_*" \
        --checkpoints "${CHECKPOINTS[@]}" \
        --output "$ARM_BASELINE"

    python scripts/collect_results.py metrics \
        --glob "$DST/arm_matched/seed_*" \
        --checkpoints "${CHECKPOINTS[@]}" \
        --output "$ARM_MATCHED"

    python scripts/collect_results.py metrics \
        --glob "$DST/arm_natural/seed_*" \
        --checkpoints "${CHECKPOINTS[@]}" \
        --output "$ARM_NATURAL"

    echo ""
    echo "--- Quality comparison (table) ---"
    python scripts/collect_results.py table \
        --arm "baseline=$ARM_BASELINE" \
        --arm "matched=$ARM_MATCHED" \
        --arm "natural=$ARM_NATURAL" \
        --baseline-label baseline

    echo ""
    echo "--- Matched arm: reached_phase2 check ---"
    matched_phase2=0
    for seed in "${SEEDS[@]}"; do
        log="$DST/arm_matched/seed_$seed/outputs.log"
        if grep -q "Phase 2 REACHED" "$log" 2>/dev/null; then
            echo "  seed=$seed: REACHED"
            (( matched_phase2++ )) || true
        elif grep -q "Phase 2 NOT reached" "$log" 2>/dev/null; then
            echo "  seed=$seed: NOT reached  <-- BUDGET NOT FILLED"
        else
            echo "  seed=$seed: no Phase-2 log line"
        fi
    done
    echo "  matched arm Phase 2 reached: $matched_phase2 / ${#SEEDS[@]} seeds"

    echo ""
    echo "--- Matched arm: Σn vs B_total ---"
    python scripts/collect_results.py total_points \
        --glob "$DST/arm_matched/seed_*" \
        --step $UPDATE_UNTIL \
        --aggregate mean

    echo ""
    echo "=== $SCENE done ==="
done

echo ""
echo "=== All BungeeNeRF scenes done ==="
