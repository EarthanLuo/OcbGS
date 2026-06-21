#!/bin/bash
# Exp 4 — Garden: no-harm control (B_total + σ + A+B verify)
#
# Phase 1: Arm A — Octree-GS native × 5 seeds → B_total
# Phase 2: Arm B (A-only controller) + Arm C (A+B, λ=1,M=10,K=4) × 5 seeds
# Phase 3: Auto-compare A+B vs A-only vs σ
#
# Concurrency: MAX_JOBS (default 3) limits simultaneous train.py processes.
# Resume: a per-seed run is skipped when its results.json already exists, so
#         re-running the script never repeats completed work.
#
# Output: /root/autodl-tmp/exp4/garden/

set -e

export PYTHONWARNINGS=ignore

SRC=/root/autodl-tmp/m360/garden
DST=/root/autodl-tmp/exp4/garden
BTOTAL_FILE="$DST/BTOTAL_GARDEN"
ITERS=30000
UPDATE_UNTIL=25000
CHECKPOINTS=(7000 15000 25000 30000)
SEEDS=(0 1 2 3 4)
MAX_JOBS=${MAX_JOBS:-3}

mkdir -p "$DST"

echo "=== Exp 4 Garden ==="
echo "Source: $SRC"
echo "Output: $DST"

# ── Phase 1: Arm A — B_total measurement (5 seeds, max $MAX_JOBS parallel) ──
if [ ! -f "$BTOTAL_FILE" ]; then
    echo ""
    echo "=== Phase 1: Arm A (Octree-GS native → B_total) ==="
    _running=0
    for seed in "${SEEDS[@]}"; do
        if [ -f "$DST/arm_a/seed_$seed/results.json" ]; then
            echo "  seed=$seed — DONE (skip)"
            continue
        fi
        while (( _running >= MAX_JOBS )); do
            wait -n 2>/dev/null || true
            (( _running-- )) || true
        done
        echo "  seed=$seed"
        PORT=$((6009 + RANDOM % 1000))
        python ocbgs/train.py \
            -s "$SRC" --ds 8 \
            -m "$DST/arm_a/seed_$seed" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations $UPDATE_UNTIL $ITERS \
            --save_iterations $UPDATE_UNTIL $ITERS \
            --seed $seed --port $PORT --no_controller &
        (( _running++ )) || true
        sleep 10s
    done
    wait

    echo ""
    echo "=== Phase 1 complete — extracting B_total ==="
    python scripts/collect_results.py total_points \
        --glob "$DST/arm_a/seed_*" \
        --step $UPDATE_UNTIL \
        --aggregate mean \
        --output-btotal "$BTOTAL_FILE"
    echo ""
    echo "B_total written to $BTOTAL_FILE"
else
    echo ""
    echo "=== Phase 1: SKIP (BTOTAL_GARDEN exists) ==="
    cat "$BTOTAL_FILE"
fi

B_TOTAL=$(cat "$BTOTAL_FILE")

# ── Phase 2: Arm B (A-only) + Arm C (A+B) interleaved, max $MAX_JOBS ──
echo ""
echo "=== Phase 2: Arm B (A-only) + Arm C (A+B, λ=1, M=10, K=4) ==="
echo "B_total=$B_TOTAL   max_jobs=$MAX_JOBS"

_running=0
for seed in "${SEEDS[@]}"; do
    # Arm B — A-only
    if [ ! -f "$DST/arm_b/seed_$seed/results.json" ]; then
        while (( _running >= MAX_JOBS )); do
            wait -n 2>/dev/null || true
            (( _running-- )) || true
        done
        echo "  arm_b seed=$seed"
        PORT=$((6009 + RANDOM % 1000))
        python ocbgs/train.py \
            -s "$SRC" --ds 8 \
            -m "$DST/arm_b/seed_$seed" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations "${CHECKPOINTS[@]}" $ITERS \
            --save_iterations $UPDATE_UNTIL $ITERS \
            --seed $seed --port $PORT --B_total $B_TOTAL &
        (( _running++ )) || true
        sleep 10s
    else
        echo "  arm_b seed=$seed — DONE (skip)"
    fi

    # Arm C — A+B
    if [ ! -f "$DST/arm_c/seed_$seed/results.json" ]; then
        while (( _running >= MAX_JOBS )); do
            wait -n 2>/dev/null || true
            (( _running-- )) || true
        done
        echo "  arm_c seed=$seed"
        PORT=$((6009 + RANDOM % 1000))
        python ocbgs/train.py \
            -s "$SRC" --ds 8 \
            -m "$DST/arm_c/seed_$seed" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations "${CHECKPOINTS[@]}" $ITERS \
            --save_iterations $UPDATE_UNTIL $ITERS \
            --seed $seed --port $PORT --B_total $B_TOTAL \
            --b_enabled --fusion_lambda 1.0 \
            --b_camlist_size 4 --b_refresh_period 10 &
        (( _running++ )) || true
        sleep 10s
    else
        echo "  arm_c seed=$seed — DONE (skip)"
    fi
done
wait

# ── Phase 3: Collect + Compare ──
echo ""
echo "=== Phase 3: Collect & Compare ==="

python scripts/collect_results.py metrics \
    --glob "$DST/arm_b/seed_*" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --output "$DST/sigma_garden.json"

python scripts/collect_results.py metrics \
    --glob "$DST/arm_c/seed_*" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --output "$DST/summary_a_plus_b.json"

echo ""
python scripts/collect_results.py compare \
    --a-only "$DST/sigma_garden.json" \
    --a-plus-b "$DST/summary_a_plus_b.json" \
    --sigma "$DST/sigma_garden.json"

echo ""
echo "=== Garden done ==="
echo "Summary:  $DST/sigma_garden.json"
echo "A+B:      $DST/summary_a_plus_b.json"
echo "Decision printed above."
