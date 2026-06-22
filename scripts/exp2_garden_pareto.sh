#!/bin/bash
# Exp 2 — Garden Pareto curve: sweep B_total × {0.25, 0.5, 1, 2}× baseline
#
# Phase 0: B_total from shared btotal path (reuses exp4/exp1 measurement)
# Phase 1: Baseline reference point (Octree-GS, single anchor count)
# Phase 2: Pareto sweep — 4 arms at {0.25, 0.5, 1, 2}× (natural-budget, A-only)
# Phase 3: Table compare across all arms
#
# Feasibility guard: if a 0.25× budget is infeasible for this scene,
# set_control_level raises a clear ValueError (non-zero exit) —
# the factor is logged as INFEASIBLE and skipped.
#
# Output: /root/autodl-tmp/exp2/garden/

set -e

export PYTHONWARNINGS=ignore

SRC=/root/autodl-tmp/m360/garden
DST=/root/autodl-tmp/exp2/garden
BTOTAL_ROOT=/root/autodl-tmp/btotal
BTOTAL_FILE="$BTOTAL_ROOT/BTOTAL_GARDEN"
ITERS=30000
UPDATE_UNTIL=25000
CHECKPOINTS=(30000)
SAVE_CHECKPOINTS=(25000 30000)
SEEDS=(0 1 2 3 4)
MAX_JOBS=${MAX_JOBS:-3}
FACTORS=(0.25 0.5 1 2)

mkdir -p "$DST" "$BTOTAL_ROOT"

echo "=== Exp 2 Garden Pareto ==="
echo "Source: $SRC"
echo "Output: $DST"

# ── Phase 0: B_total ──
EXP4_BTOTAL="/root/autodl-tmp/exp4/garden/BTOTAL_GARDEN"
if [ -f "$BTOTAL_FILE" ]; then
    echo ""
    echo "=== Phase 0: SKIP (BTOTAL exists) ==="
elif [ -f "$EXP4_BTOTAL" ]; then
    cp "$EXP4_BTOTAL" "$BTOTAL_FILE"
    echo ""
    echo "=== Phase 0: Copied B_total from exp4 ==="
else
    echo ""
    echo "=== Phase 0: Measuring B_total ==="
    PORT=$((6009 + RANDOM % 1000))
    python ocbgs/train.py \
        -s "$SRC" --ds 8 \
        -m "$DST/arm_a/seed_0" \
        --fork 2 --base_layer 10 --visible_threshold 0.0 \
        --dist2level round --update_ratio 0.2 \
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
fi

B_TOTAL=$(cat "$BTOTAL_FILE")
echo "B_total=$B_TOTAL"

# ── Phase 1: Baseline reference point ──
echo ""
echo "=== Phase 1: Baseline reference (Octree-GS, --no_controller) ==="

ARM_BASELINE_DIR="$DST/arm_baseline"
mkdir -p "$ARM_BASELINE_DIR"

_running=0
for seed in "${SEEDS[@]}"; do
    if [ -f "$ARM_BASELINE_DIR/seed_$seed/results.json" ]; then
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
        -m "$ARM_BASELINE_DIR/seed_$seed" \
        --fork 2 --base_layer 10 --visible_threshold 0.0 \
        --dist2level round --update_ratio 0.2 \
        --iterations $ITERS --update_until $UPDATE_UNTIL \
        --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
        --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
        --seed $seed --port $PORT --no_controller &
    (( _running++ )) || true
    sleep 10s
done
wait

# ── Phase 2: Pareto sweep ──
echo ""
echo "=== Phase 2: Pareto sweep ==="

PARETO_SKIPS="$DST/pareto_skips.log"
> "$PARETO_SKIPS"

ARM_SUMMARIES=()
ARM_LABELS=()

for factor in "${FACTORS[@]}"; do
    B_val=$(python -c "print(int($B_TOTAL * $factor))")
    label="${factor}x"
    armdir="$DST/arm_$label"
    mkdir -p "$armdir"

    echo ""
    echo "--- Factor $factor (B_total=$B_val) ---"

    _running=0
    for seed in "${SEEDS[@]}"; do
        if [ -f "$armdir/seed_$seed/results.json" ]; then
            echo "  seed=$seed — DONE (skip)"
            continue
        fi
        while (( _running >= MAX_JOBS )); do
            wait -n 2>/dev/null || true
            (( _running-- )) || true
        done
        echo "  seed=$seed"
        PORT=$((6009 + RANDOM % 1000))
        if python ocbgs/train.py \
            -s "$SRC" --ds 8 \
            -m "$armdir/seed_$seed" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
            --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
            --seed $seed --port $PORT \
            --B_total $B_val & then
            (( _running++ )) || true
            sleep 10s
        else
            echo "  factor=$factor INFEASIBLE (set_control_level guard fired) — skip" >> "$PARETO_SKIPS"
            break
        fi
    done
    wait

    summary="$DST/arm_$label.json"
    python scripts/collect_results.py metrics \
        --glob "$armdir/seed_*" \
        --checkpoints "${CHECKPOINTS[@]}" \
        --output "$summary"
    ARM_SUMMARIES+=("$label=$summary")
done

# ── Phase 3: Pareto table ──
echo ""
echo "=== Phase 3: Pareto table ==="

BASELINE_SUMMARY="$DST/arm_baseline.json"
python scripts/collect_results.py metrics \
    --glob "$ARM_BASELINE_DIR/seed_*" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --output "$BASELINE_SUMMARY"

TABLE_ARGS=()
TABLE_ARGS+=("--arm" "baseline=$BASELINE_SUMMARY")
for entry in "${ARM_SUMMARIES[@]}"; do
    TABLE_ARGS+=("--arm" "$entry")
done

python scripts/collect_results.py table \
    "${TABLE_ARGS[@]}" \
    --baseline-label baseline

# ── Phase 3: anchor-count table ──
echo ""
echo "--- Anchor counts at update_until ---"
echo ""
echo "  baseline:"
python scripts/collect_results.py total_points \
    --glob "$ARM_BASELINE_DIR/seed_*" \
    --step $UPDATE_UNTIL \
    --aggregate mean

for factor in "${FACTORS[@]}"; do
    label="${factor}x"
    armdir="$DST/arm_$label"
    if [ -d "$armdir/seed_0" ]; then
        echo "  $label:"
        python scripts/collect_results.py total_points \
            --glob "$armdir/seed_*" \
            --step $UPDATE_UNTIL \
            --aggregate mean
    else
        echo "  $label: INFEASIBLE (see $PARETO_SKIPS)"
    fi
done

echo ""
echo "=== Exp 2 Garden Pareto done ==="
