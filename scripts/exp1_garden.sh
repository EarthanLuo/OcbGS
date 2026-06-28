#!/bin/bash
# Exp 1 — Garden: matched vs natural (near-uniform no-harm control)
#
# Phase 1: Verify exp4 garden results + B_total
# Phase 2: Matched-budget arm (--no_plateau, A-only) — the only new arm
# Phase 3: Collect baseline (exp4 arm_a) + natural (exp4 arm_b) + matched (new)
# Phase 4: Table compare + verify matched Σn ≡ B_total + reached_phase2
#
# Reuses exp4 garden arms: baseline = exp4 arm_a, natural = exp4 arm_b.
# Exp 1 garden only runs one net-new arm (matched). If exp4 garden is not
# complete, the script errors out (re-run exp4 first).
#
# Output: /root/autodl-tmp/exp1/garden/

set -e

export PYTHONWARNINGS=ignore

SRC=/root/autodl-tmp/m360/garden
EXP4_DST=/root/autodl-tmp/exp4/garden
DST=/root/autodl-tmp/exp1/garden
BTOTAL_FILE="$EXP4_DST/BTOTAL_GARDEN"
ITERS=30000
UPDATE_UNTIL=25000
CHECKPOINTS=(30000)
SAVE_CHECKPOINTS=(25000 30000)
SEEDS=(${SEEDS:-0 1 2 3 4})
MAX_JOBS=${MAX_JOBS:-3}

mkdir -p "$DST"

echo "=== Exp 1 Garden ==="
echo "Source: $SRC"
echo "Output: $DST"

# ── Phase 1: Verify exp4 prereqs + B_total ──
echo ""
echo "=== Phase 1: Verify exp4 garden prereqs ==="

if [ ! -f "$BTOTAL_FILE" ]; then
    echo "ERROR: $BTOTAL_FILE not found — run exp4_garden.sh first"
    exit 1
fi

for seed in "${SEEDS[@]}"; do
    for arm in arm_a arm_b; do
        rf="$EXP4_DST/$arm/seed_$seed/results.json"
        if [ ! -f "$rf" ]; then
            echo "ERROR: $rf not found — exp4 garden not complete"
            exit 1
        fi
    done
done

B_TOTAL=$(cat "$BTOTAL_FILE")
echo "B_total=$B_TOTAL"
echo "exp4 arm_a (baseline) + arm_b (natural) verified"

# ── Phase 2: Matched-budget arm (new) ──
echo ""
echo "=== Phase 2: Matched-budget arm (--no_plateau, A-only) ==="
echo "B_total=$B_TOTAL  max_jobs=$MAX_JOBS"

MATCHED_DIR="$DST/arm_matched"
mkdir -p "$MATCHED_DIR"

_running=0
for seed in "${SEEDS[@]}"; do
    if [ -f "$MATCHED_DIR/seed_$seed/results.json" ]; then
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
        -m "$MATCHED_DIR/seed_$seed" \
        --fork 2 --base_layer 10 --visible_threshold 0.0 \
        --dist2level round --update_ratio 0.2 \
        --iterations $ITERS --update_until $UPDATE_UNTIL \
        --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
        --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
        --seed $seed --port $PORT \
        --B_total $B_TOTAL --no_plateau &
    (( _running++ )) || true
    sleep 10s
done
wait

# ── Phase 3: Collect metrics from all three arms ──
echo ""
echo "=== Phase 3: Collect metrics ==="

ARM_BASELINE="$DST/arm_baseline.json"
ARM_MATCHED="$DST/arm_matched.json"
ARM_NATURAL="$DST/arm_natural.json"

python scripts/collect_results.py metrics \
    --glob "$EXP4_DST/arm_a/seed_*" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --output "$ARM_BASELINE"

python scripts/collect_results.py metrics \
    --glob "$MATCHED_DIR/seed_*" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --output "$ARM_MATCHED"

python scripts/collect_results.py metrics \
    --glob "$EXP4_DST/arm_b/seed_*" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --output "$ARM_NATURAL"

# ── Phase 4: Compare + verify ──
echo ""
echo "=== Phase 4: Compare & Verify ==="

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
    log="$MATCHED_DIR/seed_$seed/outputs.log"
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
    --glob "$MATCHED_DIR/seed_*" \
    --step $UPDATE_UNTIL \
    --aggregate mean

echo ""
echo "=== Exp 1 Garden done ==="
echo "Baseline summary:  $ARM_BASELINE"
echo "Matched summary:   $ARM_MATCHED"
echo "Natural summary:   $ARM_NATURAL"
