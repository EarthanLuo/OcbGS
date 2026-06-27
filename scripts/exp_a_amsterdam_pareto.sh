#!/bin/bash
# Exp A — amsterdam controllable-budget Pareto: demand vs uniform.
#
# Two arms, both with --grow_relax_scale 0.1 so the controller hits B_total:
#   demand   : gradient demand allocation (A-only; b_enabled default off)
#   uniform  : --demand_uniform -> controller B_total/N_active flat allocation
# Sweep B_total x {0.25,0.5,1,2}x baseline. Train-set metrics (no --eval);
# held-out test is rendered separately at the locked point (Task 6).
#
# set -e + bare (( var++ )) aborts on the first 0->1 increment, so every
# counter mutation is guarded with || true (CLAUDE.md). Per-seed resume guard
# skips any seed whose results.json already exists.

set -e
export PYTHONWARNINGS=ignore

SRC=/root/autodl-tmp/bungeenerf/amsterdam
DST=/root/autodl-tmp/expA/amsterdam
BTOTAL_FILE=/root/autodl-tmp/exp4/bungeenerf/amsterdam/BTOTAL_amsterdam
ITERS=30000
UPDATE_UNTIL=25000
CHECKPOINTS=(30000)
SAVE_CHECKPOINTS=(25000 30000)
SEEDS=(0 1 2)
MAX_JOBS=${MAX_JOBS:-1}          # single GPU
FACTORS=(0.25 0.5 1 2)
RELAX=0.1

mkdir -p "$DST"

if [ ! -f "$BTOTAL_FILE" ]; then
    echo "ERROR: $BTOTAL_FILE missing — measure B_total first (eval-plan §4)."
    exit 1
fi
B_TOTAL=$(cat "$BTOTAL_FILE")
echo "=== Exp A amsterdam Pareto ===  B_total=$B_TOTAL  relax=$RELAX"

run_arm () {
    # $1 = arm name (demand|uniform); $2... = extra train.py flags
    local arm="$1"; shift
    local extra=("$@")
    for factor in "${FACTORS[@]}"; do
        local B_val
        B_val=$(python -c "print(int($B_TOTAL * $factor))")
        local armdir="$DST/$arm/arm_${factor}x"
        mkdir -p "$armdir"
        echo ""
        echo "--- arm=$arm factor=$factor (B_total=$B_val) ---"

        # Feasibility probe on seed_0 (synchronous): set_control_level's guard
        # raises a non-zero exit if 0.25x is infeasible for this scene.
        if [ -f "$armdir/seed_0/results.json" ]; then
            echo "  seed=0 — DONE (skip probe)"
        else
            echo "  seed=0 (feasibility probe)"
            if ! python ocbgs/train.py \
                -s "$SRC" --ds 1 -m "$armdir/seed_0" \
                --fork 2 --base_layer 10 --visible_threshold 0.0 \
                --dist2level round --update_ratio 0.2 \
                --iterations $ITERS --update_until $UPDATE_UNTIL \
                --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --seed 0 --port $((6009 + RANDOM % 1000)) \
                --B_total $B_val --grow_relax_scale $RELAX "${extra[@]}"; then
                echo "  arm=$arm factor=$factor INFEASIBLE — skip"
                continue
            fi
        fi

        local _running=0
        for seed in "${SEEDS[@]}"; do
            [ "$seed" -eq 0 ] && continue
            if [ -f "$armdir/seed_$seed/results.json" ]; then
                echo "  seed=$seed — DONE (skip)"; continue
            fi
            while (( _running >= MAX_JOBS )); do
                wait -n 2>/dev/null || true
                (( _running-- )) || true
            done
            echo "  seed=$seed"
            python ocbgs/train.py \
                -s "$SRC" --ds 1 -m "$armdir/seed_$seed" \
                --fork 2 --base_layer 10 --visible_threshold 0.0 \
                --dist2level round --update_ratio 0.2 \
                --iterations $ITERS --update_until $UPDATE_UNTIL \
                --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --seed $seed --port $((6009 + RANDOM % 1000)) \
                --B_total $B_val --grow_relax_scale $RELAX "${extra[@]}" &
            (( _running++ )) || true
            sleep 10s
        done
        wait
    done
}

run_arm demand
run_arm uniform --demand_uniform

echo ""
echo "=== Exp A amsterdam sweep done — collate with: ==="
echo "python scripts/collect_results.py pareto --root $DST \\"
echo "    --arms demand uniform --factors ${FACTORS[*]} \\"
echo "    --step $UPDATE_UNTIL --checkpoint ${CHECKPOINTS[0]} --output $DST/pareto.csv"
