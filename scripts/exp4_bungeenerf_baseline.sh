#!/bin/bash
# Exp 4 Step 3 — BungeeNeRF baseline: B_total + σ
#
# A-only controller × 5 seeds per scene.
# Measures σ_PSNR/SSIM/LPIPS and per-scene B_total.
# Writes BTOTAL_<SCENE> for downstream Step 4.
#
# Output: /root/autodl-tmp/exp4/bungeenerf_baseline/<scene>/seed_0..4
#
# Scenes: amsterdam, quebec, rome (3 of 8 BungeeNeRF scenes —
# selected for maximum scale contrast).
#
# Parameters mirror upstream train_bungeenerf.sh:
#   --progressive True --fork 2 --base_layer 10 --visible_threshold 0.0
#   --dist2level round --update_ratio 0.2 --levels -1 --dist_ratio 0.99
#   --init_level -1 --extra_ratio 0.25 --extra_up 0.01

set -e

BUNGEE_ROOT=/root/autodl-tmp/bungeenerf
DST=/root/autodl-tmp/exp4/bungeenerf_baseline
ITERS=30000
UPDATE_UNTIL=25000
SEEDS=(0 1 2 3 4)

if [ $# -eq 0 ]; then
    SCENES="amsterdam quebec rome"
else
    SCENES="$@"
fi

echo "=== Exp 4 Step 3: BungeeNeRF baseline ==="
echo "Scenes: $SCENES"
echo "Output: $DST"
mkdir -p "$DST"

for SCENE in $SCENES; do
    SRC="$BUNGEE_ROOT/$SCENE"

    if [ ! -d "$SRC" ]; then
        echo "WARNING: $SRC not found — skipping $SCENE"
        continue
    fi

    BTOTAL_FILE="$DST/BTOTAL_$SCENE"
    echo ""
    echo "=== Scene: $SCENE ==="

    for seed in "${SEEDS[@]}"; do
        echo "  seed=$seed"
        python train.py \
            -s "$SRC" \
            -m "$DST/$SCENE/seed_$seed" \
            --fork 2 --base_layer 10 --visible_threshold 0.0 \
            --dist2level round --update_ratio 0.2 \
            --progressive True --levels -1 --dist_ratio 0.99 \
            --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
            --iterations $ITERS --update_until $UPDATE_UNTIL \
            --test_iterations 7000 15000 $UPDATE_UNTIL $ITERS \
            --save_iterations $UPDATE_UNTIL $ITERS \
            --seed $seed &
        sleep 30s
    done
    wait

    echo ""
    echo "=== $SCENE complete ==="
    echo "Extract B_total from TB total_points @ $UPDATE_UNTIL."
    echo "Compute σ from results.json across 5 seeds."
    echo "Write B_total to $BTOTAL_FILE."
done

echo ""
echo "=== Step 3 complete ==="
echo "Per-scene output: $DST/<scene>/seed_*/"
