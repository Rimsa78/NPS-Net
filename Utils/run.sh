#!/usr/bin/env bash
# ==============================================================================
# run.sh — Baseline models launcher
#
# Usage:
#   ./run.sh vanilla              # train VanillaUNet
#   ./run.sh attunet              # train AttentionUNet
#   ./run.sh resunet              # train ResUNet
#   ./run.sh polar_unet           # train PolarUNet
#   ./run.sh transunet            # train TransUNet
#   ./run.sh all                  # train all baselines sequentially
#   ./run.sh eval vanilla         # evaluate VanillaUNet on test + external
#   ./run.sh eval all             # evaluate all baselines
#   ./run.sh refuge vanilla       # evaluate VanillaUNet on REFUGE
#   ./run.sh refuge npsnet        # evaluate NPSNet on REFUGE
#   ./run.sh refuge all           # evaluate ALL models (baselines + NPSNet) on REFUGE
#   ./run.sh papila vanilla       # evaluate VanillaUNet on Papila
#   ./run.sh papila npsnet        # evaluate NPSNet on Papila
#   ./run.sh papila all           # evaluate ALL models (baselines + NPSNet) on Papila
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Conda ─────────────────────────────────────────────────────────────────────
CONDA_ENV="myenv"
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "✓ Activated: $CONDA_ENV"
fi

MODELS="vanilla attunet resunet polar_unet transunet beal dofe"

# ── Parse ─────────────────────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: ./run.sh <model|all|eval|refuge> [model_name|all]"
    echo "Models: $MODELS npsnet"
    echo "Commands:"
    echo "  ./run.sh <model>           Train a single baseline"
    echo "  ./run.sh all               Train all baselines"
    echo "  ./run.sh eval <model|all>  Evaluate on test + external sets"
    echo "  ./run.sh refuge <model|all> Evaluate on REFUGE dataset"
    echo "  ./run.sh papila <model|all> Evaluate on Papila dataset"
    exit 1
fi

MODE="$1"
shift

# ── Train all ─────────────────────────────────────────────────────────────────
if [ "$MODE" = "all" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo " Training ALL baselines sequentially"
    echo "═══════════════════════════════════════════════════════════════"
    for m in $MODELS; do
        echo ""
        echo ">>> Training: $m"
        LOG="training_${m}_$(date +%Y%m%d_%H%M%S).log"
        python train.py --model "$m" 2>&1 | tee "$LOG"
        echo "  Log: $LOG"
    done
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " All baselines trained!"
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
fi

# ── Evaluate ──────────────────────────────────────────────────────────────────
if [ "$MODE" = "eval" ]; then
    TARGET="${1:-all}"
    if [ "$TARGET" = "all" ]; then
        EVAL_MODELS="$MODELS"
    else
        EVAL_MODELS="$TARGET"
    fi

    echo "═══════════════════════════════════════════════════════════════"
    echo " Evaluating: $EVAL_MODELS"
    echo "═══════════════════════════════════════════════════════════════"
    for m in $EVAL_MODELS; do
        echo ""
        echo ">>> Evaluating: $m"
        python inference.py --model "$m" --test --all-external --save-vis 2>&1
    done
    exit 0
fi

# ── REFUGE evaluation ─────────────────────────────────────────────────────────
if [ "$MODE" = "refuge" ]; then
    TARGET="${1:-all}"
    EXTRA_ARGS="${2:-}"

    echo "═══════════════════════════════════════════════════════════════"
    echo " REFUGE Evaluation: $TARGET"
    echo "═══════════════════════════════════════════════════════════════"

    CMD="python inference_refuge.py --model $TARGET --save-vis"
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi

    echo "Running: $CMD"
    eval "$CMD" 2>&1
    exit 0
fi

# ── Papila evaluation ─────────────────────────────────────────────────────────
if [ "$MODE" = "papila" ]; then
    TARGET="${1:-all}"
    EXTRA_ARGS="${2:-}"

    echo "═══════════════════════════════════════════════════════════════"
    echo " Papila Evaluation: $TARGET"
    echo "═══════════════════════════════════════════════════════════════"

    CMD="python inference_papila.py --model $TARGET --save-vis"
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi

    echo "Running: $CMD"
    eval "$CMD" 2>&1
    exit 0
fi

# ── Train single model ───────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo " Training: $MODE"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p checkpoints

LOG="training_${MODE}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG"

python train.py --model "$MODE" 2>&1 | tee "$LOG"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Done. Log: $LOG"
echo "═══════════════════════════════════════════════════════════════"
