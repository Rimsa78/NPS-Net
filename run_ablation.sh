#!/usr/bin/env bash
# ==============================================================================
# run_ablation.sh — NPS-Net Ablation Study Launcher
#
# Trains and evaluates ablation variants B2, B3, B4.
# B1, B5, B6 use existing checkpoints (no retraining needed).
#
# Usage:
#   ./run_ablation.sh              # train all three variants sequentially
#   ./run_ablation.sh b2           # train only B2
#   ./run_ablation.sh b3           # train only B3
#   ./run_ablation.sh b4           # train only B4
#   ./run_ablation.sh eval         # evaluate all variants (no training)
#   ./run_ablation.sh eval b2      # evaluate only B2
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Conda environment ────────────────────────────────────────────────────────
CONDA_ENV="myenv"

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "✓ Activated conda env: $CONDA_ENV"
    echo "  Python: $(python --version)"
    echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
else
    echo "⚠ conda not found, using system Python"
fi

# ── GPU info ─────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# ── Create checkpoint directories ────────────────────────────────────────────
mkdir -p checkpoints/b2 checkpoints/b3 checkpoints/b4

# ── Parse args ───────────────────────────────────────────────────────────────
MODE="${1:-train-all}"
VARIANT="${2:-}"

case "$MODE" in
    b2|b3|b4)
        # Train a single variant
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo " Ablation — Training ${MODE^^}"
        echo "═══════════════════════════════════════════════════════════════"
        LOG_FILE="training_${MODE}_$(date +%Y%m%d_%H%M%S).log"
        echo "Logging to: $LOG_FILE"
        python train_ablation.py --variant "$MODE" 2>&1 | tee "$LOG_FILE"
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo " Training ${MODE^^} finished. Log: $LOG_FILE"
        echo "═══════════════════════════════════════════════════════════════"
        ;;

    train-all)
        # Train all three variants sequentially
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo " Ablation — Training B2, B3, B4 sequentially"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""

        for V in b2 b3 b4; do
            echo "────────────────────────────────────────────────────────────"
            echo " Starting ${V^^} training..."
            echo "────────────────────────────────────────────────────────────"
            LOG_FILE="training_${V}_$(date +%Y%m%d_%H%M%S).log"
            python train_ablation.py --variant "$V" 2>&1 | tee "$LOG_FILE"
            echo ""
            echo "  ${V^^} finished. Log: $LOG_FILE"
            echo ""
        done

        echo "═══════════════════════════════════════════════════════════════"
        echo " All ablation training complete."
        echo "═══════════════════════════════════════════════════════════════"
        ;;

    eval)
        # Evaluate one or all variants (standard single-pass for B1-B4)
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo " Ablation — Evaluation"
        echo "═══════════════════════════════════════════════════════════════"

        if [ -n "$VARIANT" ]; then
            VARIANTS=("$VARIANT")
        else
            VARIANTS=(b2 b3 b4)
        fi

        for V in "${VARIANTS[@]}"; do
            echo ""
            echo "────────────────────────────────────────────────────────────"
            echo " Evaluating ${V^^} ..."
            echo "────────────────────────────────────────────────────────────"

            if [ "$V" = "b5" ]; then
                # B5 uses Polar-TTA script
                python inference_polar_tta_ablation.py --test 2>&1
                python inference_polar_tta_ablation.py --all-external 2>&1 || true
            else
                python inference_ablation.py --variant "$V" --test 2>&1
                python inference_ablation.py --variant "$V" --all-external 2>&1 || true
            fi
        done

        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo " Evaluation complete."
        echo "═══════════════════════════════════════════════════════════════"
        ;;

    eval-all)
        # Evaluate ALL variants including B1 and B5(Polar-TTA)
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo " Ablation — Full Evaluation (B1-B5)"
        echo "═══════════════════════════════════════════════════════════════"

        # B1-B4: standard single-pass
        for V in b1 b2 b3 b4; do
            echo ""
            echo "────────────────────────────────────────────────────────────"
            echo " Evaluating ${V^^} (standard inference) ..."
            echo "────────────────────────────────────────────────────────────"
            python inference_ablation.py --variant "$V" --test --all-external 2>&1 || true
        done

        # B5: Polar-TTA on B4
        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo " Evaluating B5 (Polar-TTA on B4) ..."
        echo "────────────────────────────────────────────────────────────"
        python inference_polar_tta_ablation.py --test --all-external 2>&1 || true

        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo " Full evaluation complete."
        echo "═══════════════════════════════════════════════════════════════"
        ;;

    --help|-h)
        echo "Usage: ./run_ablation.sh [command] [variant]"
        echo ""
        echo "Commands:"
        echo "  (no args)     Train all variants (B2, B3, B4) sequentially"
        echo "  b2|b3|b4      Train a single variant"
        echo "  eval [variant] Evaluate trained variants (internal + external)"
        echo "  eval b5        Evaluate B5 (Polar-TTA on B4, no training needed)"
        echo "  eval-all       Evaluate all variants B1-B5"
        echo "  --help         Show this help"
        echo ""
        echo "Ablation Configurations:"
        echo "  B1: Polar UNet baseline (already trained)"
        echo "  B2: + Monotone occupancy (independent heads)"
        echo "  B3: + Factorized nesting (P_c = P_d · Q)"
        echo "  B4: + Shape prior & confidence gating"
        echo "  B5: + Polar-TTA on B4 (inference only, uses B4 checkpoint)"
        ;;

    *)
        echo "Unknown command: $MODE"
        echo "Run './run_ablation.sh --help' for usage."
        exit 1
        ;;
esac
