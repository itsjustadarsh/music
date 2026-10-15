#!/usr/bin/env bash
# =============================================================================
# DO EVERYTHING — single script that goes from zero to trained models + results
#
# On the DGX, just do:
#   git clone <your-repo-url> music && cd music/dgx-code && ./do_everything.sh
#
# That's it. It installs deps, downloads FMA, trains all 7 models, ensembles,
# and prints a results table at the end.
#
# Optional env vars:
#   NGPU=8              — number of GPUs (default: auto-detect, fallback 1)
#   FMA_DATA_DIR=/path  — reuse existing FMA data instead of re-downloading
#   SKIP_INSTALL=1      — skip pip install (if deps already installed)
#   SKIP_DOWNLOAD=1     — skip FMA download (if data already exists)
#   ONLY=mert_lora_330m — run just one config (by run_name, comma-separated ok)
#
# Resume after interruption:
#   Just run ./do_everything.sh again — it skips completed runs and resumes
#   from last checkpoint for in-progress ones.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Colours for readability ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No colour

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; exit 1; }

# --- Detect GPUs ---
if [ -z "${NGPU:-}" ]; then
    if command -v nvidia-smi &>/dev/null; then
        NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
        NGPU=$((NGPU > 0 ? NGPU : 1))
    else
        NGPU=1
    fi
fi
log "GPUs: ${BOLD}${NGPU}${NC}"

# --- Record start time ---
STARTED_AT=$(date +%s)

# =============================================================================
# PHASE 1: Environment
# =============================================================================
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 1: Environment Setup${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    log "Installing Python dependencies..."
    pip install --quiet --upgrade \
        torch torchaudio transformers datasets librosa \
        scikit-learn peft accelerate bitsandbytes \
        pyyaml pandas numpy matplotlib seaborn \
        einops timm audiomentations 2>&1 | tail -5
    ok "Dependencies installed"
else
    warn "Skipping install (SKIP_INSTALL=1)"
fi

# Quick sanity check
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || fail "PyTorch import failed"
python -c "import transformers; print(f'Transformers {transformers.__version__}')" || fail "Transformers import failed"
ok "Environment OK"

# =============================================================================
# PHASE 2: Data
# =============================================================================
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 2: Download FMA-small${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"

export FMA_DATA_DIR="${FMA_DATA_DIR:-${SCRIPT_DIR}/fma_data}"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    log "Downloading FMA-small to ${FMA_DATA_DIR}..."
    python -m data.download
    ok "FMA-small ready"
else
    warn "Skipping download (SKIP_DOWNLOAD=1)"
fi

# Verify data exists
TRACKS_CSV="${FMA_DATA_DIR}/raw/fma_metadata/tracks.csv"
if [ ! -f "$TRACKS_CSV" ]; then
    fail "tracks.csv not found at ${TRACKS_CSV} — data download may have failed"
fi
ok "Data verified: $(ls "${FMA_DATA_DIR}/raw/fma_small/" | head -5 | wc -l) genre folders found"

# =============================================================================
# PHASE 3: Training
# =============================================================================
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 3: Train All Models${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"

mkdir -p outputs

# Define the run order: config_file:run_name
RUNS=(
    "configs/mert_lora_330m.yaml:mert_lora_330m"
    "configs/mert_lora_95m.yaml:mert_lora_95m"
    "configs/mert_unfreeze_330m.yaml:mert_unfreeze_330m"
    "configs/ast_finetune.yaml:ast_finetune"
    "configs/htsat_genre.yaml:htsat_genre"
    "configs/efficientat_genre.yaml:efficientat_genre"
    "configs/mert_multicrop_330m.yaml:mert_multicrop_330m"
)

# Filter to ONLY if set
if [ -n "${ONLY:-}" ]; then
    FILTERED=()
    IFS=',' read -ra ONLY_LIST <<< "$ONLY"
    for entry in "${RUNS[@]}"; do
        run_name="${entry#*:}"
        for wanted in "${ONLY_LIST[@]}"; do
            if [ "$run_name" = "$wanted" ]; then
                FILTERED+=("$entry")
            fi
        done
    done
    RUNS=("${FILTERED[@]}")
    warn "Running only: ${ONLY} (${#RUNS[@]} configs)"
fi

TOTAL=${#RUNS[@]}
COMPLETED=0
FAILED_RUNS=()

for i in "${!RUNS[@]}"; do
    entry="${RUNS[$i]}"
    config="${entry%%:*}"
    run_name="${entry#*:}"
    step=$((i + 1))

    echo ""
    log "${BOLD}[${step}/${TOTAL}] ${run_name}${NC}"

    # Skip if already has results
    if [ -f "outputs/${run_name}/eval_metrics.json" ]; then
        warn "Already completed — skipping (delete outputs/${run_name}/ to re-run)"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    # Pick launcher: DDP for multi-GPU, plain python for single
    if [ "$NGPU" -gt 1 ]; then
        CMD="torchrun --nproc_per_node=$NGPU train.py --config $config"
    else
        CMD="python train.py --config $config"
    fi

    # Resume if there's a last.pt checkpoint
    if [ -f "outputs/${run_name}/last.pt" ]; then
        warn "Found last.pt — resuming from checkpoint"
        CMD="$CMD --resume outputs/${run_name}/last.pt"
    fi

    log "Running: $CMD"
    RUN_START=$(date +%s)

    if $CMD; then
        RUN_END=$(date +%s)
        RUN_MINS=$(( (RUN_END - RUN_START) / 60 ))
        ok "${run_name} completed in ${RUN_MINS}m"
        COMPLETED=$((COMPLETED + 1))
    else
        RUN_END=$(date +%s)
        RUN_MINS=$(( (RUN_END - RUN_START) / 60 ))
        warn "${run_name} FAILED after ${RUN_MINS}m — continuing with next model"
        FAILED_RUNS+=("$run_name")
    fi
done

# =============================================================================
# PHASE 4: Ensemble
# =============================================================================
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 4: Build Ensemble${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"

# Collect all runs that have eval_metrics.json
ENSEMBLE_RUNS=()
for entry in "${RUNS[@]}"; do
    run_name="${entry#*:}"
    if [ -f "outputs/${run_name}/eval_metrics.json" ]; then
        ENSEMBLE_RUNS+=("outputs/${run_name}")
    fi
done

if [ ${#ENSEMBLE_RUNS[@]} -ge 2 ]; then
    log "Ensembling ${#ENSEMBLE_RUNS[@]} models..."
    python ensemble.py --runs "${ENSEMBLE_RUNS[@]}" && ok "Ensemble complete" || warn "Ensemble failed"
else
    warn "Need at least 2 successful runs for ensemble (got ${#ENSEMBLE_RUNS[@]})"
fi

# =============================================================================
# PHASE 5: Results Summary
# =============================================================================
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  RESULTS SUMMARY${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""

# Table header
printf "  ${BOLD}%-35s  %8s  %8s${NC}\n" "MODEL" "ACC" "F1"
printf "  %-35s  %8s  %8s\n" "-----------------------------------" "--------" "--------"

BEST_ACC=0
BEST_NAME=""

for metrics_file in outputs/*/eval_metrics.json; do
    [ -f "$metrics_file" ] || continue
    name=$(basename "$(dirname "$metrics_file")")
    result=$(python -c "
import json, sys
d = json.load(open('$metrics_file'))
t = d.get('track_level', d.get('clip_level', {}))
acc = t.get('accuracy', 0)
f1 = t.get('macro_f1', 0)
print(f'{acc:.4f} {f1:.4f}')
" 2>/dev/null || echo "N/A N/A")

    acc=$(echo "$result" | cut -d' ' -f1)
    f1=$(echo "$result" | cut -d' ' -f2)
    printf "  %-35s  %8s  %8s\n" "$name" "$acc" "$f1"

    # Track best
    if python -c "exit(0 if float('$acc') > float('$BEST_ACC') else 1)" 2>/dev/null; then
        BEST_ACC="$acc"
        BEST_NAME="$name"
    fi
done

echo ""
if [ -n "$BEST_NAME" ]; then
    echo -e "  ${GREEN}${BOLD}Best model: ${BEST_NAME} (acc=${BEST_ACC})${NC}"
fi

# Previous best reference
echo -e "  ${BLUE}Previous best (laptop, frozen probes): 58.5% ensemble${NC}"

# =============================================================================
# DONE
# =============================================================================
echo ""
ENDED_AT=$(date +%s)
TOTAL_MINS=$(( (ENDED_AT - STARTED_AT) / 60 ))
TOTAL_HRS=$(( TOTAL_MINS / 60 ))
REMAINING_MINS=$(( TOTAL_MINS % 60 ))

echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  DONE${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""
echo "  Completed: ${COMPLETED}/${TOTAL} models"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo -e "  ${RED}Failed: ${FAILED_RUNS[*]}${NC}"
fi
echo "  Total time: ${TOTAL_HRS}h ${REMAINING_MINS}m"
echo ""
echo "  Results:  outputs/*/eval_metrics.json"
echo "  Best:     outputs/${BEST_NAME:-'???'}/best.pt"
echo ""
echo "  To re-run a specific model:"
echo "    ONLY=mert_lora_330m ./do_everything.sh"
echo ""
echo "  To copy results back to laptop:"
echo "    scp -r outputs/*/eval_metrics.json outputs/*/summary.json user@laptop:~/Code/clgprjcts/music/outputs/"
echo ""
