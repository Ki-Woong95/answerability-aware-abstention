#!/usr/bin/env bash
set -Eeo pipefail
gpu="$1"
model="$2"
run_directory="$3"
case "$model" in
    af_next) model_environment=af-next ;;
    moss) model_environment=moss-audio ;;
    audio-reasoner) model_environment=audio-reasoner ;;
    *) exit 2 ;;
esac
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$model_environment"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export MODELSCOPE_CACHE=/tmp/v10-modelscope MPLCONFIGDIR=/tmp/v10-matplotlib NUMBA_CACHE_DIR=/tmp/v10-numba-cache
mkdir -p "$run_directory/$model"
stage=initialization
trap 'echo "FAILED model=$model stage=$stage exit=$?" >&2; touch "$run_directory/$model.failed"' ERR
run_stage() {
    stage="$1"
    shift
    echo "START model=$model stage=$stage at $(date -Is)"
    "$@" 2>&1 | tee -a "$run_directory/$model/$stage.log"
    echo "COMPLETE model=$model stage=$stage at $(date -Is)"
}
run_stage verify_frozen python -m scripts.audits.verify_umwp_test_run --record "$run_directory/launch_plan.json"
run_stage capture_test python -m src.capture.run --model "$model" --split test --condition all --resume --allow-test
run_stage diagnose_test python -m src.gate.diagnose_test_operating_point --dataset umwp --model "$model" --device cuda:0 --allow-test
run_stage intervention_preflight python -m src.intervention.run_test_intervention --dataset umwp --model "$model" --preflight --allow-test
run_stage intervention_test python -m src.intervention.run_test_intervention --dataset umwp --model "$model" --resume --retry-errors --allow-test
run_stage verify_completion python -m scripts.audits.verify_umwp_test_run --record "$run_directory/launch_plan.json" --model "$model" --complete
touch "$run_directory/$model.complete"
echo "TEST CHAIN COMPLETE model=$model at $(date -Is)"
