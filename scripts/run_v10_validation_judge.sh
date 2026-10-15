#!/usr/bin/env bash
# API judge only; no speech-model GPU jobs or test-set evaluation.
set -Eeo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate audio-reasoner
set -a
source .env.judge
set +a
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo 'Set OPENAI_API_KEY in .env.judge before starting.' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 MODELSCOPE_CACHE=/tmp/v10-modelscope
export MPLCONFIGDIR=/tmp/v10-matplotlib NUMBA_CACHE_DIR=/tmp/v10-numba-cache
mkdir -p logs/judge
python -m src.eval.build_judge_inputs --dataset umwp --split val 2>&1 | tee -a logs/judge/val_build_inputs.log
python -m src.eval.run_judge --dataset umwp --split val --kind vanilla 2>&1 | tee -a logs/judge/val_vanilla.log
python -m src.eval.run_judge --dataset umwp --split val --kind intervention 2>&1 | tee -a logs/judge/val_intervention.log
python -m src.eval.analyze_validation --dataset umwp --split val --rubric v2 2>&1 | tee -a logs/judge/val_analysis.log
echo 'VALIDATION JUDGE AND ANALYSIS COMPLETE'
