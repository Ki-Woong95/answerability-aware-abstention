#!/usr/bin/env bash
set -Eeo pipefail
run_directory="$1"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate audio-reasoner
set -a
source .env.judge
set +a
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'Judge key is missing' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 MODELSCOPE_CACHE=/tmp/v10-modelscope
export MPLCONFIGDIR=/tmp/v10-matplotlib NUMBA_CACHE_DIR=/tmp/v10-numba-cache
trap 'echo "TEST JUDGE CHAIN FAILED exit=$?" >&2; touch "$run_directory/judge.failed"' ERR
echo 'Waiting for all three fixed-policy test intervention chains'
while [[ ! -f "$run_directory/af_next.complete" || ! -f "$run_directory/moss.complete" || ! -f "$run_directory/audio-reasoner.complete" ]]; do
    if compgen -G "$run_directory/*.failed" > /dev/null; then
        echo 'GPU chain failure detected; stopping judge coordinator' >&2
        exit 1
    fi
    sleep 20
done
python -m scripts.audits.verify_umwp_test_run --record "$run_directory/launch_plan.json"
python -m src.eval.build_judge_inputs --dataset umwp --split test --allow-test 2>&1 | tee -a "$run_directory/judge_inputs.log"
python -m src.eval.run_judge --dataset umwp --split test --kind vanilla --allow-test 2>&1 | tee -a "$run_directory/judge_vanilla.log"
python -m src.eval.run_judge --dataset umwp --split test --kind intervention --allow-test 2>&1 | tee -a "$run_directory/judge_intervention.log"
python -m src.eval.analyze_validation --dataset umwp --split test --rubric v2 2>&1 | tee -a "$run_directory/test_analysis.log"
python -m scripts.audits.verify_umwp_test_run --record "$run_directory/launch_plan.json"
touch "$run_directory/judge.complete"
echo 'UMWP FROZEN TEST JUDGE AND ANALYSIS COMPLETE'
