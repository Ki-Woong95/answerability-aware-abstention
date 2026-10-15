#!/usr/bin/env bash
# Frozen UMWP validation/calibration/intervention only. No test evaluation.
set -Eeo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: bash $0 PHYSICAL_GPU MODEL RUN_DIRECTORY" >&2
    exit 2
fi
gpu="$1"
model="$2"
run_directory="$3"
case "$model" in
    af_next) model_environment=af-next ;;
    moss) model_environment=moss-audio ;;
    audio-reasoner) model_environment=audio-reasoner ;;
    *) echo "Unsupported model: $model" >&2; exit 2 ;;
esac
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "$model_environment"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODELSCOPE_CACHE=/tmp/v10-modelscope MPLCONFIGDIR=/tmp/v10-matplotlib NUMBA_CACHE_DIR=/tmp/v10-numba-cache
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
mkdir -p "$run_directory/$model"
model_logs="$run_directory/$model"
stage=initialization
trap 'echo "FAILED model=$model stage=$stage exit=$? at $(date -Is)" >&2' ERR
run_stage() {
    stage="$1"
    shift
    echo "START model=$model stage=$stage at $(date -Is)"
    "$@" 2>&1 | tee -a "$model_logs/$stage.log"
    echo "COMPLETE model=$model stage=$stage at $(date -Is)"
}
run_stage verify_frozen_split python -c 'from src.data.frozen_umwp import CANONICAL_SPLIT, verify_frozen_split; verify_frozen_split(CANONICAL_SPLIT); print("Frozen split verified")'
run_stage capture_val python -m src.capture.run --model "$model" --split val --condition all --resume
run_stage select_layer python -m src.gate.select_layer --dataset umwp --model "$model" --device cuda:0
run_stage capture_calibration python -m src.capture.run --model "$model" --split calibration --condition all --resume
run_stage calibrate_tau5 python -m src.gate.calibrate_tau5 --dataset umwp --model "$model" --device cuda:0
run_stage diagnose_val python -m src.gate.diagnose_val_operating_point --dataset umwp --model "$model" --device cuda:0
run_stage intervention_preflight python -m src.intervention.run_val_intervention --dataset umwp --model "$model" --preflight
run_stage intervention_val python -m src.intervention.run_val_intervention --dataset umwp --model "$model" --resume --retry-errors
run_stage verify_completion python - "$model" <<'PY'
import json
import sys
from pathlib import Path
from src.gate.datasets import get_dataset_config, load_split_ids

model = sys.argv[1]
expected_pairs = load_split_ids(get_dataset_config('umwp'), 'val')
for seed in (17, 29, 43):
    path = Path('data/interventions') / model / 'val' / f'seed_{seed}.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities = [(r['pair_id'], r['condition']) for r in rows]
    expected = {(p, c) for p in expected_pairs for c in ('answerable', 'unanswerable')}
    assert len(identities) == len(expected) and set(identities) == expected, f'{path}: incomplete/duplicate records'
    assert all(r.get('test_read') is False for r in rows), f'{path}: test access'
    assert all(not r.get('intervention_error') for r in rows), f'{path}: intervention errors; inspect log and resume with --retry-errors'
    assert all(not r['triggered'] or (r.get('intervention_run') and r.get('intervention_result') is not None) for r in rows), f'{path}: missing triggered results'
    print(f'{model} seed={seed}: {len(rows)} complete validation records; no intervention errors')
PY
echo "CHAIN COMPLETE model=$model at $(date -Is)"
