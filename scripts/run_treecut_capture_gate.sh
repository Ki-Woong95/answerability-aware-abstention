#!/usr/bin/env bash
set -Eeo pipefail
run_directory="$1"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MODELSCOPE_CACHE=/tmp/v10-modelscope
export MPLCONFIGDIR=/tmp/v10-matplotlib NUMBA_CACHE_DIR=/tmp/v10-numba-cache
trap 'echo "TREECUT CAPTURE/GATE CHAIN FAILED exit=$?" >&2; touch "$run_directory/capture_gate.failed"' ERR
activate_model() {
 case "$1" in
  af_next) conda activate af-next ;;
  moss) conda activate moss-audio ;;
  audio-reasoner) conda activate audio-reasoner ;;
  *) return 2 ;;
 esac
}
run_stage() {
 local stage="$1"
 shift
 echo "START $stage at $(date -Is)"
 "$@" 2>&1 | tee -a "$run_directory/$stage.log"
 echo "COMPLETE $stage at $(date -Is)"
}
echo 'Waiting for TreeCut full TTS and frozen input QC'
while [[ ! -f "$run_directory/data.complete" ]]; do
 if [[ -f "$run_directory/data.failed" ]]; then echo 'Data QC failed; capture blocked' >&2; exit 1; fi
 sleep 20
done
conda activate audio-reasoner
run_stage data_report python -m scripts.audits.report_treecut_dataset
for model in af_next moss audio-reasoner; do
 activate_model "$model"
 run_stage "${model}_verify" python -m scripts.audits.verify_treecut_execution --run-directory "$run_directory"
 run_stage "${model}_smoke" python -m src.capture.run_treecut --model "$model" --split gate_train \
  --smoke-stratified --output "$run_directory/smoke" --alignment-audit
 touch "$run_directory/$model.smoke.complete"
done
touch "$run_directory/smoke.complete"
for model in af_next moss audio-reasoner; do
 activate_model "$model"
 run_stage "${model}_verify_full" python -m scripts.audits.verify_treecut_execution --run-directory "$run_directory"
 for split in gate_train calibration val; do
  run_stage "${model}_${split}" python -m src.capture.run_treecut --model "$model" --split "$split" --resume
 done
 run_stage "${model}_gate" python -m src.gate.train_treecut --model "$model" --device cuda:0
 run_stage "${model}_complete_audit" python -m scripts.audits.verify_treecut_execution --run-directory "$run_directory" --model "$model" --complete
 touch "$run_directory/$model.complete"
done
touch "$run_directory/capture_gate.complete"
echo 'TREECUT THREE-MODEL CAPTURE AND GATE FITTING COMPLETE; TEST UNTOUCHED'
