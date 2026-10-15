# Answerability-Aware Abstention

Hidden-state monitoring and selective reassessment for speech reasoning models.

The project asks whether a small gate can detect insufficient information during reasoning and improve abstention while limiting unnecessary intervention on answerable questions. The speech backbones remain frozen. Vanilla and Gate receive the same explicit-abstention task prompt; a triggered gate adds a conditional reassessment instruction.

## Method

1. Capture generated token IDs and decoder hidden states at causal `.?!;` boundary-confirmation checkpoints.
2. Fit a separate MLP64 gate for each decoder layer and seed using a pair-balanced A/U loss.
3. Select a layer by mean validation trajectory AUROC.
4. Calibrate a positive threshold on an independent answerable-only split, targeting a 5% A false-brake risk under the calibration assumptions.
5. Replay the exact generated-token prefix at the first cumulative-mean crossing and continue with reassessment.

Zero-checkpoint responses remain in behavioral evaluations. Generation failures are recorded separately from successful abstentions. Output-token counts are not GPU latency or compute measurements.

## Models and data

| Model | Checkpoint | Runtime |
| --- | --- | --- |
| AF-Next | `nvidia/audio-flamingo-next-think-hf` | Python 3.11, PyTorch 2.6.0+cu118, Transformers 5.17.0 |
| MOSS | `OpenMOSS-Team/MOSS-Audio-8B-Thinking` | Python 3.12, PyTorch 2.9.1+cu128, Transformers 4.57.1 |
| Audio-Reasoner | `zhifeixie/Audio-Reasoner` | Python 3.10, PyTorch 2.6.0+cu118, Transformers 4.48.0, ms-swift 3.0.0 |

| Dataset | Retained A/U pairs | Gate training | Calibration A | Validation pairs | Test pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| UMWP | 2,556 | 1,423 | 360 | 386 | 387 |
| Matched TreeCut | 4,995 | 2,781 | 703 | 755 | 756 |

The complete problem and question are spoken using Kokoro-82M (`af_heart`, speed 1, 24 kHz). TreeCut U removes one necessary condition from its paired A problem. Its shorter U recordings are an interpretation limitation. Both datasets are math tasks rendered as synthetic speech.

Datasets, frozen split files, audio, model weights and generated artifacts are not distributed in this repository. Small input hashes and sanitized tokenizer metadata are retained for integrity checks. Populate licensed data separately, configure local model paths and rerun the input/tokenizer audits before capture. The repository is source-only, not a turnkey artifact bundle.

## UMWP test results

387 answerable and 387 unanswerable items per model. Gate values average seeds 17, 29 and 43.

| Model | A accuracy: Vanilla → Gate | U abstention: Vanilla → Gate |
| --- | ---: | ---: |
| AF-Next | 72.09 → 71.83% | 78.55 → 85.01% |
| MOSS | 81.91 → 81.74% | 50.65 → 56.68% |
| Audio-Reasoner | 43.93 → 44.10% | 11.63 → 18.43% |

These are descriptive results, not a claim of unchanged answer accuracy or prompt-independent awareness. TreeCut capture and gate fitting are complete for AF-Next and Audio-Reasoner; MOSS capture is in progress. TreeCut intervention results are not yet available.

## Code layout

```text
src/data/          Data preparation and frozen-input verification
src/models/        Native model wrappers, token decoding and state hooks
src/capture/       Shared prompts, trajectory storage, resume and failure handling
src/gate/          MLP fitting, layer selection and threshold calibration
src/intervention/  Exact-prefix replay and conditional continuation
src/eval/          Behavioral judge and paired outcome analysis
scripts/           Input audits and experiment launchers
tests/             CPU regression tests with synthetic fixtures
MOSS-Audio/        Vendored MOSS implementation
vendor/treecut_math/  Pinned TreeCut generator and license
```

Legacy compatibility modules are retained because shared regression tests use them; they do not add datasets to the active experiments.

## Setup and verification

Use separate environments for the three backends; pinned dependency lists are under `requirements/`. The environment versions above match the experimental installation. The lists are not complete platform lockfiles. Install a compatible PyTorch build before the CPU test dependencies:

```bash
python -m pip install -r requirements/test.txt
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 python -m unittest discover -s tests
```

The source-only CPU regression suite passed **167 tests** in the existing Audio-Reasoner environment. Verification used synthetic fixtures and small CPU gate calculations; it did not load backbone weights, access research datasets, run speech-model inference or send API requests.

Run commands from the repository root. Model inference requires local checkpoints and the omitted input artifacts. Audio-Reasoner refuses an unavailable local snapshot rather than silently downloading weights. Tokenizer audit records can be regenerated with `scripts/audits/check_v10_tokenizer.py --model MODEL --path LOCAL_SNAPSHOT`.

Task prompts are defined in `src/capture/v10_prompt.py`. Capture entry points are `python -m src.capture.run --help` for UMWP and `python -m src.capture.run_treecut --help` for TreeCut. The TreeCut automatic chain currently stops after capture and gate fitting; later dataset-specific evaluation wiring remains to be verified.

For shell launchers, `CONDA_BASE` defaults to `$HOME/miniconda3`. Set `KOKORO_MODEL_PATH` to override the pinned local TTS snapshot path. For behavioral judging, copy `.env.example` to `.env.judge` and fill it locally. Never commit credentials.

## References

- [Liu et al., AAAI 2026: Answering the Unanswerable Is to Err Knowingly](https://ojs.aaai.org/index.php/AAAI/article/view/40496)
- [Sun et al., 2024: UMWP](https://arxiv.org/abs/2403.03558)
- [Ouyang, ACL 2025: TreeCut](https://aclanthology.org/2025.acl-short.84/)
- [Angelopoulos et al.: Learn then Test](https://arxiv.org/abs/2110.01052)

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream source attribution and license notes.
