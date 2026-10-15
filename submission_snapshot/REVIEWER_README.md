# ReGate — Code and Data Supplement

This archive accompanies the anonymous submission *ReGate: Mitigating Hallucinations in
Visual Grounding via Metamorphic Counterfactual Supervision*. It contains the benchmark
annotations, the construction, training, evaluation and analysis code, the recorded
environment, the runtime logs, and the record-level predictions behind the reported results.

Unpacking the archive yields a single top-level directory,
`regate_code_data_supplement/`; all commands below are run from inside it. Per-directory
`README.md` files document the individual artifact directories and the fields the paper reads
from them.

Two sections need neither images nor GPUs: §4 verifies the shipped artifacts and recomputes
reported numbers in a few minutes, and §8 maps each table and figure to the file it comes
from. §2, §5 and §6 apply only to rebuilding the images or retraining.

## 1. Contents

| Path | Contents |
|------|----------|
| `benchmark/annotations/` | Raw, split, SFT, evaluation and audit annotations (27 files). Raw and split records carry `sample_id`, `group_id`, `source_image_id`, `split`, `decision` and `metadata.relation`; the SFT and evaluation exports carry the labels and metadata each consumer reads. |
| `benchmark/MANIFEST.json` | SHA-256, byte size and record count per annotation file, plus the path audit and image counts. |
| `benchmark/IMAGE_MANIFEST.json` | 1,254 entries: relative path, SHA-256, byte size and domain for every image the benchmark uses. |
| `configs/` | The PlantDoc and COCO construction configurations read by §5. |
| `reason_abstain/bench/` | Benchmark construction package: relation operators, transforms, split assignment, Qwen export. |
| `reason_abstain/eval/` | Metric and parser package. The metrics of Tables 3–6 are defined in `eval/metrics.py`. |
| `scripts/` | Construction, split, evaluation-probe and audit tools. |
| `training/sft_final/` | Stage-1 adaptation and Stage-2 head calibration. |
| `training/margin_kl/` | The two margin+KL continuations. |
| `training/qwenvl/` | Vendored Qwen3-VL training framework (see §7). |
| `evaluation/` | The single inference and scoring entry point, with its launcher. |
| `experiments/` | Mechanism experiments. Each `main/` directory holds the frozen results the paper reports. |
| `environment/` | Interpreter, OS, accelerator and dependency versions of the reported runs. |
| `repro_artifacts/` | Record-level predictions, gate probabilities, runtime logs, and the GPT-5.6 Sol request configurations and responses. |
| `requirements.txt` | Dependencies grouped by task, pinned to the recorded versions (see §3). |
| `RELEASE_MANIFEST.json` | SHA-256 and size of every file in this archive. |

Absolute paths inside logs and JSON artifacts appear as `<REPO_ROOT>`, `<USER_HOME>` and
`<MODEL_ROOT>`, the interpreter environment name as `<ENV>`, and the GPT-5.6 Sol service
address as `<GPT56_ENDPOINT>`.

The three evidence states carry shorter names in the records and metric keys than in the
paper:

| Paper | Record fields | Gate probability | Per-relation metric key |
|-------|---------------|------------------|-------------------------|
| supported | `decision: answer` | `p_answer` | `OOD_answer` |
| contradiction | `decision: abstain`, `reason: wrong_object` | `p_wo` | `OOD_confusion` |
| unavailable | `decision: abstain`, `reason: no_target` | `p_nt` | `OOD_absent` |

Not included:

- **Images.** The source images (PlantDoc, COCO, RefCOCOg, PlantWild) stay under their
  original licences and are not redistributed here. `benchmark/IMAGE_MANIFEST.json` pins
  every image by SHA-256, so copies fetched from the original providers can be verified
  byte-for-byte. See §2.
- **Derived (metamorphically transformed) images.** Regenerated deterministically from the
  source images and the shipped configs. See §5.
- **Model weights.** The base Qwen3-VL-8B-Instruct checkpoint, the Stage-1 LoRA adapters,
  `discrim_musd.pt`, `zref_cache.pt` and the calibrated heads are far too large to ship.
  The scripts that produce all of them are included. See §6.

Tables 3, 5 and 6, the margin+KL results and the L2/L3 evidence analyses ship their
record-level predictions or gate probabilities, so their metrics can be recomputed on CPU
(§4); regenerating the predictions themselves requires the images and GPUs (§5, §6). Three
results ship as frozen outputs instead of per-record files: Table 4, whose two ablation runs
are not distributed, and the post-adaptation depth scan and frozen-base representation
comparison, whose inputs are the large `.npz` feature archives.

The code and annotations are included in this archive; upon publication they will also be
released publicly under a licence permitting free use for research purposes.
`training/qwenvl/` derives from a third-party Qwen3-VL fine-tuning framework; its Apache-2.0
notice is preserved in `training/qwenvl/train/train_qwen.py`.

## 2. Obtaining the images

| Domain | Count | Source |
|--------|-------|--------|
| PlantDoc (source) | 726 images + YOLO labels | PlantDoc dataset release |
| COCO 2017 train (source) | 203 images | COCO download page |
| RefCOCOg (external probe) | 181 images + `val` annotation parquet | RefCOCOg release |
| PlantWild closed-12 (external probe) | 144 images | PlantWild dataset release |

Place them so the paths in `benchmark/IMAGE_MANIFEST.json` resolve under `benchmark/`, i.e.
`benchmark/data/source/plantdoc/images/...`, `benchmark/data/source/coco/images/...`,
`benchmark/data/ood/refcocog/images/...` and
`benchmark/data/ood/plantwild_closed12/images/...`. For COCO, restrict
`instances_train2017.json` to the image IDs in the manifest and save it as
`benchmark/data/source/coco/annotations/instances_selected.json`. The RefCOCOg `val` parquet
belongs at `benchmark/data/ood/refcocog/metadata/val-00000-of-00003.parquet`; it is read only
when rebuilding `ood_refcocog.jsonl`, which the shipped annotations make unnecessary.

Verify the images once they are in place:

```bash
python -c "
import json, hashlib, os
images = json.load(open('benchmark/IMAGE_MANIFEST.json'))
errors = []
for img in images:
    p = 'benchmark/' + img['path']
    if not os.path.isfile(p):
        errors.append('MISSING ' + img['path'])
    elif hashlib.sha256(open(p,'rb').read()).hexdigest() != img['sha256']:
        errors.append('HASH ' + img['path'])
    elif os.path.getsize(p) != img['bytes']:
        errors.append('SIZE ' + img['path'])
print(f'{len(images)} entries, {len(errors)} problems')
for e in errors[:20]: print(' ', e)
"
```

## 3. Environment

`environment/training_environment.json` records the environment of the reported SFT and
grouping-ablation runs:
Python 3.11.15, PyTorch 2.8.0 (CUDA build 12.8), Transformers 4.57.0, PEFT 0.17.1,
DeepSpeed 0.16.4, NumPy 2.2.6, scikit-learn 1.9.0, on Linux 5.15 / glibc 2.35 with
8 x RTX 4090 (24,564 MiB each), 20 CPU cores and 200 GB of system memory.
`repro_artifacts/logs/grouping/environment.log` is the training-time record that the
overlapping fields are checked against.

`requirements.txt` lists the dependencies in four groups: the CPU-only checks in §4 need
only Python 3.10+ and group 1 (`numpy`, `scikit-learn`, `scipy`, `PyYAML`); group 2 adds
construction, auditing and plotting; group 3 is the GPU training and evaluation stack; and
group 4 is the test suite. Its pins are taken from the record above; the few imported
packages that record does not cover are listed without a pin.
`flash-attn` and `vllm` are optional: the former is imported under `try`/`except`
and the reported runs fall back to the standard attention path, and every evaluation behind
the paper ran with `--backend hf`, so neither is needed to reproduce a reported number.

## 4. Checking the reported numbers without images or GPUs

These run in minutes.

```bash
# Annotation hashes, record counts and the path audit
python -c "
import json, hashlib
m = json.load(open('benchmark/MANIFEST.json'))
pa = m['path_audit']
assert pa['old_prefix_occurrences_total'] == 0 and pa['missing_image_refs_total'] == 0
for a in m['annotations']:
    h = hashlib.sha256(open('benchmark/' + a['path'], 'rb').read()).hexdigest()
    assert h == a['sha256'], a['path']
print(f\"{len(m['annotations'])} annotation files: hashes and audit OK\")
"

# Every file in this archive against its recorded digest
python -c "
import json, hashlib, os
for e in json.load(open('RELEASE_MANIFEST.json'))['files']:
    assert os.path.isfile(e['relative']), e['relative']
    h = hashlib.sha256(open(e['relative'], 'rb').read()).hexdigest()
    assert h == e['sha256'], e['relative']
print('RELEASE_MANIFEST: all files match')
"

# Test suite: 234 passed, 2 skipped
python -m pytest -q
```

The suite covers the metric definitions, the output parser, the construction operators, the
capability-gate computation and the analysis pipeline. Three of its files assert directly
against shipped artifacts and so pin reported values:
`experiments/grouping_ablation/test_grouping_ablation.py` fixes the checkpoint selection at
steps 444 and 222, their development losses, and Table 4's MTS, MCC, FCA and E2E-mIoU
entries; `experiments/layer_selection/test_layer_selection.py` fixes the pre-adaptation
candidate comparison over API indices 9, 18 and 27 and its selection of index 27; and
`experiments/evidence_judge/test_evidence_judge.py` recomputes the L2 pairing from the
shipped gate probabilities and fixes it at 148 complete pairs. The two remaining skips
require per-run training artifacts and an L2 development extraction that are not in this
archive.

Any table cell can also be recomputed from the record-level predictions. For example, the
ReGate row of Table 3 on Comprehensive (FCA 0.132, retention 0.976) follows from the
three-class argmax rule applied to the released gate probabilities:

```bash
python -c "
import json
sup = uns = ans_sup = ans_uns = 0
for line in open('repro_artifacts/base_native_regate/predictions/regate/comprehensive/predictions.jsonl'):
    r = json.loads(line)
    answered = r['p_answer'] >= max(r['p_wo'], r['p_nt'])
    if r['gt']['decision'] == 'answer':
        sup += 1; ans_sup += answered
    else:
        uns += 1; ans_uns += answered
print(f'retention={ans_sup/sup:.3f}  FCA={ans_uns/uns:.3f}  supported={sup}  unsupported={uns}')
"
```

The same directory holds the Base and SFT-native rows, whose operating rule is the generated
decision itself (`pred.decision`) rather than a gate threshold.

## 5. Rebuilding the derived images

The annotations under `benchmark/annotations/` are final and need no regeneration. The SFT
records reference `data/derived/...` image paths, so the derived images must be rebuilt
before any training run that reads them. Rebuilding needs the source images from §2 and
several GB of disk:

```bash
python scripts/build_benchmark.py --dataset plantdoc --config configs/plantdoc_full.yaml --output /tmp/plantdoc_rebuild.jsonl
python scripts/build_benchmark.py --dataset coco     --config configs/coco_full.yaml     --output /tmp/coco_rebuild.jsonl
```

This yields 20,815 derived images (13,666 PlantDoc + 7,149 COCO), matching the unique
derived paths referenced by the raw annotations.

## 6. Full retraining

Stage 1 and Stage 2, then the margin+KL continuations. Seeds for Python, NumPy, PyTorch, the
data loader, sampling and generation are fixed by the launchers.

```bash
# 1. Base-model representation normalisation statistics
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
MUSD_OUTPUT=/scratch/discrim_musd.pt \
torchrun --nproc_per_node=8 training/sft_final/compute_musd.py

# 2. Stage-1 adaptation with the three-state objective (seed 42, head_lambda 0.4,
#    LoRA 16/32/0.05, lr 1e-4). The reported checkpoint-444 is the epoch-4 endpoint;
#    the launcher runs all 6 epochs.
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
DISCRIM_MUSD_PATH=/scratch/discrim_musd.pt \
SFT_OUTPUT_DIR=/scratch/sft_final \
bash training/sft_final/run_train.sh

# 3. Stage-2: extract representations from the selected checkpoint, then fit the gate
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
CALIB_ADAPTER=/scratch/sft_final/checkpoint-444 \
CALIB_LOG_DIR=/scratch/calibration \
torchrun --nproc_per_node=4 training/sft_final/calibrate_head.py --mode extract

BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
CALIB_ADAPTER=/scratch/sft_final/checkpoint-444 \
CALIB_LOG_DIR=/scratch/calibration \
python training/sft_final/calibrate_head.py --mode train

# 4. Evaluation (uses --system-override decision --max-pixels 262144)
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
ADAPTER=/scratch/sft_final/checkpoint-444 \
CALIBRATED_HEAD=/scratch/sft_final/checkpoint-444/discrim_head_calibrated.pt \
OUTPUT_DIR=/scratch/eval_sft_final \
bash evaluation/run_sft_final.sh
```

Representation extraction for the normalisation statistics and for gate calibration keeps
the processor's default resolution; the pixel budget above applies to inference only.

The margin+KL continuations initialise from the Stage-1 adapter and the calibrated head:

```bash
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
ZREF_ADAPTER=/scratch/sft_final/checkpoint-444 \
ZREF_CACHE_PATH=/scratch/zref_cache.pt \
torchrun --nproc_per_node=8 training/margin_kl/dump_zref.py

export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
export SFT_INIT_CKPT=/scratch/sft_final/checkpoint-444
export DISCRIM_MUSD_PATH=/scratch/discrim_musd.pt
export ZREF_CACHE_PATH=/scratch/zref_cache.pt
OUTPUT_DIR=/scratch/margin_kl_v1 bash training/margin_kl/run_v1.sh
OUTPUT_DIR=/scratch/margin_kl_v2 bash training/margin_kl/run_v2.sh

BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct CKPT_DIR=/scratch/margin_kl_v1/checkpoint-50 bash training/margin_kl/eval_closed12.sh
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct CKPT_DIR=/scratch/margin_kl_v2/checkpoint-60 bash training/margin_kl/eval_closed12.sh
```

## 7. Which code path produces the reported results

`training/qwenvl/` is a working copy of a Qwen3-VL fine-tuning framework and carries several
head variants. The reported path is
`training/qwenvl/model/qwen3vl_dual_head_3reason.py` (`Qwen3VLWithTriHead`), the three-class
discriminative head over supported / contradiction / unavailable used by Stage 1 and by both
margin+KL continuations. `training/qwenvl/train/train_qwen.py` selects it when the
environment variable `USE_TRI_HEAD=true` is set, which `training/sft_final/run_train.sh`,
`training/margin_kl/run_v1.sh` and `training/margin_kl/run_v2.sh` all export alongside
`--use_dual_head True` and `head_lambda=0.4`.

The framework's other head variants belong to two-class formulations with their own loss
weights and are not part of the reported pipeline: `qwen3vl_dual_head.py` (reached only when
`USE_TRI_HEAD` is unset), `qwen3vl_dual_head_2reason.py` (reached only under
`USE_2REASON_HEAD=true`), and the `aux_head.py` / `repr_head.py` trainers, which no launcher
in this archive instantiates.

Inference and scoring go through the single entry point
`evaluation/eval_qwen3vl_3reason.py`, and the metrics of Tables 3–6 are defined in
`reason_abstain/eval/metrics.py`. In that module `E2E_mIoU` and `E2E_mIoU_c` are the
canonical names; `A_Ground` and `A_Ground_c` are equal-valued aliases kept for compatibility
with older artifacts. The two supplement diagnostics are computed where they are used: the
worst-state score `m_ℓ` of the depth scan in `experiments/layer_selection/scan_layers.py`,
and the frozen-base probe MCC in `experiments/evidence_judge/train_probes.py`.

## 8. Paper claim to artifact

Main paper:

| Claim | Artifact |
|-------|----------|
| Table 1 (training composition) | `benchmark/annotations/sft/`, `benchmark/annotations/splits/split_manifest.json` |
| Table 2 (training-disjoint probes) | `benchmark/annotations/eval/`, `benchmark/annotations/eval/BUILD_AUDIT.json` |
| Table 3 (Base / SFT / ReGate at fixed operating points) | `experiments/base_native_regate/analysis/{baseline_points.csv,regate_argmax_points.csv,native_comparison.csv,summary.json}`; predictions in `repro_artifacts/base_native_regate/predictions/` |
| Table 4 (matched construction contrast) | `experiments/grouping_ablation/main/grouping_ablation_fixed_gt.json`; logs in `repro_artifacts/logs/grouping/` |
| L3 visibility-response figure | `experiments/evidence_judge/main/L3_test/analysis.json`; per-record scores in `repro_artifacts/evidence_judge/L3/` |
| Table 5 (routing interfaces) | `experiments/selective_grounding/main/prompt_interface_metrics_fixed_gt.json`; predictions in `repro_artifacts/prompt_interfaces/predictions/`. Its native-LM arm is the SFT-native run of Table 3, so `predictions/native_lm/` repeats the files under `repro_artifacts/base_native_regate/predictions/native/sft/`. |
| Table 6 (ReGate vs. GPT-5.6 Sol) | `repro_artifacts/gpt56_sol/*/comparison.json` with the matching `run_config.json` and `predictions.jsonl` |

Technical supplement:

| Claim | Artifact |
|-------|----------|
| Post-adaptation depth diagnostic, layer 27 (`tab:depth_scan`) | `experiments/layer_selection/main/post_adaptation/{dev_scan.json,best_layer.json,min_score_curve.json}` |
| Complete FCA–retention operating curves and figures | `experiments/base_native_regate/analysis/{comprehensive,plantwild,refcocog}_regate_curve.csv`; the plotted panels are `comprehensive_hallucination_retention_paper`, `plantwild_hallucination_retention` and `refcocog_hallucination_retention` |
| Development-selected operating point (52.0% relative FCA reduction) | `repro_artifacts/matched_retention_dev/` |
| Same-source grouping counts (`tab:explicit_grouping_counts`) | `benchmark/annotations/splits/split_manifest.json`, `experiments/grouping_ablation/main/` |
| PlantDoc capability gate at 0.40 (`tab:plantdoc_capability`) | `benchmark/annotations/audits/PLANTDOC_CAPABILITY_GATE_AUDIT.json` |
| margin+KL on PlantWild (`tab:margin_kl_plantwild`) | `experiments/margin_kl/main/v{1,2}_checkpoint-*/plantwild_real_closed12_v1/metrics.json`, with `trainer_state.json` one level up in each checkpoint directory; training logs in `repro_artifacts/logs/margin_kl/` |
| Frozen-base representation comparison (5 seeds) | `experiments/evidence_judge/probe_results.json` |
| L2 score sensitivity | `experiments/evidence_judge/main/L2_test/analysis.json`, `repro_artifacts/evidence_judge/L2/` |
| PlantWild exact/perceptual duplicate separation | `repro_artifacts/plantwild/dedup_audit.json`; the checks are implemented in `scripts/plantwild/build_ood_plantwild.py` |
| RefCOCOg image-ID isolation | `benchmark/annotations/eval/REFCOCOG_IMAGE_ID_ISOLATION_AUDIT.json` |
