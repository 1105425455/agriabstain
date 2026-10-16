# Copied original scripts (provenance)

These files are byte-copies of the scripts that produced the primary evidence, taken from
the source repository `/user/huangyongbin/newrepo/agriabstain/papermodel_lineargate/scripts/`.
Only file permissions/paths were adapted; script bodies are unmodified.

| file | purpose | source |
|---|---|---|
| `frozen_probe.py` | CPU LR probe on the frozen base layer-27 cache | `source/papermodel_lineargate/scripts/frozen_probe.py` |
| `train_linear_gate.py` | fit linear gate on released calibration shards (100-record dev) | `source/papermodel_lineargate/scripts/train_linear_gate.py` |
| `reroute_linear.py` | apply a fitted linear gate to shared H27 predictions | `source/papermodel_lineargate/scripts/reroute_linear.py` |
| `score_mlp.py` | score the released MLP head at the paper operating point | `source/papermodel_lineargate/scripts/score_mlp.py` |
| `fit_gates.py` | Stage-2 MLP fit on calibration shards (4096-512-256-3, lr 1e-3, wd 1e-3, eval 50, patience 5) | `source/papermodel_lineargate/scripts/fit_gates.py` |
| `matched_retention.py` | AUROC + FCA at matched retention (threshold-free gate comparison) | `source/papermodel_lineargate/scripts/matched_retention.py` |
| `summarize.py` | gate-comparison collection vs the published ablation table | `source/papermodel_lineargate/scripts/summarize.py` |
| `check_features.py` | verify calibration shards reproduce a head's stored mu/sd | `source/papermodel_lineargate/scripts/check_features.py` |
| `run_calibrate.py` | seed wrapper delegating to `aaai_code/training/sft_final/calibrate_head.py` (GPU feature extraction; NOT used in this CPU review) | `source/papermodel_lineargate/scripts/run_calibrate.py` |

Path adaptations in the review scripts (no scripts here were edited, only copied):
- The fitted gates/reroute utilities in `../review_frozen/*.py` are separate CPU-only
  re-implementations; they import the unchanged metric code from
  `source/aaai_code/experiments/base_native_regate/analyze_native_regate.py`.
- `frozen_probe.py`/`train_linear_gate.py` accept `--repr-root` / `--calibration-dir`
  as CLI args, so repository-local absolute paths (base model, data splits, conda env of
  the GPU runs) are not hard-coded in the CPU review invocations.