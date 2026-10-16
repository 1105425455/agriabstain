#!/usr/bin/env bash
# 16 个 (pooling, C) 拟合并行铺开:sklearn lbfgs 是 CPU 求解器,GPU 无加速,
# 这台机器 96 核,每个进程限 6 个 BLAS 线程,墙钟时间压到单个拟合量级。
set -eo pipefail
source /user/huangyongbin/miniconda3/etc/profile.d/conda.sh
conda activate agrisam3
cd /user/huangyongbin/newrepo/agriabstain/papermodel_lineargate
REPR=/user/huangyongbin/newrepo/agriabstain/new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/repr
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
mkdir -p runs/frozen_base/logs
find runs/frozen_base -maxdepth 1 -name 'metrics_*.jsonl' -delete
for pool in last_token visual_mean visual_max visual_query_attn; do
  for c in 0.01 0.1 1 10; do
    tag="${pool}_C${c}"
    python scripts/frozen_probe.py --repr-root "$REPR" --out runs/frozen_base \
      --pools "$pool" --c-values "$c" --tag "$tag" \
      > "runs/frozen_base/logs/${tag}.log" 2>&1 &
  done
done
wait
echo "[launch] all 16 shards finished"
