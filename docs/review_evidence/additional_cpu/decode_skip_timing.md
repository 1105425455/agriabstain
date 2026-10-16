## 源日志 timing 能否量化 decode-skip（同 hardware/batch）

扫描 155 份日志，命中 timing 样式（`it/s`、`s/it`、latency、throughput、elapsed 等）的共 117 份。命中内容仅为 HF/eval 进度条（`it/s`），且各 run 的 GPU 数量/分片不同（如 openref run `nproc=2, allocated_gpus=4, parallelism=2 data-parallel + 2-way HF sharding`；groundingme run `nproc=4`），没有 per-decision latency、没有 generation token 计数、没有同机同 batch 的受控对比。结论：现有日志无法量化 generation-skipping/decode-skip 延迟收益；未进行 GPU 计时，仅报告缺失。JSON：`docs/review_evidence/additional_cpu/decode_skip_timing.json`。
