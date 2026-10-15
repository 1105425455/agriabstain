# Runtime environment record

`training_environment.json` records the interpreter, operating system, accelerator build,
and dependency versions of the conda environment used for the reported SFT,
grouping-ablation, and margin+KL runs.

## Provenance

Package versions are read from the installed distribution metadata of the training
interpreter.

The four fields that the training launcher recorded at run time —
`python`, `torch`, `transformers`, and `cuda` — agree with
`repro_artifacts/logs/grouping/environment.log`:

| Field | Training-time log | This snapshot |
|-------|-------------------|---------------|
| python | 3.11.15 | 3.11.15 |
| torch | 2.8.0+cu128 | 2.8.0, CUDA build 12.8 |
| transformers | 4.57.0 | 4.57.0 |
| cuda | 12.8 | 12.8 |

## Scope

`nvidia_driver_version` and `nvcc` are `null`; the CUDA build the runs used is recorded as
`accelerator.torch_cuda_build`, and the training-time log holds the per-device record.

The `hardware` block is the resource allocation of the training job: 8 x NVIDIA GeForce RTX
4090 (24564 MiB each), 20 CPU cores, and 200 GB of system memory. The GPU inventory matches
the per-device lines in `repro_artifacts/logs/grouping/environment.log`.

Dependency versions describe the environment as of the reported runs. The launchers only
extend `PYTHONPATH`, so the environment must be activated before invoking them.
