# Runtime environment record

`training_environment.json` records the interpreter, operating system, accelerator build,
and dependency versions of the conda environment used for the reported SFT,
grouping-ablation, and margin+KL runs.

## Provenance

Package versions are read from the installed distribution metadata of the training
interpreter. Nothing in this file is inferred, and no experiment was re-executed to
produce it.

The four fields that the training launcher recorded at run time —
`python`, `torch`, `transformers`, and `cuda` — agree with
`repro_artifacts/logs/grouping/environment.log`:

| Field | Training-time log | This snapshot |
|-------|-------------------|---------------|
| python | 3.11.15 | 3.11.15 |
| torch | 2.8.0+cu128 | 2.8.0, CUDA build 12.8 |
| transformers | 4.57.0 | 4.57.0 |
| cuda | 12.8 | 12.8 |

## Scope and limitations

`nvidia_driver_version` and `nvcc` are `null`: the analysis container has no driver access,
so these probes return nothing usable.

The `hardware` block records the resources allocated to the training job: 8 x NVIDIA
GeForce RTX 4090 (24564 MiB each), 20 CPU cores, and 200 GB of system memory. The GPU
inventory is corroborated by the per-device lines in
`repro_artifacts/logs/grouping/environment.log`; CPU and memory allocations
are recorded by the authors, since the analysis container cannot observe the original job
allocation.

Dependency versions describe the environment as of the reported runs. The launchers do not
pin an environment name; they only extend `PYTHONPATH`, so the environment is activated
externally before invoking them.
