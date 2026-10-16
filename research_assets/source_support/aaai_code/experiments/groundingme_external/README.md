---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: description
    dtype: string
  - name: image
    dtype: image
  - name: bbox
    list: int64
  - name: detection_type
    dtype: string
  - name: subtask_l1
    dtype: string
  - name: subtask_l2
    dtype: string
  - name: width
    dtype: int64
  - name: height
    dtype: int64
  splits:
  - name: test
    num_bytes: 1532196041
    num_examples: 1005
  download_size: 1526836865
  dataset_size: 1532196041
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
task_categories:
- zero-shot-object-detection
license: other
tags:
- multimodal
- visual-grounding
- mllm
---

# (CVPR 2026) GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2512.17495-b31b1b.svg)](https://arxiv.org/abs/2512.17495)
[![GitHub](https://img.shields.io/badge/GitHub-GroundingME-black?logo=github)](https://github.com/lirang04/GroundingME)
[![GroundingME](https://img.shields.io/badge/🤗-GroundingME-yellow)](https://huggingface.co/datasets/lirang04/GroundingME)
[![GroundingME](https://img.shields.io/badge/🤗-RefCOCOg_rej-yellow)](https://huggingface.co/datasets/lirang04/RefCOCOg_rej)
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-blue)](https://groundingme.github.io)

</div>

## 🔍 Overview

Visual grounding—localizing objects from natural language descriptions—represents a critical bridge between language and vision understanding. While multimodal large language models (MLLMs) achieve impressive scores on existing benchmarks, a fundamental question remains: **can MLLMs truly ground language in vision with human-like sophistication, or are they merely pattern-matching on simplified datasets?**

Current benchmarks fail to capture real-world complexity where humans effortlessly navigate ambiguous references and recognize when grounding is impossible. To rigorously assess MLLMs' true capabilities, we introduce **GroundingME**, a benchmark that systematically challenges models across four critical dimensions:

- 🎯 **Discriminative** — Distinguishing highly similar objects
- 📐 **Spatial** — Understanding complex relational descriptions  
- 🔬 **Limited** — Handling occlusions or tiny objects
- ❌ **Rejection** — Recognizing ungroundable queries

<p align="center">
  <img src="https://groundingme.github.io/images/examples.jpg" width="70%">
</p>


## 🚀 Quick Start

### Load Dataset from HuggingFace

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("lirang04/GroundingME", split="test")

# Access a sample
sample = dataset[0]
image = sample["image"]
description = sample["description"]
bbox = sample["bbox"]  # Ground truth [x1, y1, x2, y2]
category = sample["subtask_l1"]  # Discriminative/Spatial/Limited/Rejection
```

### Installation

```bash
# Clone the repository
git clone https://github.com/lirang04/GroundingME.git
cd GroundingME

# Install dependencies
pip install datasets pillow tqdm openai
```

### Evaluate Your Model

The evaluation script supports any OpenAI-compatible API:

```bash
# Local vLLM server
python evaluate.py \
  --api-url http://localhost:8000/v1 \
  --api-key dummy \
  --model-name Qwen/Qwen3-VL-8B-Thinking \
  --workers 16 \
  --output results.json
```

## 📈 Benchmark Statistics

<p align="center">
  <img src="https://groundingme.github.io/images/category.jpg" width="60%">
</p>

## 📜 License

This benchmark follows the licensing terms of [SA-1B](https://ai.meta.com/datasets/segment-anything/) and [HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench). **Research use only.**

## 📖 Citation

If you find GroundingME useful for your research, please cite our paper:

```bibtex
@article{li2025groundingme,
  title={GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation},
  author={Li, Rang and Li, Lei and Ren, Shuhuai and Tian, Hao and Gu, Shuhao and Li, Shicheng and Yue, Zihao and Wang, Yudong and Ma, Wenhan and Yang, Zhe and others},
  journal={arXiv preprint arXiv:2512.17495},
  year={2025}
}
```