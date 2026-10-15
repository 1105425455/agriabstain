#!/usr/bin/env python3
"""Baseline runner: 纯LM + prompt/cot variant 跑对比实验。
避免命令行转义地狱：从 prompts.py 读 prompt，逐 variant 逐探针调 eval 脚本。
用法（8卡 torchrun 由本脚本内部 subprocess 调 eval）：
  python run_baselines.py --variant prompt_reward_norms --probes L1_discrim
输出隔离在 new_reason/baselines/prompt_cot/outputs/<variant>/<probe>/
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
EVAL = f"{REPO}/aaai_code/evaluation/eval_qwen3vl_3reason.py"
CKPT = f"{REPO}/new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444"
BASE = os.environ.get("BASE_MODEL", "/user/huangyongbin/models/modelscope/Qwen/Qwen3-VL-8B-Instruct")
OUT_ROOT = f"{REPO}/new_reason/baselines/prompt_cot/outputs"

sys.path.insert(0, f"{REPO}/new_reason/baselines/prompt_cot/configs")
from prompts import VARIANTS  # noqa

PROBE_PATHS = {
    "L1_discrim": "new_reason/outputs/eval_probes/test_L1_discrim.jsonl",
    "L2_evidence": "new_reason/outputs/eval_probes/test_L2_evidence.jsonl",
    "L3_dose": "new_reason/outputs/eval_probes/test_L3_dose.jsonl",
    "test_full": "new_reason/outputs/sft_3reason/sft_test.jsonl",
    "ood_refcocog": "new_reason/outputs/eval_probes/ood_refcocog.jsonl",
    "plantwild_real_v1": "new_reason/outputs/eval_probes/plantwild_real_v1/plantwild_real_paired_444.jsonl",
    "plantwild_real_closed12_v1": "aaai_code/benchmark/annotations/eval/plantwild_real_closed12_paired_408.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    # native_lm: 纯LM对照，用与判别头完全相同的 DECISION_SYSTEM_PROMPT(--system-override decision)，
    # 不加任何额外 prompt 工程，唯一变量 = decision 来自 LM 生成 vs 判别头。这是最干净的"架构 vs 架构"消融。
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()) + ["native_lm"])
    ap.add_argument("--probes", nargs="+", default=["L1_discrim"])
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前N条(冒烟测试)")
    args = ap.parse_args()

    # native_lm 用 "decision"(触发脚本内 DECISION_SYSTEM_PROMPT，与判别头同一 prompt)；
    # 其余 variant 用 prompts.py 里定义的额外 prompt。
    prompt = "decision" if args.variant == "native_lm" else VARIANTS[args.variant]
    # prompt 写临时文件，避免命令行转义；eval 的 --system-override 接受字符串，
    # 但长prompt用文件更稳：这里直接把 prompt 作为 --system-override 值传，
    # subprocess 的 list 形式不经过 shell，无转义问题。
    probe_list = ",".join(PROBE_PATHS[p] for p in args.probes)
    tags = ",".join(args.probes)
    out_dirs = ",".join(f"{OUT_ROOT}/{args.variant}/{p}" for p in args.probes)
    for p in args.probes:
        os.makedirs(f"{OUT_ROOT}/{args.variant}/{p}", exist_ok=True)

    cmd = [
        "torchrun", f"--nproc_per_node={args.nproc}",
        "--master_port", str(20000 + os.getpid() % 9999),
        EVAL,
        "--data", probe_list,
        "--base", BASE,
        "--adapter", CKPT,
        "--out", out_dirs,
        "--tag", tags,
        "--backend", "hf",
        "--system-override", prompt,   # 纯LM：不加 --dual-head！靠 prompt 生成 decision
        "--max-new-tokens", "384" if args.variant == "native_lm" else "512",  # CoT 需更长；native_lm 与判别头一致用 384
        "--max-pixels", "262144",
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]

    print(f"[runner] variant={args.variant} probes={args.probes}", flush=True)
    print(f"[runner] 纯LM模式(无--dual-head), prompt前80字: {prompt[:80]}...", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}/aaai_code/training:{REPO}/aaai_code:" + env.get("PYTHONPATH", "")
    r = subprocess.run(cmd, env=env, cwd=REPO)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
