"""Consolidated closed-form verification gate for the ReGate supplement package.

Every check either reports concrete findings or proves their absence.
Run from inside the unpacked package directory.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(".")
results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # a crashing check is a failing check
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name:52} {detail}", flush=True)
        return fn

    return deco


@check("1  RELEASE_MANIFEST 正向：每个记录文件哈希与大小一致")
def _():
    bad = []
    for e in json.load(open("RELEASE_MANIFEST.json"))["files"]:
        p = pathlib.Path(e["relative"])
        if not p.is_file():
            bad.append(f"缺失 {p}")
            continue
        b = p.read_bytes()
        if hashlib.sha256(b).hexdigest() != e["sha256"] or len(b) != e["size"]:
            bad.append(f"不符 {p}")
    return not bad, f"{len(bad)} 处问题" if bad else "全部一致"


@check("2  RELEASE_MANIFEST 反向：磁盘无清单外文件")
def _():
    m = {e["relative"] for e in json.load(open("RELEASE_MANIFEST.json"))["files"]}
    d = {
        str(p) for p in ROOT.rglob("*") if p.is_file() and str(p) != "RELEASE_MANIFEST.json"
    }
    extra, missing = sorted(d - m), sorted(m - d)
    return not (extra or missing), f"多余 {len(extra)}，缺失 {len(missing)}"


@check("3  benchmark 标注哈希与 path_audit")
def _():
    m = json.load(open("benchmark/MANIFEST.json"))
    pa = m["path_audit"]
    if pa["old_prefix_occurrences_total"] or pa["missing_image_refs_total"]:
        return False, "path_audit 非零"
    bad = [
        a["path"]
        for a in m["annotations"]
        if hashlib.sha256(open("benchmark/" + a["path"], "rb").read()).hexdigest() != a["sha256"]
    ]
    return not bad, f"{len(m['annotations'])} 个标注全部一致" if not bad else f"{len(bad)} 处不符"


@check("4  匿名性：姓名/项目名/环境名/私有路径/RL 残留")
def _():
    pats = [
        "huangyongbin",
        "agriabstain",
        "AgriAbstain",
        "agrisam3",
        "agrivllm",
        "/user/",
        "GRPO",
        r"\bRL\b",
        r"[\w.+-]+@[\w-]+\.(?:com|edu|cn|org)",
    ]
    hits = []
    for pat in pats:
        r = subprocess.run(["grep", "-rIlE", pat, "."], capture_output=True, text=True)
        files = [f for f in r.stdout.split() if f]
        if files:
            hits.append(f"{pat}:{len(files)}")
    return not hits, "无命中" if not hits else "; ".join(hits)


@check("5  清洁度：缓存/空文件/空目录")
def _():
    cache = [p for p in ROOT.rglob("*") if p.is_dir() and p.name in {"__pycache__", ".pytest_cache", ".ruff_cache"}]
    pyc = list(ROOT.rglob("*.pyc"))
    empty_f = [p for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size == 0 and p.name != "__init__.py"]
    empty_d = [p for p in ROOT.rglob("*") if p.is_dir() and not any(p.iterdir())]
    n = len(cache) + len(pyc) + len(empty_f) + len(empty_d)
    return n == 0, f"缓存{len(cache)} pyc{len(pyc)} 空文件{len(empty_f)} 空目录{len(empty_d)}"


@check("6  无未声明的字节级重复内容")
def _():
    # Table 5 的 native-LM 分支复用 Table 3 的 SFT-native 运行，已在 REVIEWER_README §8 声明
    def documented(group: list[str]) -> bool:
        return len(group) == 2 and any("prompt_interfaces/predictions/native_lm" in g for g in group) \
            and any("base_native_regate/predictions/native/sft" in g for g in group)

    h = collections.defaultdict(list)
    for p in ROOT.rglob("*"):
        if p.is_file() and p.name not in {"__init__.py", "RELEASE_MANIFEST.json"}:
            h[hashlib.sha256(p.read_bytes()).hexdigest()].append(str(p))
    dups = [v for v in h.values() if len(v) > 1 and not documented(v)]
    return not dups, f"{len(dups)} 组未声明重复" if dups else "无未声明重复（3 组已声明复用）"


@check("7  无孤儿产物文件（每个产物都有 md 覆盖）")
def _():
    md = "\n".join(p.read_text(encoding="utf-8") for p in ROOT.rglob("*.md"))
    ext = {".json", ".jsonl", ".csv", ".log", ".pdf", ".png", ".txt", ".yaml"}
    orph = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in ext or "RELEASE_MANIFEST" in str(p):
            continue
        s = str(p)
        if s in md or p.name in md:
            continue
        if any(str(par) in md for par in p.parents if str(par) != "."):
            continue
        orph.append(s)
    return not orph, f"{len(orph)} 个孤儿" if orph else "全部有文档覆盖"


@check("8  .md 中无失效路径引用")
def _():
    not_shipped = (
        "discrim_musd.pt", "zref_cache.pt", "instances_train2017.json", "instances_selected.json",
        "hidden_train.npz", "hidden_holdout.npz", "val-00000-of-00003.parquet",
        "checkpoint-444", "plantwild.zip",
    )
    bad = []
    for md in ROOT.rglob("*.md"):
        txt = md.read_text(encoding="utf-8")
        for m in re.findall(r"`([A-Za-z0-9_./<>-]+\.(?:py|sh|json|jsonl|csv|md|log|pdf|png|yaml|txt))`", txt):
            if "<" in m or any(k in m for k in not_shipped):
                continue
            if pathlib.Path(m).exists() or (md.parent / m).exists():
                continue
            if list(ROOT.rglob(pathlib.Path(m).name)):
                continue
            bad.append(f"{md}:{m}")
    return not bad, f"{len(bad)} 处断链" if bad else "全部可达"


@check("9  图像引用 100% 被源图清单或派生图重建覆盖")
def _():
    mani = {i["path"] for i in json.load(open("benchmark/IMAGE_MANIFEST.json"))}

    def paths(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in {"image", "image_path", "images"}:
                    if isinstance(v, str):
                        yield v
                    elif isinstance(v, list):
                        yield from (x for x in v if isinstance(x, str))
                else:
                    yield from paths(v)
        elif isinstance(o, list):
            for x in o:
                yield from paths(x)

    total = uncovered = 0
    for f in ROOT.glob("benchmark/annotations/**/*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                for p in paths(json.loads(line)):
                    total += 1
                    if "derived" not in p and p.lstrip("./") not in mani:
                        uncovered += 1
    return uncovered == 0, f"{total:,} 条引用，{uncovered} 条未覆盖"


@check("10 requirements 覆盖代码全部三方 import")
def _():
    declared = {
        "numpy", "sklearn", "scipy", "yaml", "PIL", "tqdm", "matplotlib", "pyarrow", "torch",
        "transformers", "peft", "accelerate", "deepspeed", "safetensors", "qwen_vl_utils",
        "datasets", "pytest", "flash_attn", "vllm", "requests",
    }
    local_pkgs = {"reason_abstain", "experiments", "scripts", "training", "evaluation", "qwenvl", "benchmark"}
    sibling = {p.stem for p in ROOT.rglob("*.py")}
    missing = set()
    for p in ROOT.rglob("*.py"):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods = [n.module.split(".")[0]]
            for m in mods:
                if m in sys.stdlib_module_names or m in local_pkgs or m in declared or m in sibling:
                    continue
                missing.add(m)
    return not missing, f"未声明: {sorted(missing)}" if missing else "全部覆盖"


@check("11 环境记录与训练时日志交叉一致")
def _():
    env = json.load(open("environment/training_environment.json"))
    log = pathlib.Path("repro_artifacts/logs/grouping/environment.log").read_text(encoding="utf-8")
    want = {
        "python=3.11.15": env["interpreter"]["python_version"] == "3.11.15",
        "torch=2.8.0+cu128": env["packages"]["torch"] == "2.8.0",
        "transformers=4.57.0": env["packages"]["transformers"] == "4.57.0",
        "cuda=12.8": str(env["accelerator"]["torch_cuda_build"]) == "12.8",
    }
    bad = [k for k, v in want.items() if not (v and k in log)]
    gpus = log.count("NVIDIA GeForce RTX 4090")
    if gpus != env["hardware"]["gpu_count"]:
        bad.append(f"GPU 数 {gpus}!={env['hardware']['gpu_count']}")
    return not bad, "4 字段 + 8 卡全部一致" if not bad else str(bad)


@check("12 全部 Python 文件可编译")
def _():
    bad = []
    for p in ROOT.rglob("*.py"):
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except SyntaxError as e:
            bad.append(f"{p}:{e.lineno}")
    return not bad, f"{len(bad)} 处语法错误" if bad else f"{len(list(ROOT.rglob('*.py')))} 个文件全部通过"


@check("13 无中文残留（md / json / py / sh）")
def _():
    cjk = re.compile(r"[\u4e00-\u9fff]")
    bad = []
    for suf in ("*.md", "*.json", "*.py", "*.sh", "*.txt"):
        for p in ROOT.rglob(suf):
            if "RELEASE_MANIFEST" in str(p):
                continue
            try:
                if cjk.search(p.read_text(encoding="utf-8")):
                    bad.append(str(p))
            except Exception:
                pass
    return not bad, f"{len(bad)} 个文件含中文: {bad[:3]}" if bad else "无"


@check("14 §6 环境变量名全部存在于对应启动脚本")
def _():
    cases = [
        (["MUSD_OUTPUT", "BASE_MODEL"], "training/sft_final/compute_musd.py"),
        (["BASE_MODEL", "DISCRIM_MUSD_PATH", "SFT_OUTPUT_DIR"], "training/sft_final/run_train.sh"),
        (["BASE_MODEL", "CALIB_ADAPTER", "CALIB_LOG_DIR"], "training/sft_final/calibrate_head.py"),
        (["BASE_MODEL", "ADAPTER", "CALIBRATED_HEAD", "OUTPUT_DIR"], "evaluation/run_sft_final.sh"),
        (["BASE_MODEL", "ZREF_ADAPTER", "ZREF_CACHE_PATH"], "training/margin_kl/dump_zref.py"),
        (["SFT_INIT_CKPT", "ZREF_CACHE_PATH", "OUTPUT_DIR"], "training/margin_kl/run_v1.sh"),
        (["SFT_INIT_CKPT", "ZREF_CACHE_PATH", "OUTPUT_DIR"], "training/margin_kl/run_v2.sh"),
        (["BASE_MODEL", "CKPT_DIR"], "training/margin_kl/eval_closed12.sh"),
    ]
    bad = []
    for vs, s in cases:
        t = pathlib.Path(s).read_text(encoding="utf-8")
        bad += [f"{s}:{v}" for v in vs if v not in t]
    return not bad, f"{len(bad)} 个缺失" if bad else "全部匹配"


@check("15 论文数字复算：Table 3 ReGate 行")
def _():
    sup = uns = a_s = a_u = 0
    for line in open("repro_artifacts/base_native_regate/predictions/regate/comprehensive/predictions.jsonl"):
        r = json.loads(line)
        ans = r["p_answer"] >= max(r["p_wo"], r["p_nt"])
        if r["gt"]["decision"] == "answer":
            sup += 1
            a_s += ans
        else:
            uns += 1
            a_u += ans
    ret, fca = round(a_s / sup, 3), round(a_u / uns, 3)
    return (ret, fca) == (0.976, 0.132), f"retention={ret} FCA={fca} (论文 0.976 / 0.132)"


@check("16 论文数字复算：margin+KL 两行共 14 个数")
def _():
    want = {
        "v1_checkpoint-50": (0.155, 0.257, 0.584, 0.740, 0.708, 0.500, 0.972),
        "v2_checkpoint-60": (0.174, 0.222, 0.592, 0.733, 0.722, 0.458, 0.972),
    }
    bad = 0
    for run, w in want.items():
        m = json.load(open(f"experiments/margin_kl/main/{run}/plantwild_real_closed12_v1/metrics.json"))
        got = (
            m["metrics"]["headline"]["FCA"], m["metrics"]["safeguard"]["A_FU"],
            m["metrics"]["safeguard"]["MCC"], m["mts"]["MTS_micro"],
            m["mts"]["per_relation"]["OOD_answer"], m["mts"]["per_relation"]["OOD_confusion"],
            m["mts"]["per_relation"]["OOD_absent"],
        )
        bad += sum(abs(round(g, 3) - x) > 1e-9 for g, x in zip(got, w))
    return bad == 0, f"{bad}/14 不符"


@check("17 论文数字复算：evidence judge L2/L3 共 13 个数")
def _():
    l2 = json.load(open("experiments/evidence_judge/main/L2_test/analysis.json"))["L2"]
    l3 = json.load(open("experiments/evidence_judge/main/L3_test/analysis.json"))["L3"]
    chk = [
        (l2["delta_p_answer_mean"], 0.786), (l2["bootstrap_95ci"]["low"], 0.720),
        (l2["bootstrap_95ci"]["high"], 0.840), (l2["correct_direction_ratio"], 0.966),
        (l2["n_pairs"], 148), (l2["incomplete_pairs"], 249), (l3["n_groups"], 183),
        (l3["mean_spearman_rho"], 0.750), (l3["mean_kendall_tau"], 0.682),
        (l3["mean_adjacent_violation_rate"], 0.214),
        (l3["spearman_rho_bootstrap_95ci"]["low"], 0.694),
        (l3["kendall_tau_bootstrap_95ci"]["low"], 0.625),
        (l3["violation_rate_bootstrap_95ci"]["low"], 0.183),
    ]
    bad = sum(
        (abs(round(g, 3) - w) > 1e-9) if isinstance(g, float) else (g != w) for g, w in chk
    )
    return bad == 0, f"{bad}/13 不符"


@check("18 论文数字复算：matched-retention 操作点")
def _():
    r = json.load(open("repro_artifacts/matched_retention_dev/analysis/result.json"))
    s = json.dumps(r)
    need = ["0.9888232350349426", "0.8604651162790697", "0.8510959939531368", "0.889644746787604"]
    bad = [n for n in need if n not in s]
    return not bad, f"缺 {bad}" if bad else "阈值与 4 个保留率全部一致"


print()
n_fail = sum(1 for _, ok, _ in results if not ok)
print("=" * 78)
print(f"闭式检查 {len(results)} 项：通过 {len(results) - n_fail}，失败 {n_fail}")
print("=" * 78)
sys.exit(1 if n_fail else 0)
