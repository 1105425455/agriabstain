"""CPU评分固定缓存上的Stage1、发布Stage2和三seed refit heads。"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from sklearn.metrics import matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/user/huangyongbin/newrepo/agriabstain")
sys.path.insert(0, str(ROOT / "aaai_code"))
from experiments.selective_grounding.analyze_selective_grounding import (  # noqa: E402
    _candidate_prediction,  # noqa: E402
)
from fit_gates import build_mlp, load_shards  # noqa: E402
from reason_abstain.eval.metrics import (  # noqa: E402
    aggregate_metrics,
    metamorphic_test_satisfaction,
)

OUT = ROOT / "docs/review_evidence/cpu_gate"
CKPTS = {
    "grouped": SOURCE
    / "new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444",
    "independent": SOURCE
    / "aaai_code/experiments/grouping_ablation/artifacts/checkpoints/independent_negative_seed42/checkpoint-222",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(probs: np.ndarray, records: list[dict]) -> dict:
    y = np.asarray(
        [
            0
            if r["gt"]["decision"] == "answer"
            else 1
            if r["gt"]["reason"] == "wrong_object"
            else 2
            for r in records
        ]
    )
    selected = probs.argmax(1) == 0
    pairs = []
    for r, p, answer in zip(records, probs, selected, strict=True):
        pred = (
            _candidate_prediction(r)
            if answer
            else {
                "valid": True,
                "decision": "abstain",
                "reason": "wrong_object" if p[1] >= p[2] else "no_target",
            }
        )
        pairs.append((pred, {**r["gt"], "relation": r.get("relation")}))
    m = aggregate_metrics(pairs)
    return {
        "FCA": float(selected[y != 0].mean()),
        "retention": float(selected[y == 0].mean()),
        "binary_MCC": float(matthews_corrcoef(y == 0, selected)),
        "state_accuracy": float((probs.argmax(1) == y).mean()),
        "AUROC": float(roc_auc_score(y == 0, probs[:, 0])),
        "Strict3": m["semantic"]["Strict_3"],
        "E2E_mIoU": m["grounding_reason"]["E2E_mIoU"],
        "MTS_macro": metamorphic_test_satisfaction(pairs)["MTS_macro"],
    }


def main() -> None:
    torch.set_num_threads(4)
    rows, norms, provenance = [], {}, {}
    for arm, checkpoint in CKPTS.items():
        run = SOURCE / "papermodel_lineargate/runs" / arm
        prediction_path = run / "evaluation/regate/comprehensive/predictions.jsonl"
        records = [json.loads(x) for x in prediction_path.read_text().splitlines()]
        assert len(records) == 2646 and len({r["sample_id"] for r in records}) == 2646
        hidden = torch.tensor(
            np.asarray([r["head_hidden"] for r in records], dtype=np.float32)
        )
        original_path = (
            (
                ROOT
                / "new_reason/outputs/base_native_regate_20260720/regate/comprehensive/predictions.jsonl"
            )
            if arm == "grouped"
            else (
                ROOT
                / "aaai_code/experiments/grouping_ablation/artifacts/evaluation/regate/comprehensive/predictions.jsonl"
            )
        )
        original = {
            r["sample_id"]: r
            for r in map(json.loads, original_path.read_text().splitlines())
        }
        assert {r["sample_id"] for r in records} == original.keys()
        assert all(r["gt"] == original[r["sample_id"]]["gt"] for r in records)
        published = torch.load(
            checkpoint / "discrim_head_calibrated.pt",
            map_location="cpu",
            weights_only=True,
        )
        xtrain, ytrain = load_shards(run / "calibration", "calib_train")
        mu, sd = xtrain.mean(0), xtrain.std(0) + 1e-6
        norms[arm] = {
            name: {
                "max_abs": float(np.abs(arr - published[name].numpy()).max()),
                "rms": float(np.sqrt(np.mean((arr - published[name].numpy()) ** 2))),
            }
            for name, arr in [("mu", mu), ("sd", sd)]
        }
        provenance[arm] = {
            "predictions": str(prediction_path),
            "sha256": sha(prediction_path),
            "checkpoint": str(checkpoint),
            "adapter_sha256": sha(checkpoint / "adapter_model.safetensors"),
            "train_shape": list(xtrain.shape),
            "train_counts": np.bincount(ytrain.astype(int)).tolist(),
        }
        stage1 = {}
        with safe_open(
            checkpoint / "adapter_model.safetensors", framework="pt", device="cpu"
        ) as payload:
            for key in payload.keys():
                marker = ".discrim_head."
                if marker in key:
                    stage1[key.split(marker, 1)[1]] = payload.get_tensor(key).float()
        assert stage1["classifier.6.weight"].shape[0] == 3
        variants = [
            (
                "stage1_auxiliary",
                {
                    "head_state": {
                        k: v for k, v in stage1.items() if k.startswith("classifier.")
                    },
                    "mu": stage1["mu"],
                    "sd": stage1["sd"],
                },
            ),
            ("published_stage2", published),
        ]
        variants += [
            (
                f"refit_seed{seed}",
                torch.load(
                    OUT / f"{arm}_seed{seed}/mlp_ours.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
            )
            for seed in (7, 21, 42)
        ]
        for name, payload in variants:
            model = build_mlp()
            state = {
                k.removeprefix("classifier."): v.float()
                for k, v in payload["head_state"].items()
            }
            model.load_state_dict(state)
            model.eval()
            with torch.inference_mode():
                probs = torch.softmax(
                    model((hidden - payload["mu"].float()) / payload["sd"].float()),
                    dim=1,
                ).numpy()
            result = {"arm": arm, "head": name, **score(probs, records)}
            if name == "published_stage2":
                old = np.asarray(
                    [
                        [
                            original[r["sample_id"]][k]
                            for k in ("p_answer", "p_wo", "p_nt")
                        ]
                        for r in records
                    ]
                )
                result["max_probability_delta_vs_original"] = float(
                    np.max(np.abs(old - probs))
                )
                result["decision_disagreements_vs_original"] = int(
                    np.sum(old.argmax(1) != probs.argmax(1))
                )
            rows.append(result)
            print(json.dumps(result), flush=True)
            np.savez_compressed(
                OUT / f"{arm}_{name}_probabilities.npz",
                probabilities=probs,
                sample_id=np.asarray([r["sample_id"] for r in records]),
            )
        del hidden, records, xtrain
    summary = {}
    for arm in CKPTS:
        fits = [r for r in rows if r["arm"] == arm and r["head"].startswith("refit_")]
        summary[arm] = {
            key: {
                "mean": float(np.mean([r[key] for r in fits])),
                "sample_sd": float(np.std([r[key] for r in fits], ddof=1)),
            }
            for key in (
                "FCA",
                "retention",
                "binary_MCC",
                "state_accuracy",
                "AUROC",
                "Strict3",
                "E2E_mIoU",
                "MTS_macro",
            )
        }
    (OUT / "scored_heads.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "refit_summary": summary,
                "normalization_differences": norms,
                "provenance": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    with (OUT / "scored_heads.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
