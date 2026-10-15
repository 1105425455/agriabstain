#!/usr/bin/env python
"""Resolve relative image paths in benchmark JSONL files to absolute paths.

For each of the three fixed datasets (comprehensive sft_test 2646, refcocog 362,
plantwild closed12 408), resolves all ``images`` entries to absolute paths:

* ``data/source/…`` entries → ``<benchmark-root>/data/source/…``
* ``data/ood/…`` entries    → ``<benchmark-root>/data/ood/…``
* ``data/derived/…`` entries → ``<derived-root>/…``  (stripping the ``data/derived`` prefix)

Output is written atomically to ``<output-root>/inputs/<name>_resolved.jsonl``.
A companion ``manifest.json`` tracks source file hashes, record counts, and
resolution roots so that a subsequent run with identical inputs is a no-op.

Usage
-----
::

    # default: resolve all three sets
    python resolve_inputs.py --output-root /scratch/out

    # single set
    python resolve_inputs.py --output-root /scratch/out --only comprehensive

    # custom paths
    python resolve_inputs.py \\
        --benchmark-root /data/aaai_benchmark \\
        --derived-root /data/derived_images \\
        --output-root /scratch/out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Fixed data definitions  (do not change)
# ---------------------------------------------------------------------------

DATASETS = {
    "comprehensive": {
        "rel_path": "annotations/sft/sft_test.jsonl",
        "expected_lines": 2646,
        "source_type": "sft",  # images under data/source or data/derived
    },
    "refcocog": {
        "rel_path": "annotations/eval/ood_refcocog.jsonl",
        "expected_lines": 362,
        "source_type": "ood",
    },
    "plantwild": {
        "rel_path": "annotations/eval/plantwild_real_closed12_paired_408.jsonl",
        "expected_lines": 408,
        "source_type": "ood",
    },
}

PLANTWILDFILENAME = "plantwild_real_closed12_paired_408.jsonl"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _resolve_one_record(
    record: dict[str, Any],
    benchmark_root: Path,
    derived_root: Path,
) -> dict[str, Any]:
    """Resolve image paths: first try benchmark_root, fall back to derived_root for derived images."""
    resolved = dict(record)
    images: list[str] = []
    for raw_path in record.get("images", []):
        p = Path(raw_path)
        if p.is_absolute():
            candidate = p
        else:
            candidate = benchmark_root / p
            if not candidate.is_file() and len(p.parts) >= 3 and p.parts[:2] == ("data", "derived"):
                # Fallback to external derived root
                candidate = derived_root / Path(*p.parts[2:])
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Cannot resolve image '{raw_path}': {candidate} does not exist"
            )
        images.append(str(candidate.resolve()))
    resolved["images"] = images
    return resolved


def resolve_dataset(
    name: str,
    benchmark_root: Path,
    derived_root: Path,
    output_root: Path,
    force: bool = False,
) -> Path:
    """Resolve one dataset, write atomically, return output path."""
    meta = DATASETS[name]
    src = benchmark_root / meta["rel_path"]
    out = output_root / "inputs" / f"{name}_resolved.jsonl"
    manifest_path = output_root / "inputs" / "manifest.json"

    # ---- pre-flight validation ------------------------------------------------
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")

    # line count
    with src.open(encoding="utf-8") as fh:
        src_lines = sum(1 for _ in fh)
    if src_lines != meta["expected_lines"]:
        raise ValueError(f"{src.name}: expected {meta['expected_lines']} lines, got {src_lines}")

    # PlantWild filename check
    if name == "plantwild" and src.name != PLANTWILDFILENAME:
        raise ValueError(f"PlantWild file must be named '{PLANTWILDFILENAME}', got '{src.name}'")

    # sample_id uniqueness
    sample_ids: set[str] = set()
    duplicates: list[str] = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = (rec.get("metadata") or {}).get("sample_id")
            if sid is None:
                raise ValueError(f"Missing sample_id in {src.name} record")
            if sid in sample_ids:
                duplicates.append(sid)
            sample_ids.add(sid)
    if duplicates:
        raise ValueError(f"{src.name}: {len(duplicates)} duplicate sample_ids: {duplicates[:10]}")

    src_hash = _sha256_file(src)

    # ---- fingerprint / reuse -------------------------------------------------
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ds_entry = manifest.get(name)
    if (
        not force
        and ds_entry is not None
        and ds_entry.get("sha256") == src_hash
        and ds_entry.get("count") == src_lines
        and ds_entry.get("benchmark_root") == str(benchmark_root.resolve())
        and ds_entry.get("derived_root") == str(derived_root.resolve())
        and out.is_file()
    ):
        # Count resolved lines too
        with out.open(encoding="utf-8") as fh:
            resolved_count = sum(1 for _ in fh)
        if resolved_count == src_lines:
            print(f"[resolve_inputs] Reusing existing {name} -> {out}  (fingerprint match)")
            return out

    # ---- resolve --------------------------------------------------------------
    print(f"[resolve_inputs] Resolving {name} ({src_lines} records) -> {out}")
    tmpf = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        dir=str(out.parent),
        delete=False,
    )
    try:
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                resolved = _resolve_one_record(rec, benchmark_root, derived_root)
                tmpf.write(json.dumps(resolved, ensure_ascii=False) + "\n")
        tmpf.close()
        # atomic replace
        os.replace(tmpf.name, str(out))
    except BaseException:
        if Path(tmpf.name).is_file():
            Path(tmpf.name).unlink()
        raise

    # ---- update manifest -----------------------------------------------------
    manifest[name] = {
        "sha256": src_hash,
        "count": src_lines,
        "benchmark_root": str(benchmark_root.resolve()),
        "derived_root": str(derived_root.resolve()),
        "resolved_path": str(out.resolve()),
    }
    # write manifest atomically
    m_tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=str(manifest_path.parent),
        delete=False,
    )
    try:
        json.dump(manifest, m_tmp, indent=2, ensure_ascii=False)
        m_tmp.close()
        os.replace(m_tmp.name, str(manifest_path))
    except BaseException:
        if Path(m_tmp.name).is_file():
            Path(m_tmp.name).unlink()
        raise

    print(f"[resolve_inputs] Done  {name} -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help="Path to aaai_code/benchmark  (default: auto-detect relative to this script)",
    )
    ap.add_argument(
        "--derived-root",
        type=Path,
        default=None,
        help="Root for derived (transformed) images  (default: DERIVED_ROOT env or "
        "new_reason/outputs/benchmarks/transformed_images)",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root output directory (must be outside aaai_code)",
    )
    ap.add_argument(
        "--only",
        choices=list(DATASETS),
        default=None,
        help="Resolve only one dataset (default: all three)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-resolve even if fingerprints match",
    )
    args = ap.parse_args()

    # ---- auto-detect benchmark root -------------------------------------------
    if args.benchmark_root is None:
        # script -> experiments/base_native_regate -> experiments -> aaai_code -> benchmark
        args.benchmark_root = Path(__file__).resolve().parent.parent.parent / "benchmark"
    bm = args.benchmark_root.resolve()
    if not bm.is_dir():
        ap.error(f"Benchmark root not found: {bm}")

    # ---- derived root ---------------------------------------------------------
    if args.derived_root is None:
        derived_env = os.environ.get("DERIVED_ROOT")
        if derived_env:
            args.derived_root = Path(derived_env)
        else:
            # fallback: new_reason/outputs/benchmarks/transformed_images relative to repo root
            repo_candidate = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "new_reason"
                / "outputs"
                / "benchmarks"
                / "transformed_images"
            )
            if repo_candidate.is_dir():
                args.derived_root = repo_candidate
            else:
                ap.error(
                    "Cannot locate derived images root. Set --derived-root or DERIVED_ROOT env."
                )
    dr = args.derived_root.resolve()

    # ---- output root validation -----------------------------------------------
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Must be outside aaai_code
    aaai_code = Path(__file__).resolve().parent.parent.parent  # experiments/.. = aaai_code
    try:
        out.relative_to(aaai_code)
        ap.error(
            f"OUTPUT_ROOT ({out}) is inside aaai_code ({aaai_code}). "
            "It must be an external directory (e.g. /scratch/...)."
        )
    except ValueError:
        pass  # good — not inside aaai_code

    # ---- resolve --------------------------------------------------------------
    names: list[str] = [args.only] if args.only else list(DATASETS)

    # Ensure the inputs subdirectory exists before tempfile writing
    (out / "inputs").mkdir(parents=True, exist_ok=True)

    for n in names:
        resolve_dataset(n, bm, dr, out, force=args.force)

    print("[resolve_inputs] All datasets resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
