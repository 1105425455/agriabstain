"""检查受Git跟踪LFS对象的实体完整性，以及是否误纳入论文。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hash", action="store_true", help="读取全部大文件核验SHA-256，耗时较长"
    )
    args = parser.parse_args()
    listing = subprocess.check_output(
        ["git", "lfs", "ls-files", "--long"], cwd=ROOT, text=True
    )
    entries = [
        {"oid": line[:64], "name": line[67:]}
        for line in listing.splitlines()
        if len(line) >= 68
    ]
    errors = []
    for entry in entries:
        p = ROOT / entry["name"]
        if not p.is_file() or p.is_symlink():
            errors.append({"path": entry["name"], "error": "missing or symlink"})
            continue
        with p.open("rb") as stream:
            prefix = stream.read(120)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            errors.append(
                {"path": entry["name"], "error": "pointer only; run git lfs pull"}
            )
            continue
        if args.hash:
            digest = hashlib.sha256()
            with p.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["oid"]:
                errors.append({"path": entry["name"], "error": "sha256 mismatch"})
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    excluded = [
        p
        for p in tracked
        if p.startswith(("paper/", "figure/", "submission_snapshot/"))
    ]
    final_manifest = ROOT / "docs/FINAL_ASSETS.json"
    missing_final = []
    unexpected_assets = []
    if final_manifest.exists():
        expected = json.loads(final_manifest.read_text())["files"]
        tracked_set = set(tracked)
        expected_set = {entry["path"] for entry in expected}
        binary_suffixes = {
            ".pt",
            ".pth",
            ".bin",
            ".safetensors",
            ".npz",
            ".npy",
            ".ckpt",
        }
        unexpected_assets = [
            name
            for name in tracked
            if name
            and (
                name.startswith("research_assets/")
                or Path(name).suffix in binary_suffixes
            )
            and name not in expected_set
        ]
        missing_final = [
            entry["path"] for entry in expected if entry["path"] not in tracked_set
        ]
        for entry in expected:
            path = ROOT / entry["path"]
            if not path.is_file() or path.is_symlink():
                errors.append(
                    {"path": entry["path"], "error": "manifest file missing or symlink"}
                )
            elif path.stat().st_size != entry["bytes"]:
                # Git may normalize CRLF in small text assets; LFS objects must retain exact size.
                if entry["path"] in {e["name"] for e in entries}:
                    errors.append(
                        {
                            "path": entry["path"],
                            "error": "LFS size differs from manifest",
                        }
                    )
    else:
        errors.append(
            {"path": "docs/FINAL_ASSETS.json", "error": "required manifest missing"}
        )
    print(
        json.dumps(
            {
                "lfs_files": len(entries),
                "errors": errors,
                "forbidden_tracked_paths": excluded,
                "missing_final_assets_in_git": missing_final,
                "unexpected_assets_outside_manifest": unexpected_assets,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(bool(errors or excluded or missing_final or unexpected_assets))


if __name__ == "__main__":
    main()
