"""在clone与git lfs pull后恢复工作区关联，不覆盖已有实体文件。"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link(destination: Path, source: Path, execute: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"缺少资产，请先git lfs pull: {source}")
    if source.is_file():
        with source.open("rb") as stream:
            if stream.read(80).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise ValueError(f"仅有LFS指针，请先git lfs pull: {source}")
    if destination.exists():
        print(f"KEEP {destination.relative_to(ROOT)}")
        return
    if destination.is_symlink():
        print(f"BROKEN {destination.relative_to(ROOT)} -> {destination.readlink()}")
        if execute:
            destination.unlink()
    print(f"LINK {destination.relative_to(ROOT)} -> {source}")
    if execute:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="默认只预览")
    parser.add_argument(
        "--public-data-root",
        type=Path,
        help="已下载并整理为source/ood结构的公共图片根目录",
    )
    args = parser.parse_args()
    archive = ROOT / "research_assets/datasets/derived_images.tar.gz"
    derived = ROOT / "aaai_code/benchmark/data/derived"
    marker = derived / ".regate-extracted-complete"
    if not archive.exists():
        raise FileNotFoundError(f"缺少派生图归档: {archive}")
    if not marker.exists() and not derived.is_symlink():
        print(f"EXTRACT {archive.relative_to(ROOT)}")
        if args.execute:
            target = derived.parent.resolve()
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as bundle:
                for entry in bundle.getmembers():
                    resolved = (target / entry.name).resolve()
                    if (
                        not resolved.is_relative_to(target)
                        or entry.issym()
                        or entry.islnk()
                        or not (entry.isfile() or entry.isdir())
                    ):
                        raise ValueError(f"不安全归档条目: {entry.name}")
                bundle.extractall(target)
            marker.write_text("derived_images.tar.gz extracted successfully\n")
    pairs = {
        ROOT / "new_reason/outputs/benchmarks/transformed_images": ROOT
        / "aaai_code/benchmark/data/derived",
        ROOT / "final_story/outputs/discrim_musd.pt": ROOT
        / "research_assets/discrim_musd.pt",
    }
    for name in ("papermodel_lineargate", "h27experiment", "devtau_multiseed"):
        pairs[ROOT / name] = ROOT / "research_assets" / name
    if args.public_data_root:
        public = args.public_data_root.resolve()
        for name in ("source/coco/images", "ood/refcocog/images"):
            pairs[ROOT / "aaai_code/benchmark/data" / name] = public / name
    for destination, source in pairs.items():
        link(destination, source, args.execute)
    for relative, source in {
        "source/plantdoc/images": ROOT / "research_assets/exact_images/plantdoc",
        "source/plantdoc/labels": ROOT / "research_assets/plantdoc_yolo_labels",
        "source/coco/annotations/instances_selected.json": ROOT
        / "research_assets/coco_annotations/instances_selected.json",
        "ood/plantwild_closed12/images": ROOT
        / "research_assets/exact_images/plantwild_closed12",
        "ood/refcocog/metadata": ROOT / "research_assets/refcocog_metadata",
    }.items():
        link(ROOT / "aaai_code/benchmark/data" / relative, source, args.execute)
    print("旧日志与历史配置仍含原绝对路径；运行前按资产说明设置模型和输出路径。")


if __name__ == "__main__":
    main()
