from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def resolve_record_images(
    record: dict[str, Any], benchmark_root: Path, derived_root: Path
) -> dict[str, Any]:
    resolved = dict(record)
    images: list[str] = []
    for raw_path in record.get("images", []):
        path = Path(raw_path)
        if path.is_absolute():
            candidate = path
        else:
            candidate = benchmark_root / path
            if not candidate.is_file() and path.parts[:2] == ("data", "derived"):
                candidate = derived_root.joinpath(*path.parts[2:])
        if not candidate.is_file():
            raise FileNotFoundError(f"cannot resolve image: {raw_path}")
        images.append(str(candidate.resolve()))
    resolved["images"] = images
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        args.input.open(encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{args.input}:{line_number}: expected JSON object")
            resolved = resolve_record_images(record, args.benchmark_root, args.derived_root)
            destination.write(json.dumps(resolved, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"{args.input}: no records")
    print(f"resolved {count} records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
