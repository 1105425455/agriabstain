#!/usr/bin/env python3
"""Run an AgriAbstain evaluation split through the GPT-5.6 Sol API."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


DEFAULT_ENDPOINT = os.environ.get("GPT56_ENDPOINT", "")
DEFAULT_MODEL = "gpt-5.6-sol"
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_THREAD_LOCAL = threading.local()
REPO = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = REPO / "aaai_code" / "benchmark"
DECISION_SYSTEM_PROMPT = (
    "You are AgriAbstain, a vision-language grounding assistant for high-stakes plant disease diagnosis. "
    "Locate the queried target. If the queried target is present and localizable, return "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]} with box in [0,1000] relative coords. '
    "Otherwise abstain: "
    '{"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object: the image shows a visually similar but different object/disease, not the queried one. "
    "no_target: the queried target is not locatable (absent from the image, or cropped out of view)."
)


def _load_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records[:limit] if limit is not None else records


def _sample_id(record: dict[str, Any]) -> str:
    sample_id = record.get("metadata", {}).get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("every evaluation record must have metadata.sample_id")
    return sample_id


def _keyed_records(
    records: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    occurrences: dict[str, int] = {}
    keyed: list[tuple[str, dict[str, Any]]] = []
    for record in records:
        sample_id = _sample_id(record)
        occurrence = occurrences.get(sample_id, 0)
        occurrences[sample_id] = occurrence + 1
        keyed.append((f"{sample_id}::occurrence={occurrence}", record))
    return keyed


def _session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def _image_data_uri(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}", hashlib.sha256(data).hexdigest()


def _user_prompt(record: dict[str, Any]) -> str:
    user = ""
    for message in record["messages"]:
        if message["role"] == "user":
            user = str(message["content"]).replace("<image>", "").strip()
    if not user:
        raise ValueError(f"{_sample_id(record)}: missing user prompt")
    return user


def _request_payload(
    record: dict[str, Any],
    model: str,
    image_detail: str,
    max_completion_tokens: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    images = record.get("images") or []
    if len(images) != 1:
        raise ValueError(f"{_sample_id(record)}: expected exactly one image")
    image_path = Path(images[0])
    if not image_path.is_absolute():
        image_path = BENCHMARK_ROOT / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    data_uri, image_sha256 = _image_data_uri(image_path)
    system = DECISION_SYSTEM_PROMPT
    user = _user_prompt(record)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                            "detail": image_detail,
                        },
                    },
                    {"type": "text", "text": user},
                ],
            },
        ],
        "stream": False,
        "temperature": 0,
        "max_completion_tokens": max_completion_tokens,
    }
    audit = {
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "user_sha256": hashlib.sha256(user.encode()).hexdigest(),
    }
    return payload, audit


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"response has no choices: {response}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ]
        if texts:
            return "".join(texts)
    raise ValueError(f"unsupported response content: {content!r}")


def _call_one(
    record_key: str,
    record: dict[str, Any],
    endpoint: str,
    api_key: str,
    model: str,
    image_detail: str,
    max_completion_tokens: int,
    timeout_seconds: float,
    max_attempts: int,
) -> dict[str, Any]:
    payload, audit = _request_payload(
        record,
        model,
        image_detail,
        max_completion_tokens,
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = _session().post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout_seconds,
            )
            if response.status_code in TRANSIENT_STATUS_CODES:
                raise requests.HTTPError(
                    f"transient HTTP {response.status_code}: {response.text[:300]}"
                )
            response.raise_for_status()
            body = response.json()
            raw = _response_text(body)
            return {
                "record_key": record_key,
                "sample_id": _sample_id(record),
                "relation": record.get("metadata", {}).get("relation"),
                "raw": raw,
                "request_audit": audit,
                "api_model": body.get("model"),
                "usage": body.get("usage"),
                "response_id": body.get("id"),
                "attempts": attempt,
            }
        except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == max_attempts:
                break
            delay = min(30.0, 2 ** (attempt - 1)) + random.random()
            time.sleep(delay)
    raise RuntimeError(f"{_sample_id(record)} failed after {max_attempts} attempts") from last_error


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(record["record_key"])
        for record in _load_jsonl(path, limit=None)
        if isinstance(record.get("record_key"), str)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="GPT-5.6-compatible endpoint; set GPT56_ENDPOINT or pass this option",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--image-detail", choices=("low", "high", "original"), default="original")
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.endpoint:
        raise RuntimeError("GPT endpoint is unset; pass --endpoint or set GPT56_ENDPOINT")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"API key environment variable is unset: {args.api_key_env}")
    records = _load_jsonl(args.data, args.limit)
    keyed_records = _keyed_records(records)
    sample_ids = [_sample_id(record) for record in records]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(args.output)
    pending = [
        (record_key, record)
        for record_key, record in keyed_records
        if record_key not in done
    ]
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []
    with args.output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _call_one,
                    record_key,
                    record,
                    args.endpoint,
                    api_key,
                    args.model,
                    args.image_detail,
                    args.max_completion_tokens,
                    args.timeout_seconds,
                    args.max_attempts,
                ): record_key
                for record_key, record in pending
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"GPT-5.6 Sol {args.data.stem}",
                unit="sample",
            ):
                record_key = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001
                    failures.append((record_key, repr(error)))
                    continue
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()

    final_keys = _completed_keys(args.output)
    expected_keys = {record_key for record_key, _record in keyed_records}
    missing = sorted(expected_keys - final_keys)
    summary = {
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "requested_records": len(records),
        "completed_records": len(expected_keys & final_keys),
        "missing_records": len(missing),
        "duplicate_sample_id_records": len(sample_ids) - len(set(sample_ids)),
        "model_requested": args.model,
        "endpoint": args.endpoint,
        "image_detail": args.image_detail,
        "max_completion_tokens": args.max_completion_tokens,
        "workers": args.workers,
        "temperature": 0,
        "stream": False,
        "prompt_policy": (
            "Exact ReGate decision system override and per-record user text; "
            "no JSON mode or repair call."
        ),
        "system_sha256": hashlib.sha256(DECISION_SYSTEM_PROMPT.encode()).hexdigest(),
        "key_persisted": False,
        "failures": failures[:20],
        "missing_sample_ids": missing[:20],
    }
    summary_path = args.output.with_suffix(".run_config.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
