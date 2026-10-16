#!/usr/bin/env python3
"""Poll the two gate-comparison jobs and surface only milestone lines.

Milestones are printed with an upper-case [MONITOR] prefix so they can be watched without
drowning in per-batch progress output. Once both jobs reach a terminal state the combined
comparison table is written and printed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent

JOBS = {
    "632981": "independent",
    "633055": "grouped",
}
TERMINAL = {"Succeeded", "Failed", "Stopped", "Killed", "Preempted"}

# Lines worth waking someone up for.
MILESTONE = re.compile(
    r"\[smoke\]|\[check\]|\[linear-reroute\]|\[mlp-score\]|\[complete\]|\[linear-gate\]"
    r"|\[FATAL|\[check\]\[FATAL\]|Traceback|CUDA out of memory|torch\.OutOfMemoryError"
    r"|Killing subprocess|ERROR"
)
# Lines that are useful context but far too frequent to notify on.
PROGRESS = re.compile(r"rank\d+: sft_train|加载 sft_train|dev 三层分层采样|\[extract\]|\[guard\]")

POLL_SECONDS = 60
MAX_HOURS = 8


def run(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=180, check=False
        ).stdout
    except subprocess.TimeoutExpired:
        return ""


def job_status(job_id: str) -> str:
    payload = run(["cctl", "pytorchjob", "get", job_id, "-F", "status"])
    match = re.search(r'"status"\s*:\s*"([^"]+)"', payload)
    return match.group(1) if match else "Unknown"


def job_messages(job_id: str) -> list[str]:
    messages = []
    for line in run(["cctl", "pytorchjob", "logs", job_id]).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            messages.append(json.loads(line).get("Message", ""))
        except json.JSONDecodeError:
            continue
    return messages


def main() -> int:
    seen: dict[str, set[str]] = {job_id: set() for job_id in JOBS}
    last_status: dict[str, str] = {job_id: "" for job_id in JOBS}
    deadline = time.time() + MAX_HOURS * 3600

    while time.time() < deadline:
        finished = 0
        for job_id, arm in JOBS.items():
            status = job_status(job_id)
            if status != last_status[job_id]:
                print(f"[MONITOR] {arm} ({job_id}) status={status}", flush=True)
                last_status[job_id] = status

            for message in job_messages(job_id):
                if message in seen[job_id]:
                    continue
                if MILESTONE.search(message):
                    seen[job_id].add(message)
                    print(f"[MONITOR] {arm}: {message}", flush=True)
                elif PROGRESS.search(message):
                    seen[job_id].add(message)
                    print(f"[monitor] {arm}: {message}", flush=True)

            if status in TERMINAL:
                finished += 1

        if finished == len(JOBS):
            print("[MONITOR] both jobs reached a terminal state", flush=True)
            break
        time.sleep(POLL_SECONDS)
    else:
        print(f"[MONITOR] giving up after {MAX_HOURS}h", flush=True)

    summary = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "summarize.py"),
            "--runs-root",
            str(EXP_ROOT / "runs"),
            "--output",
            str(EXP_ROOT / "summary" / "gate_comparison.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(summary.stdout, flush=True)
    if summary.stderr.strip():
        print(summary.stderr, flush=True)
    print("[MONITOR] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
