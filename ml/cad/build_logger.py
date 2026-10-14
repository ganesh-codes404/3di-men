"""Structured, append-only JSONL logging for /api/build requests --
server.py had NO logging at all before this (confirmed directly: no
`log`/`logger` reference anywhere in it), which is exactly why a real
production failure (a bad treasure-chest generation) couldn't be traced
back after the fact -- there was no record of the prompt, the LLM's raw
output, the per-point predictions, or where things went wrong.

One line per /api/build call, written to logs/build_log.jsonl (gitignored,
local-only, regenerable -- never committed). Deliberately NOT a Python
logging.Logger: this project has no other logging infrastructure to
integrate with, and a single append-only JSONL writer is simpler to grep,
`jq`, and diff than configuring a logger for one call site. No secrets or
unrelated user data recorded -- just the build request/response shape
already visible in the HTTP response, plus the LLM's raw output and each
point's predicted program (not currently returned to the frontend at all),
which is the missing piece for tracing WHERE a bad generation went wrong.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_PATH = LOG_DIR / "build_log.jsonl"


def log_build(
    prompt: str,
    raw_llm_output: str | None,
    fell_back: bool,
    points: list[dict[str, Any]],
    elapsed_seconds: float,
    error: str | None = None,
) -> None:
    """points: one dict per decomposed point, each with at least
    `description`, `position_mm`, `status`, `program` (op/args list from
    the predictor, the actual thing that determines what gets built -- the
    earlier-missing piece), and `error`."""
    LOG_DIR.mkdir(exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "prompt": prompt,
        "raw_llm_output": raw_llm_output,
        "fell_back_to_single_point": fell_back,
        "points": points,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "error": error,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_recent(n: int = 20) -> list[dict[str, Any]]:
    """For diagnosis -- python3 -c "from build_logger import read_recent; ..." """
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().splitlines()
    return [json.loads(line) for line in lines[-n:]]
