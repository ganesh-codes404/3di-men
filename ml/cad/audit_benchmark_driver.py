"""Part 8 of the audit -- runs every prompt in audit_benchmark_prompts.py
through audit_benchmark_worker.py, ONE FRESH SUBPROCESS AT A TIME (never
concurrently -- this project's own standing rule), recording a typed
outcome for every prompt even if its own worker process crashes natively
or times out. Never raises itself; a crashed/timed-out prompt is recorded
and the driver moves on to the next one, so one bad prompt never loses
the rest of the batch.

New audit-only artifact -- run directly (`python3 audit_benchmark_driver.py`),
never imported by production code.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "audit_results" / "full_llm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DRIVER_LOG = ROOT / "audit_results" / "full_llm_driver_log.json"
PER_PROMPT_TIMEOUT_SECONDS = 420


def main() -> None:
    from audit_benchmark_prompts import PROMPTS

    log: list[dict] = []
    if DRIVER_LOG.exists():
        log = json.loads(DRIVER_LOG.read_text())
    done_ids = {entry["id"] for entry in log}

    for bp in PROMPTS:
        if bp.id in done_ids:
            print(f"skip {bp.id} (already in driver log)")
            continue
        result_path = OUT_DIR / f"{bp.id}.json"
        if result_path.exists():
            print(f"skip {bp.id} (result file already exists)")
            log.append({"id": bp.id, "outcome": "already_present"})
            DRIVER_LOG.write_text(json.dumps(log, indent=2))
            continue

        print(f"=== running {bp.id} ({bp.category}): {bp.prompt[:80]!r} ===", flush=True)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "audit_benchmark_worker.py"), bp.id],
                cwd=str(ROOT), timeout=PER_PROMPT_TIMEOUT_SECONDS,
                capture_output=True, text=True,
            )
            elapsed = time.perf_counter() - t0
            if proc.returncode == 0 and result_path.exists():
                outcome = "completed"
            elif result_path.exists():
                outcome = "completed_with_nonzero_exit"
            else:
                outcome = "crashed_no_result_file"
            log.append({
                "id": bp.id, "outcome": outcome, "returncode": proc.returncode,
                "elapsed_seconds": elapsed, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-4000:],
            })
            print(f"{bp.id}: {outcome} (returncode={proc.returncode}, {elapsed:.1f}s)")
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - t0
            log.append({
                "id": bp.id, "outcome": "timeout", "elapsed_seconds": elapsed,
                "stdout_tail": (exc.stdout or b"").decode("utf-8", "replace")[-2000:] if isinstance(exc.stdout, bytes) else (exc.stdout or "")[-2000:],
            })
            print(f"{bp.id}: TIMEOUT after {elapsed:.1f}s")
        except Exception as exc:
            log.append({"id": bp.id, "outcome": "driver_error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{bp.id}: driver-level error: {exc}")

        DRIVER_LOG.write_text(json.dumps(log, indent=2))

    completed = sum(1 for e in log if e["outcome"] in ("completed", "already_present"))
    print(f"\n=== driver finished: {completed}/{len(PROMPTS)} prompts produced a result file ===")


if __name__ == "__main__":
    main()
