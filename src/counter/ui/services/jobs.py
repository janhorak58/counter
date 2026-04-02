from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class JobHandle:
    kind: str
    command: List[str]
    cwd: str
    process: subprocess.Popen[str]
    log_queue: queue.Queue[str]
    logs: List[str] = field(default_factory=list)
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    exit_code: Optional[int] = None
    output_path: Optional[str] = None


def _log_reader(proc: subprocess.Popen[str], out_q: queue.Queue[str]) -> None:
    if proc.stdout is None:
        return

    try:
        while True:
            line = proc.stdout.readline()
            if line == "":
                break
            out_q.put(line.rstrip("\n"))
    except Exception as exc:  # pragma: no cover - best effort logging
        out_q.put(f"[log_reader_error] {type(exc).__name__}: {exc}")


def start_job(
    *,
    kind: str,
    command: List[str],
    cwd: str | Path,
    env_extra: Optional[Dict[str, str]] = None,
) -> JobHandle:
    env = os.environ.copy()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})

    popen_kwargs = {
        "cwd": str(cwd),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen([str(x) for x in command], **popen_kwargs)

    q: queue.Queue[str] = queue.Queue()
    t = threading.Thread(target=_log_reader, args=(proc, q), daemon=True)
    t.start()

    return JobHandle(
        kind=str(kind),
        command=[str(x) for x in command],
        cwd=str(cwd),
        process=proc,
        log_queue=q,
    )


def poll_job(job: JobHandle, *, max_logs: int = 5000) -> JobHandle:
    while True:
        try:
            line = job.log_queue.get_nowait()
            job.logs.append(line)
        except queue.Empty:
            break

    if len(job.logs) > max_logs:
        job.logs = job.logs[-max_logs:]

    rc = job.process.poll()
    if rc is None:
        job.status = "running"
        return job

    if job.status == "running":
        job.exit_code = int(rc)
        job.ended_at = time.time()
        job.status = "completed" if rc == 0 else "failed"

    return job


def cancel_job(job: JobHandle, *, timeout_s: float = 5.0) -> JobHandle:
    if job.process.poll() is not None:
        return poll_job(job)

    try:
        if os.name == "posix":
            os.killpg(job.process.pid, signal.SIGTERM)
        else:
            job.process.terminate()
        job.process.wait(timeout=timeout_s)
    except Exception:
        try:
            if os.name == "posix":
                os.killpg(job.process.pid, signal.SIGKILL)
            else:
                job.process.kill()
            job.process.wait(timeout=timeout_s)
        except Exception:
            pass

    while True:
        try:
            line = job.log_queue.get_nowait()
            job.logs.append(line)
        except queue.Empty:
            break

    rc = job.process.poll()
    job.exit_code = int(rc) if rc is not None else None
    job.status = "cancelled"
    job.ended_at = time.time()

    return job


def guess_output_path_from_logs(logs: List[str], *, cwd: str | Path | None = None) -> Optional[str]:
    base = Path(cwd) if cwd is not None else None

    for line in reversed(logs):
        candidate = (line or "").strip().strip('"').strip("'")
        if not candidate:
            continue

        p = Path(candidate)
        if not p.is_absolute() and base is not None:
            p = (base / p).resolve()

        if p.exists():
            return str(p)

    return None
