"""Stop one Causal Forcing job runner and everything it started (Linux / WSL side).

Adapted from ComfyUI-MonarchRT (same author, Apache-2.0).

Usage: cf_kill.py <job_dir> [grace_seconds]

The runner calls setsid() at start-up and records {"pid", "pgid", "starttime"}
in <job_dir>/runner.pid. This helper only signals that process group and the
runner's descendants, and only after /proc confirms the recorded pid is still
the runner of this very job (command line and start time, so a recycled pid is
never touched). If the runner already exited, it stops only processes that
are still in the runner's own group and session and started after it. It
prints a JSON report on stdout and exits 0 when nothing of the job is left
running.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path


def _stat(pid: int) -> list[str] | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    # comm may contain spaces and parentheses; the fields after the last ')' are fixed:
    # 0 state, 1 ppid, 2 pgrp, 3 session, ..., 19 starttime
    return stat[stat.rfind(")") + 2 :].split()


def _procs() -> dict[int, tuple[int, int, int, int]]:
    """pid -> (ppid, pgid, sid, starttime) for every visible process."""
    out = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            st = _stat(int(entry))
            if st:
                out[int(entry)] = (int(st[1]), int(st[2]), int(st[3]), int(st[19]))
    return out


def _cmdline(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [p.decode("utf-8", "replace") for p in f.read().split(b"\0") if p]
    except OSError:
        return []


def _members(root: int, pgid: int) -> set[int]:
    procs = _procs()
    found = {pid for pid, (_, g, _, _) in procs.items() if g == pgid}
    if root in procs:
        found.add(root)
    changed = True
    while changed:  # descendants that left the group (e.g. called setsid themselves)
        changed = False
        for pid, (ppid, _, _, _) in procs.items():
            if ppid in found and pid not in found:
                found.add(pid)
                changed = True
    found.discard(os.getpid())
    return found


def _orphans(pgid: int, starttime: int | None) -> set[int]:
    """Members left behind after the runner itself exited.

    A process group id stays allocated while any member exists, so remaining processes whose group *and*
    session equal the runner's pid, and which started no earlier than the runner, belong to this job.
    """
    if starttime is None:
        return set()
    return {pid for pid, (_, g, sid, st) in _procs().items() if g == pgid and sid == pgid and st >= starttime and pid != os.getpid()}


# the runtime scripts this helper may stop, and the request file each one is started with
SCRIPTS = {"cf_job.py": "job.json", "cf_doctor.py": "request.json"}


def _is_ours(cmd: list[str], job_dir: Path) -> bool:
    return any(any(a.endswith(script) for a in cmd) and str(job_dir / request) in cmd for script, request in SCRIPTS.items())


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print("usage: cf_kill.py <job_dir> [grace_seconds]", file=sys.stderr)
        return 2
    job_dir = Path(argv[1]).resolve()
    grace = float(argv[2]) if len(argv) == 3 else 5.0
    report: dict = {"job_dir_name": job_dir.name}
    try:
        info = json.loads((job_dir / "runner.pid").read_text(encoding="utf-8"))
        pid, pgid = int(info["pid"]), int(info["pgid"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status="no_runner_record", detail=type(exc).__name__)
        print(json.dumps(report))
        return 0
    if pid <= 1 or pgid != pid:
        report.update(status="refused", detail="runner.pid is not a session leader record")
        print(json.dumps(report))
        return 3
    cmd = _cmdline(pid)
    starttime = info.get("starttime")
    starttime = int(starttime) if isinstance(starttime, int) else None
    if not cmd:
        targets = _orphans(pgid, starttime)
        if not targets:
            report.update(status="not_running")
            print(json.dumps(report))
            return 0
        report["orphans"] = True
    elif not _is_ours(cmd, job_dir):
        report.update(status="refused", detail="pid no longer belongs to this job")
        print(json.dumps(report))
        return 3
    else:
        st = _stat(pid)
        if starttime is not None and st and int(st[19]) != starttime:
            report.update(status="refused", detail="pid was reused")
            print(json.dumps(report))
            return 3
        targets = _members(pid, pgid)
    report["signalled"] = sorted(targets)
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        for p in list(targets):
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                targets.discard(p)
            except PermissionError:
                pass
        deadline = time.time() + wait
        while time.time() < deadline:
            # zombies only wait for their parent to reap them; treat state 'Z' as exited
            if not {p for p in targets if os.path.exists(f"/proc/{p}") and not _is_zombie(p)}:
                break
            time.sleep(0.2)
        targets = {p for p in targets if os.path.exists(f"/proc/{p}") and not _is_zombie(p)}
        if not targets:
            break
    report["remaining"] = sorted(targets)
    report["status"] = "stopped" if not targets else "still_running"
    print(json.dumps(report))
    return 0 if not targets else 1


def _is_zombie(pid: int) -> bool:
    st = _stat(pid)
    return st is None or st[0] == "Z"


if __name__ == "__main__":
    sys.exit(main(sys.argv))
