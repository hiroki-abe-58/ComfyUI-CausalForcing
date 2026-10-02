"""Start, watch, cancel and collect Causal Forcing runtime jobs from the ComfyUI process.

Adapted from ComfyUI-MonarchRT (same author, Apache-2.0).

Design rules (see docs/SECURITY.md):

- The command line is an argv list built only from the administrator's runtime
  config and this package's own runtime scripts; never ``shell=True``.
- The workflow contributes data only (prompt text, seed, profile, count), which
  travels as a schema-checked UTF-8 JSON file inside a fresh job directory.
- The child gets an allowlisted environment; tokens and the ComfyUI process
  environment are not inherited.
- stdout/stderr go straight to files in the job directory (no pipes to fill up),
  progress is read from ``events.jsonl``.
- On cancel/timeout/error the runner's own process group is stopped with
  ``runtime/cf_kill.py`` (which checks that the pid still belongs to
  this job), then the local launcher process is killed.

This isolates *configuration and arguments*; it is not a sandbox. The runtime
runs with the permissions of the user that started ComfyUI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from .config import Runtime

PACKAGE_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PACKAGE_DIR / "runtime"
JOB_SCRIPT = "cf_job.py"
DOCTOR_SCRIPT = "cf_doctor.py"
KILL_SCRIPT = "cf_kill.py"
# the only programs this package ever asks a runtime to execute
RUNTIME_SCRIPTS = {name: RUNTIME_DIR / name for name in (JOB_SCRIPT, DOCTOR_SCRIPT, KILL_SCRIPT)}

PROFILES = ("cfpp_framewise_2step", "cf_framewise_4step", "cfpp_framewise_1step", "cf_chunkwise_4step")
FRAMEWISE = ("cfpp_framewise_2step", "cf_framewise_4step", "cfpp_framewise_1step")
# generator forwards per 21-latent-frame video (denoising steps incl. the 4-step first frame of the 1/2-step
# models, plus one clean-context KV update per block, plus the input-image context pass in I2V)
FORWARDS = {
    ("cfpp_framewise_2step", "t2v"): 65,
    ("cfpp_framewise_2step", "i2v"): 63,
    ("cf_framewise_4step", "t2v"): 105,
    ("cf_framewise_4step", "i2v"): 101,
    ("cfpp_framewise_1step", "t2v"): 45,
    ("cfpp_framewise_1step", "i2v"): 44,
    ("cf_chunkwise_4step", "t2v"): 35,
}
IMAGE_NAME = "input.png"
MAX_VIDEOS = 8
MAX_PROMPT_CHARS = 2000
MAX_RESULT_BYTES = 4 * 1024 * 1024
_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,80}")
_VIDEO_FILE_RE = re.compile(r"videos/\d{2}\.mp4")


class RuntimeJobError(RuntimeError):
    """The runtime reported an error (invalid job, runtime failure, missing files...)."""


class JobCancelled(RuntimeError):
    pass


class JobTimeout(RuntimeError):
    pass


@dataclass
class JobOutcome:
    job_dir: Path
    result: dict
    videos: list[Path]


# --------------------------------------------------------------------------- paths and argv


def _wsl_exe() -> str:
    root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or r"C:\Windows"
    return str(PureWindowsPath(root) / "System32" / "wsl.exe")


def host_to_runtime_path(rt: Runtime, path: Path) -> str:
    """Translate a host path (job directory, runtime script) into the path the runtime sees."""
    if rt.kind == "posix":
        return str(path)
    s = str(path)
    m = re.fullmatch(r"([A-Za-z]):[\\/](.*)", s)
    if not m or s.startswith("\\\\"):
        raise RuntimeJobError(f"WSL runtimes need local drive paths (X:\\...), got {s!r}")
    rest = m.group(2).replace("\\", "/").strip("/")
    return f"{rt.wsl_mount_root}{m.group(1).lower()}/{rest}" if rest else f"{rt.wsl_mount_root}{m.group(1).lower()}"


def build_argv(rt: Runtime, script: str, *args: str) -> list[str]:
    return argv_for(rt, host_to_runtime_path(rt, RUNTIME_SCRIPTS[script]), *args)


def argv_for(rt: Runtime, script_path: str, *args: str) -> list[str]:
    """argv that runs one of this package's scripts (already translated to a runtime path) with the runtime's Python."""
    if rt.kind == "wsl":
        return [_wsl_exe(), "-d", rt.distro, "--cd", "/", "--exec", rt.python, script_path, *args]
    return [rt.python, script_path, *args]


def child_env(rt: Runtime) -> dict[str, str]:
    """Minimal environment for the launcher process. Nothing secret, nothing from WSLENV."""
    src = {k.upper(): v for k, v in os.environ.items()} if os.name == "nt" else dict(os.environ)
    if rt.kind == "wsl":
        keep = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC", "PROGRAMDATA", "LOCALAPPDATA", "USERPROFILE")
        env = {k: src[k] for k in keep if k in src}
        sysroot = env.get("SYSTEMROOT", r"C:\Windows")
        env["PATH"] = f"{sysroot}\\System32;{sysroot}"
        env["WSLENV"] = ""  # do not forward any Windows variable into Linux
        return env
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TMPDIR", "USER", "LOGNAME")
    if os.name == "nt":  # only the test suite runs 'posix' runtimes on a Windows host; Python needs these there
        keep += ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP")
    env = {k: src[k] for k in keep if k in src}
    env.setdefault("PATH", "/usr/bin:/bin")
    return env


def jobs_root(rt: Runtime) -> Path:
    if rt.jobs_dir:
        root = Path(rt.jobs_dir)
    else:
        import folder_paths

        root = Path(folder_paths.get_temp_directory()) / "causalforcing"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def new_job_dir(rt: Runtime, kind: str = "job") -> tuple[str, Path]:
    job_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:12]}"
    assert _JOB_ID_RE.fullmatch(job_id)
    root = jobs_root(rt)
    job_dir = (root / job_id).resolve()
    if job_dir.parent != root:
        raise RuntimeJobError("job directory escaped the jobs root")
    job_dir.mkdir(parents=False, exist_ok=False)
    return job_id, job_dir


def make_job(rt: Runtime, job_id: str, profile: str, videos: list[dict], mode: str = "t2v") -> dict:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if profile not in rt.checkpoints:
        raise ValueError(f"runtime {rt.id!r} has no checkpoint for {profile!r}")
    if mode not in ("t2v", "i2v") or (profile, mode) not in FORWARDS:
        raise ValueError(f"{profile} does not support {mode}")
    if not 1 <= len(videos) <= MAX_VIDEOS:
        raise ValueError(f"1..{MAX_VIDEOS} videos per job")
    clean = []
    for v in videos:
        prompt, seed = v["prompt"], v["seed"]
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS or "\0" in prompt:
            raise ValueError(f"prompt must be 1..{MAX_PROMPT_CHARS} characters")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**31 - 1:
            raise ValueError("seed must be an integer in [0, 2^31-1]")
        clean.append({"prompt": prompt, "seed": seed})
    return {
        "schema_version": 1,
        "job_id": job_id,
        "profile": profile,
        "mode": mode,
        "videos": clean,
        "upstream_dir": rt.upstream_dir,
        "models_dir": rt.models_dir,
        "checkpoint": rt.checkpoints[profile],
        "fps": 16,
        "env": dict(rt.env),
        "offload_text_encoder": rt.offload_text_encoder,
    }


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- process control


def _popen(rt: Runtime, argv: list[str], job_dir: Path, stdin_pipe: bool) -> subprocess.Popen:
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    with open(job_dir / "stdout.log", "wb") as out, open(job_dir / "stderr.log", "wb") as err:
        return subprocess.Popen(  # noqa: S603 - argv list from admin config + package scripts, no shell
            argv,
            stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env=child_env(rt),
            cwd=str(job_dir),
            close_fds=True,
            **kwargs,
        )


def stop_job(rt: Runtime, proc: subprocess.Popen | None, job_dir: Path, grace_s: float = 10.0) -> dict:
    """Stop the runner of ``job_dir`` and everything it started. Returns the kill helper's report."""
    try:
        (job_dir / "CANCEL").write_text("cancel\n", encoding="utf-8")
    except OSError:
        pass
    if proc is not None and proc.stdin is not None:
        try:
            proc.stdin.close()
        except OSError:
            pass
    if proc is not None:
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    report: dict = {}
    kill_argv = build_argv(rt, KILL_SCRIPT, host_to_runtime_path(rt, job_dir), "5")
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            kill_argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=child_env(rt),
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        out = done.stdout.decode("utf-8", "replace").strip().splitlines()
        report = json.loads(out[-1]) if out else {"status": "no_output", "returncode": done.returncode}
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        report = {"status": "kill_helper_failed", "error": type(exc).__name__}
    if proc is not None and proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            report["launcher_still_running"] = True
    try:
        _write_json(job_dir / "stop_report.json", report)
    except OSError:
        pass
    return report


def _read_events(path: Path, offset: int) -> tuple[list[dict], int]:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    events = []
    for line in data[: end + 1].splitlines():
        try:
            ev = json.loads(line.decode("utf-8"))
            if isinstance(ev, dict):
                events.append(ev)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return events, offset + end + 1


def _error_from_events(job_dir: Path) -> str:
    events, _ = _read_events(job_dir / "events.jsonl", 0)
    for ev in reversed(events):
        if ev.get("event") in ("failed", "invalid_job"):
            return f"{ev.get('error_type', ev['event'])}: {ev.get('error', '')}"[:2000]
    try:
        tail = (job_dir / "stderr.log").read_bytes()[-1500:].decode("utf-8", "replace")
    except OSError:
        tail = ""
    return f"runtime exited without a result; stderr tail:\n{tail}"


def run_process(
    rt: Runtime,
    script: str,
    job_dir: Path,
    request_name: str,
    *,
    timeout_s: float,
    interrupted: Callable[[], bool] = lambda: False,
    on_event: Callable[[dict], None] | None = None,
    poll_s: float = 0.25,
) -> int:
    """Run one runtime script on ``job_dir/request_name`` and return its exit code."""
    extra = ["--watch-stdin"] if script == JOB_SCRIPT else []
    argv = build_argv(rt, script, *extra, host_to_runtime_path(rt, job_dir / request_name))
    proc = _popen(rt, argv, job_dir, stdin_pipe=True)
    deadline = time.monotonic() + timeout_s
    offset = 0
    try:
        while True:
            events, offset = _read_events(job_dir / "events.jsonl", offset)
            if on_event:
                for ev in events:
                    on_event(ev)
            code = proc.poll()
            if code is not None:
                if proc.stdin is not None:
                    proc.stdin.close()
                events, offset = _read_events(job_dir / "events.jsonl", offset)
                if on_event:
                    for ev in events:
                        on_event(ev)
                return code
            if interrupted():
                stop_job(rt, proc, job_dir)
                raise JobCancelled("cancelled by the user")
            if time.monotonic() > deadline:
                stop_job(rt, proc, job_dir)
                raise JobTimeout(f"runtime job exceeded {timeout_s:.0f} s")
            time.sleep(poll_s)
    except (JobCancelled, JobTimeout):
        raise
    except BaseException:
        stop_job(rt, proc, job_dir)
        raise


# --------------------------------------------------------------------------- jobs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_result(job_dir: Path, job: dict) -> JobOutcome:
    path = job_dir / "result.json"
    if not path.is_file() or path.stat().st_size > MAX_RESULT_BYTES:
        raise RuntimeJobError("runtime finished without a valid result.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("job_id") != job["job_id"] or result.get("profile") != job["profile"]:
        raise RuntimeJobError("result.json does not belong to this job")
    records = result.get("videos")
    if not isinstance(records, list) or len(records) != len(job["videos"]):
        raise RuntimeJobError("result.json has the wrong number of videos")
    files = []
    videos_dir = (job_dir / "videos").resolve()
    for rec in records:
        rel = rec.get("file")
        if not isinstance(rel, str) or not _VIDEO_FILE_RE.fullmatch(rel):
            raise RuntimeJobError(f"unexpected video path in result.json: {rel!r}")
        p = (job_dir / rel).resolve()
        if p.parent != videos_dir or not p.is_file():
            raise RuntimeJobError(f"video file missing: {rel}")
        if _sha256(p) != rec.get("sha256"):
            raise RuntimeJobError(f"video file does not match its recorded sha256: {rel}")
        if rec.get("rgb_frames") != 81 or (rec.get("width"), rec.get("height")) != (832, 480):
            raise RuntimeJobError(f"unexpected video geometry in result.json: {rec.get('rgb_frames')} frames {rec.get('width')}x{rec.get('height')}")
        files.append(p)
    return JobOutcome(job_dir=job_dir, result=result, videos=files)


def generate(
    rt: Runtime,
    profile: str,
    videos: list[dict],
    *,
    image=None,
    interrupted: Callable[[], bool] = lambda: False,
    on_progress: Callable[[int, int], None] | None = None,
    timeout_s: float | None = None,
) -> JobOutcome:
    """image: optional HxWx3 uint8 array for image-to-video (frame-wise profiles only)."""
    mode = "i2v" if image is not None else "t2v"
    job_id, job_dir = new_job_dir(rt, "job")
    job = make_job(rt, job_id, profile, videos, mode)
    if image is not None:
        save_image(image, job_dir / IMAGE_NAME)
    _write_json(job_dir / "job.json", job)
    per_video = FORWARDS[(profile, mode)]
    total = per_video * len(videos)

    def on_event(ev: dict) -> None:
        if on_progress and ev.get("event") == "progress":
            done = int(ev.get("video", 0)) * per_video + int(ev.get("forwards", 0))
            on_progress(min(done, total), total)

    code = run_process(rt, JOB_SCRIPT, job_dir, "job.json", timeout_s=timeout_s or rt.timeout_minutes * 60, interrupted=interrupted, on_event=on_event)
    if code == 4:
        stop_job(rt, None, job_dir, grace_s=0)
        raise JobCancelled("the runtime job was cancelled")
    if code != 0:
        stop_job(rt, None, job_dir, grace_s=0)  # make sure nothing of a failed job lingers
        raise RuntimeJobError(f"runtime job failed (exit {code}): {_error_from_events(job_dir)}")
    return load_result(job_dir, job)


def save_image(image, path: Path) -> None:
    """Write the conditioning image as an 8-bit RGB PNG (the runtime applies the official resize/normalise)."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4) or arr.dtype != np.uint8 or arr.shape[0] < 16 or arr.shape[1] < 16:
        raise ValueError("image must be an HxWx3 uint8 array")
    if arr.shape[0] * arr.shape[1] > 4096 * 4096:
        raise ValueError("image is larger than 16 megapixels")
    Image.fromarray(arr[:, :, :3]).save(path, format="PNG")


def doctor(rt: Runtime, *, verify_sha256: bool, interrupted: Callable[[], bool] = lambda: False) -> dict:
    job_id, job_dir = new_job_dir(rt, "doctor")
    req = {
        "schema_version": 1,
        "job_id": job_id,
        "upstream_dir": rt.upstream_dir,
        "models_dir": rt.models_dir,
        "checkpoints": dict(rt.checkpoints),
        "env": dict(rt.env),
        "offload_text_encoder": rt.offload_text_encoder,
        "verify_sha256": bool(verify_sha256),
    }
    _write_json(job_dir / "request.json", req)
    code = run_process(rt, DOCTOR_SCRIPT, job_dir, "request.json", timeout_s=3600, interrupted=interrupted)
    path = job_dir / "doctor.json"
    if path.is_file() and path.stat().st_size < MAX_RESULT_BYTES:
        report = json.loads(path.read_text(encoding="utf-8"))
    else:
        report = {"overall": "fail", "checks": [{"name": "launch", "status": "fail", "detail": _error_from_events(job_dir)}]}
    report["runtime_id"] = rt.id
    report["exit_code"] = code
    report["host"] = {"python": sys.version.split()[0], "platform": sys.platform}
    return report
