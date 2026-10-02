"""Real subprocesses with the fake runtime: success, failure, cancel, timeout, cleanup."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from causalforcing_comfy import client

POSIX = os.name == "posix"


def _alive(pid: int) -> bool:
    if POSIX:
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                return f.read().rsplit(b")", 1)[1].split()[0] != b"Z"
        except OSError:
            return False
    import ctypes

    h = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)  # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
    if not h:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT -> still running
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def _wait_gone(pid: int, seconds: float = 15) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def test_success_two_videos(fake_runtime):
    seen = []
    out = client.generate(
        fake_runtime,
        "cfpp_framewise_2step",
        [{"prompt": "MODE:ok a fox", "seed": 1}, {"prompt": "MODE:ok a fox", "seed": 2}],
        on_progress=lambda d, t: seen.append((d, t)),
    )
    assert [p.name for p in out.videos] == ["00.mp4", "01.mp4"]
    assert seen and seen[-1][1] == 130 and seen == sorted(seen)
    assert out.result["fake_runtime"] is True
    job = json.loads((out.job_dir / "job.json").read_text(encoding="utf-8"))
    assert job["videos"][1]["seed"] == 2 and job["profile"] == "cfpp_framewise_2step" and job["mode"] == "t2v"
    assert (out.job_dir / "stdout.log").stat().st_size > 0


def test_failure_is_reported(fake_runtime):
    with pytest.raises(client.RuntimeJobError, match="simulated failure"):
        client.generate(fake_runtime, "cf_framewise_4step", [{"prompt": "MODE:fail", "seed": 1}])


def test_result_outside_videos_dir_is_rejected(fake_runtime):
    with pytest.raises(client.RuntimeJobError, match="unexpected video path"):
        client.generate(fake_runtime, "cf_framewise_4step", [{"prompt": "MODE:badfile", "seed": 1}])


@pytest.mark.parametrize("how", ["interrupt", "timeout"])
def test_cancel_and_timeout_stop_the_whole_tree(fake_runtime, how):
    state = {"t0": time.time()}

    def interrupted():
        return how == "interrupt" and time.time() - state["t0"] > 2.0

    exc = client.JobCancelled if how == "interrupt" else client.JobTimeout
    with pytest.raises(exc):
        client.generate(
            fake_runtime, "cf_framewise_4step", [{"prompt": "MODE:hang", "seed": 1}], interrupted=interrupted, timeout_s=4 if how == "timeout" else 120
        )
    job_dir = max(client.jobs_root(fake_runtime).iterdir(), key=lambda p: p.stat().st_mtime)
    runner = json.loads((job_dir / "runner.pid").read_text(encoding="utf-8"))
    assert (job_dir / "CANCEL").exists() and (job_dir / "stop_report.json").exists()
    assert _wait_gone(runner["pid"]), "runner still alive"
    grandchild = int((job_dir / "grandchild.pid").read_text(encoding="utf-8"))
    if POSIX:
        # the kill helper stops the runner's process group, grandchildren included
        assert _wait_gone(grandchild), "grandchild still alive"
        report = json.loads((job_dir / "stop_report.json").read_text(encoding="utf-8"))
        assert report["status"] in ("stopped", "not_running"), report
    else:
        # Windows host + 'posix' kind only exists in these tests; kill the orphan ourselves.
        # (Real Windows hosts use kind 'wsl', where the same helper runs inside Linux.)
        os.kill(grandchild, 9)


@pytest.mark.skipif(not POSIX, reason="the kill helper reads /proc; on Windows hosts it runs inside WSL")
def test_kill_helper_refuses_foreign_pid(tmp_path):
    import subprocess

    job_dir = tmp_path / "job-x"
    job_dir.mkdir()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        (job_dir / "runner.pid").write_text(json.dumps({"pid": other.pid, "pgid": other.pid}), encoding="utf-8")
        done = subprocess.run([sys.executable, str(client.RUNTIME_DIR / client.KILL_SCRIPT), str(job_dir), "1"], capture_output=True, text=True)
        assert json.loads(done.stdout)["status"] == "refused"
        assert other.poll() is None, "a process that is not this job's runner was signalled"
    finally:
        other.kill()
        other.wait()


@pytest.mark.skipif(not POSIX, reason="the kill helper reads /proc; on Windows hosts it runs inside WSL")
def test_kill_helper_stops_orphans_after_the_runner_died(fake_runtime):
    """The runner is SIGKILLed from outside (no chance to clean up); its grandchild must still be stopped."""
    import signal
    import subprocess

    fake_job_script = client.RUNTIME_SCRIPTS[client.JOB_SCRIPT]  # patched by the fake_runtime fixture
    job_id, job_dir = client.new_job_dir(fake_runtime)
    job = client.make_job(fake_runtime, job_id, "cf_framewise_4step", [{"prompt": "MODE:hang", "seed": 1}])
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    runner = subprocess.Popen([sys.executable, str(fake_job_script), str(job_dir / "job.json")], start_new_session=True)
    try:
        end = time.time() + 20
        while not (job_dir / "grandchild.pid").exists() and time.time() < end:
            time.sleep(0.1)
        grandchild = int((job_dir / "grandchild.pid").read_text(encoding="utf-8"))
        os.kill(runner.pid, signal.SIGKILL)
        runner.wait()
        assert _alive(grandchild)
        done = subprocess.run([sys.executable, str(client.RUNTIME_DIR / client.KILL_SCRIPT), str(job_dir), "2"], capture_output=True, text=True)
        report = json.loads(done.stdout)
        assert report["status"] == "stopped" and report.get("orphans") is True and grandchild in report["signalled"], report
        assert _wait_gone(grandchild)
    finally:
        if runner.poll() is None:
            runner.kill()


@pytest.mark.skipif(not POSIX, reason="the kill helper reads /proc; on Windows hosts it runs inside WSL")
def test_kill_helper_stops_a_doctor_run(tmp_path):
    """A cancelled Doctor (cf_doctor.py <dir>/request.json) is stopped like a job runner."""
    import subprocess

    job_dir = tmp_path / "doctor-x"
    job_dir.mkdir()
    (job_dir / "request.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "cf_doctor.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script), str(job_dir / "request.json")], start_new_session=True)
    try:
        with open(f"/proc/{proc.pid}/stat", "rb") as f:
            starttime = int(f.read().decode().rsplit(")", 1)[1].split()[19])
        (job_dir / "runner.pid").write_text(json.dumps({"pid": proc.pid, "pgid": proc.pid, "starttime": starttime}), encoding="utf-8")
        done = subprocess.run([sys.executable, str(client.RUNTIME_DIR / client.KILL_SCRIPT), str(job_dir), "1"], capture_output=True, text=True)
        assert json.loads(done.stdout)["status"] == "stopped"
        assert proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
