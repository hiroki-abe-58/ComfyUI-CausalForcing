"""Config validation, argv/env/job construction, runner job validation and result validation (pure, CPU)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest

from causalforcing_comfy import client
from causalforcing_comfy.config import ConfigError, load_runtimes, parse_runtime

REPO = Path(__file__).resolve().parents[1]
BASE = {
    "kind": "wsl",
    "distro": "Ubuntu-24.04",
    "python": "/opt/cf/venv/bin/python",
    "upstream_dir": "/opt/cf/Causal-Forcing",
    "models_dir": "/opt/cf/models",
    "checkpoints": {"cfpp_framewise_2step": "/opt/cf/ckpt/framewise-2step.pt", "cf_framewise_4step": "/opt/cf/ckpt/causal_forcing.pt"},
    "env": {"TRITON_CACHE_DIR": "/opt/cf/cache/triton"},
}
WSL = parse_runtime("w", BASE)


def _runner():
    spec = importlib.util.spec_from_file_location("cf_job_under_test", REPO / "runtime" / "cf_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_example_config_parses():
    data = json.loads((REPO / "examples" / "causalforcing.runtimes.example.json").read_text(encoding="utf-8"))
    rts = {k: parse_runtime(k, v) for k, v in data["runtimes"].items()}
    assert all(set(r.checkpoints) <= set(client.PROFILES) for r in rts.values())


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"distro": "Ubuntu; rm -rf /"}, "distro"),
        ({"python": "venv/bin/python"}, "python"),
        ({"upstream_dir": "/opt/../x"}, "'..'"),
        ({"checkpoints": {}}, "checkpoints"),
        ({"checkpoints": {"other_model": "/x.pt"}}, "checkpoints"),
        ({"checkpoints": {"cf_framewise_4step": "relative.pt"}}, "checkpoints"),
        ({"checkpoint": "/x.pt"}, "unknown keys"),
        ({"env": {"HF_TOKEN": "x"}}, "env"),
        ({"env": {"FLASHINFER_WORKSPACE_BASE": "/x"}}, "env"),
        ({"timeout_minutes": 0}, "timeout"),
        ({"kind": "shell"}, "kind"),
    ],
)
def test_rejects_bad_runtime(patch, message):
    with pytest.raises(ConfigError, match=message):
        parse_runtime("rt", {**BASE, **patch})


def test_missing_and_broken_config(tmp_path, runtimes_file):
    assert load_runtimes(tmp_path / "none.json") == {}
    p = tmp_path / "bad.json"
    p.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_runtimes(p)
    runtimes_file({"a": BASE})
    assert list(load_runtimes()) == ["a"]


def test_argv_and_env(monkeypatch):
    monkeypatch.setitem(client.RUNTIME_SCRIPTS, client.JOB_SCRIPT, PureWindowsPath(r"E:\nodes\ComfyUI-CausalForcing\runtime\cf_job.py"))
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("HF_TOKEN", "secret-value")
    monkeypatch.setenv("WSLENV", "HF_TOKEN/u")
    argv = client.build_argv(WSL, client.JOB_SCRIPT, "--watch-stdin", "/mnt/e/jobs/j/job.json")
    assert argv == [
        r"C:\Windows\System32\wsl.exe",
        "-d",
        "Ubuntu-24.04",
        "--cd",
        "/",
        "--exec",
        "/opt/cf/venv/bin/python",
        "/mnt/e/nodes/ComfyUI-CausalForcing/runtime/cf_job.py",
        "--watch-stdin",
        "/mnt/e/jobs/j/job.json",
    ]
    env = client.child_env(WSL)
    assert env["WSLENV"] == "" and "secret-value" not in env.values()


def test_job_round_trip_and_profile_rules(tmp_path):
    runner = _runner()
    job = client.make_job(WSL, "job-20260101-000000-abcdef012345", "cfpp_framewise_2step", [{"prompt": "a cat 猫", "seed": 7}], "t2v")
    assert job["checkpoint"] == "/opt/cf/ckpt/framewise-2step.pt" and job["mode"] == "t2v"
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job), encoding="utf-8")
    assert runner.load_job(p)["profile"] == "cfpp_framewise_2step"
    with pytest.raises(ValueError, match="no checkpoint"):
        client.make_job(WSL, "job-20260101-000000-abcdef012345", "cfpp_framewise_1step", [{"prompt": "x", "seed": 1}])
    rt_chunk = parse_runtime("c", {**BASE, "checkpoints": {"cf_chunkwise_4step": "/opt/cf/ckpt/chunk.pt"}})
    with pytest.raises(ValueError, match="does not support i2v"):
        client.make_job(rt_chunk, "job-20260101-000000-abcdef012345", "cf_chunkwise_4step", [{"prompt": "x", "seed": 1}], "i2v")


@pytest.mark.parametrize(
    "patch",
    [
        {"job_id": "../x"},
        {"profile": "cf_framewise_8step"},
        {"mode": "v2v"},
        {"fps": 24},
        {"env": {"LD_PRELOAD": "/x.so"}},
        {"extra": 1},
        {"videos": [{"prompt": "x", "seed": 1, "cmd": "id"}]},
    ],
)
def test_runner_rejects_tampered_jobs(tmp_path, patch):
    runner = _runner()
    job = client.make_job(WSL, "job-20260101-000000-abcdef012345", "cf_framewise_4step", [{"prompt": "x", "seed": 1}])
    job.update(patch)
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(runner.JobError):
        runner.load_job(p)


def test_runner_i2v_needs_image_and_framewise(tmp_path):
    runner = _runner()
    job = client.make_job(WSL, "job-20260101-000000-abcdef012345", "cf_framewise_4step", [{"prompt": "x", "seed": 1}], "i2v")
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(runner.JobError, match="input.png"):
        runner.load_job(p)
    client.save_image(np.zeros((64, 96, 3), dtype=np.uint8), tmp_path / "input.png")
    assert runner.load_job(p)["mode"] == "i2v"
    job["profile"] = "cf_chunkwise_4step"
    p.write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(runner.JobError, match="frame-wise"):
        runner.load_job(p)


def test_save_image_rejects_bad_arrays(tmp_path):
    for bad in (np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((64, 64), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.float32)):
        with pytest.raises(ValueError):
            client.save_image(bad, tmp_path / "x.png")


class _Cfg(dict):
    __getattr__ = dict.get


def test_expected_forward_counts_match_the_official_schedules():
    """The counts the runner enforces, derived from the step lists of the official configs."""
    runner = _runner()
    cfg = lambda steps, first=None, nfpb=1: _Cfg(num_frame_per_block=nfpb, denoising_step_list=steps, denoising_step_list_first_chunk=first)  # noqa: E731
    two = cfg([1000, 500], [1000, 750, 500, 250])
    four = cfg([1000, 750, 500, 250])
    one = cfg([1000], [1000, 750, 500, 250])
    chunk = cfg([1000, 750, 500, 250], None, 3)
    got = {
        ("cfpp_framewise_2step", "t2v"): runner.expected_forwards(two, "t2v"),
        ("cfpp_framewise_2step", "i2v"): runner.expected_forwards(two, "i2v"),
        ("cf_framewise_4step", "t2v"): runner.expected_forwards(four, "t2v"),
        ("cf_framewise_4step", "i2v"): runner.expected_forwards(four, "i2v"),
        ("cfpp_framewise_1step", "t2v"): runner.expected_forwards(one, "t2v"),
        ("cfpp_framewise_1step", "i2v"): runner.expected_forwards(one, "i2v"),
        ("cf_chunkwise_4step", "t2v"): runner.expected_forwards(chunk, "t2v"),
    }
    assert got[("cfpp_framewise_2step", "t2v")] == {"denoise": 44, "context": 21, "blocks": 21, "first_block_steps": 4, "steps_per_block": 2}
    for key, e in got.items():
        assert e["denoise"] + e["context"] == client.FORWARDS[key], key


def test_load_result_validation(fake_runtime):
    job_id, d = client.new_job_dir(fake_runtime)
    job = client.make_job(fake_runtime, job_id, "cf_framewise_4step", [{"prompt": "x", "seed": 1}])
    (d / "videos").mkdir()
    mp4 = d / "videos" / "00.mp4"
    mp4.write_bytes(b"x")
    rec = {"file": "videos/00.mp4", "sha256": hashlib.sha256(b"x").hexdigest(), "rgb_frames": 81, "width": 832, "height": 480}
    (d / "result.json").write_text(json.dumps({"job_id": job_id, "profile": job["profile"], "videos": [rec]}), encoding="utf-8")
    assert client.load_result(d, job).videos == [mp4.resolve()]
    rec["file"] = "../job.json"
    (d / "result.json").write_text(json.dumps({"job_id": job_id, "profile": job["profile"], "videos": [rec]}), encoding="utf-8")
    with pytest.raises(client.RuntimeJobError):
        client.load_result(d, job)


def test_doctor_upstream_hashes_ignore_line_endings(tmp_path):
    """A Windows git with core.autocrlf=true checks the pinned upstream out with CRLF; the doctor must accept it."""
    spec = importlib.util.spec_from_file_location("cf_doctor_under_test", REPO / "runtime" / "cf_doctor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert mod._sha256_text(lf) == mod._sha256_text(crlf) == hashlib.sha256(b"a = 1\nb = 2\n").hexdigest()
    assert len(mod.UPSTREAM_FILES) == 16 and all(len(v) == 64 for v in mod.UPSTREAM_FILES.values())
    assert set(mod.CHECKPOINTS) == set(client.PROFILES)
