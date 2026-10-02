"""Test setup.

All tests here run on CPU without model weights. Tests marked ``comfy`` import
a real ComfyUI checkout given by ``COMFYUI_PATH`` and fail (not skip) when it
is missing, so CI cannot pass by silently skipping them; deselect them
explicitly with ``-m "not comfy"``. Real-GPU checks live in
``scripts/gpu_e2e.py`` (see docs/TESTING.md).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FAKE_RUNTIME = Path(__file__).resolve().parent / "fake_runtime"

_COMFY_STATE: dict = {}


def _boot_comfy():
    if _COMFY_STATE:
        return _COMFY_STATE
    path = os.environ.get("COMFYUI_PATH")
    if not path or not (Path(path) / "comfy").is_dir():
        pytest.fail("COMFYUI_PATH must point to a ComfyUI checkout for tests marked 'comfy'", pytrace=False)
    if path not in sys.path:
        sys.path.insert(0, path)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    comfy.cli_args.args.disable_all_custom_nodes = False
    import nodes  # noqa: E402  (ComfyUI's nodes.py)

    asyncio.run(nodes.init_builtin_extra_nodes())
    ok = asyncio.run(nodes.load_custom_node(str(REPO_ROOT)))
    if not ok:
        pytest.fail("ComfyUI failed to load the custom node package", pytrace=False)
    _COMFY_STATE["nodes"] = nodes
    _COMFY_STATE["path"] = Path(path)
    return _COMFY_STATE


@pytest.fixture(scope="session")
def comfyui():
    return _boot_comfy()


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    """A 'posix' runtime that runs tests/fake_runtime/cf_job.py with this interpreter."""
    from causalforcing_comfy import client
    from causalforcing_comfy.config import parse_runtime

    monkeypatch.setitem(client.RUNTIME_SCRIPTS, client.JOB_SCRIPT, FAKE_RUNTIME / "cf_job.py")
    jobs = tmp_path / "jobs dir ü"
    rt = parse_runtime(
        "fake",
        {
            "kind": "posix",
            "python": "/fake/python",
            "upstream_dir": "/fake/upstream",
            "models_dir": "/fake/models",
            "checkpoints": {"cfpp_framewise_2step": "/fake/2step.pt", "cf_framewise_4step": "/fake/4step.pt"},
            "jobs_dir": str(jobs),
            "timeout_minutes": 5,
        },
    )
    # the interpreter path is a host path here (Windows or Linux), which a real 'posix' runtime config would not allow
    object.__setattr__(rt, "python", sys.executable)
    return rt


@pytest.fixture
def runtimes_file(tmp_path, monkeypatch):
    """Write a runtimes config and point COMFYUI_CAUSALFORCING_CONFIG at it."""

    def write(runtimes: dict) -> Path:
        p = tmp_path / "causalforcing.runtimes.json"
        p.write_text(json.dumps({"schema_version": 1, "runtimes": runtimes}), encoding="utf-8")
        monkeypatch.setenv("COMFYUI_CAUSALFORCING_CONFIG", str(p))
        return p

    return write


def pytest_configure(config):
    config.addinivalue_line("markers", "comfy: needs a ComfyUI checkout (COMFYUI_PATH)")
