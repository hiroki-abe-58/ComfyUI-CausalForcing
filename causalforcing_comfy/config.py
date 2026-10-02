"""Administrator-registered Causal Forcing runtimes.

A workflow can only *name* a runtime (``runtime_id``). Everything that is
executed - the WSL distribution, the runtime's Python, the upstream checkout,
the model folder - comes from a local JSON file that the ComfyUI API cannot
write to:

1. the path in the ``COMFYUI_CAUSALFORCING_CONFIG`` environment variable, else
2. ``<ComfyUI user directory>/causalforcing.runtimes.json`` (a file at the root of
   the user directory, outside every per-user folder that the userdata API
   serves).

See ``examples/causalforcing.runtimes.example.json``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

CONFIG_ENV = "COMFYUI_CAUSALFORCING_CONFIG"
CONFIG_FILENAME = "causalforcing.runtimes.json"
SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 256 * 1024

# Settings the runner applies inside the runtime (mirrors runtime/cf_job.py ENV_KEYS).
RUNTIME_ENV_KEYS = frozenset({"CC", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME", "HF_HOME", "CUDA_VISIBLE_DEVICES"})
# Profiles a runtime may provide (mirrors runtime/cf_job.py PROFILES).
PROFILE_IDS = ("cfpp_framewise_2step", "cf_framewise_4step", "cfpp_framewise_1step", "cf_chunkwise_4step")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_DISTRO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_RUNTIME_KEYS = {
    "kind",
    "distro",
    "python",
    "upstream_dir",
    "models_dir",
    "checkpoints",
    "env",
    "offload_text_encoder",
    "timeout_minutes",
    "jobs_dir",
    "wsl_mount_root",
    "description",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Runtime:
    id: str
    kind: str  # "wsl" (Windows host -> WSL2 distro) or "posix" (Linux host, same machine)
    python: str
    upstream_dir: str
    models_dir: str
    checkpoints: dict  # profile id -> absolute runtime path of that profile's checkpoint
    distro: str | None = None
    env: dict = field(default_factory=dict)
    offload_text_encoder: bool = False
    timeout_minutes: int = 60
    jobs_dir: str | None = None  # host path; default <ComfyUI temp>/causalforcing
    wsl_mount_root: str = "/mnt/"
    description: str = ""


def _linux_abs_path(value, what: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or any(c in value for c in "\0\n\r"):
        raise ConfigError(f"{what} must be an absolute Linux path")
    if ".." in PurePosixPath(value).parts:
        raise ConfigError(f"{what} must not contain '..'")
    return value


def parse_runtime(rid: str, raw) -> Runtime:
    if not isinstance(rid, str) or not _ID_RE.fullmatch(rid):
        raise ConfigError(f"invalid runtime id {rid!r}")
    if not isinstance(raw, dict):
        raise ConfigError(f"runtime {rid}: expected an object")
    extra = set(raw) - _RUNTIME_KEYS
    if extra:
        raise ConfigError(f"runtime {rid}: unknown keys {sorted(extra)}")
    kind = raw.get("kind")
    if kind not in ("wsl", "posix"):
        raise ConfigError(f"runtime {rid}: kind must be 'wsl' or 'posix'")
    distro = raw.get("distro")
    if kind == "wsl":
        if not isinstance(distro, str) or not _DISTRO_RE.fullmatch(distro):
            raise ConfigError(f"runtime {rid}: distro must be a WSL distribution name")
    elif distro is not None:
        raise ConfigError(f"runtime {rid}: distro is only valid for kind 'wsl'")
    env = raw.get("env", {})
    if not isinstance(env, dict) or set(env) - RUNTIME_ENV_KEYS or not all(isinstance(v, str) and "\0" not in v for v in env.values()):
        raise ConfigError(f"runtime {rid}: env may only set {sorted(RUNTIME_ENV_KEYS)} (strings)")
    timeout = raw.get("timeout_minutes", 60)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 24 * 60:
        raise ConfigError(f"runtime {rid}: timeout_minutes must be an integer in 1..1440")
    offload = raw.get("offload_text_encoder", False)
    if not isinstance(offload, bool):
        raise ConfigError(f"runtime {rid}: offload_text_encoder must be true/false")
    jobs_dir = raw.get("jobs_dir")
    if jobs_dir is not None and (not isinstance(jobs_dir, str) or not Path(jobs_dir).is_absolute()):
        raise ConfigError(f"runtime {rid}: jobs_dir must be an absolute path on this machine")
    mount_root = raw.get("wsl_mount_root", "/mnt/")
    _linux_abs_path(mount_root, f"runtime {rid}: wsl_mount_root")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ConfigError(f"runtime {rid}: description must be a string")
    ckpts = raw.get("checkpoints")
    if not isinstance(ckpts, dict) or not ckpts or set(ckpts) - set(PROFILE_IDS):
        raise ConfigError(f"runtime {rid}: checkpoints must map some of {PROFILE_IDS} to checkpoint paths")
    ckpts = {p: _linux_abs_path(v, f"runtime {rid}: checkpoints.{p}") for p, v in ckpts.items()}
    return Runtime(
        id=rid,
        kind=kind,
        distro=distro,
        python=_linux_abs_path(raw.get("python"), f"runtime {rid}: python"),
        upstream_dir=_linux_abs_path(raw.get("upstream_dir"), f"runtime {rid}: upstream_dir"),
        models_dir=_linux_abs_path(raw.get("models_dir"), f"runtime {rid}: models_dir"),
        checkpoints=ckpts,
        env=dict(env),
        offload_text_encoder=offload,
        timeout_minutes=timeout,
        jobs_dir=jobs_dir,
        wsl_mount_root=mount_root if mount_root.endswith("/") else mount_root + "/",
        description=description[:300],
    )


def config_path() -> Path | None:
    explicit = os.environ.get(CONFIG_ENV)
    if explicit:
        return Path(explicit)
    try:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / CONFIG_FILENAME
    except Exception:
        return None


def load_runtimes(path: Path | None = None) -> dict[str, Runtime]:
    """Return the configured runtimes. A missing file means 'none configured'; a broken file raises."""
    path = path if path is not None else config_path()
    if path is None or not path.is_file():
        return {}
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ConfigError(f"{path.name} is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path.name}: not valid UTF-8 JSON ({exc})") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("runtimes"), dict):
        raise ConfigError(f"{path.name}: expected {{'schema_version': {SCHEMA_VERSION}, 'runtimes': {{...}}}}")
    return {rid: parse_runtime(rid, raw) for rid, raw in data["runtimes"].items()}
