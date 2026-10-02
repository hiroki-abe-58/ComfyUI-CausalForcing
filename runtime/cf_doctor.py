"""Check a Causal Forcing runtime environment (runs inside the runtime, like cf_job.py).

Adapted from ComfyUI-MonarchRT (same author, Apache-2.0).

Usage: cf_doctor.py <request.json>

The request has the same paths/env as a job plus:
  {"schema_version": 1, "job_id": ..., "upstream_dir": ..., "models_dir": ..., "checkpoints": {profile: path},
   "env": {...}, "offload_text_encoder": bool, "verify_sha256": false}

Writes <request dir>/doctor.json and prints it. Exit 0 when no check failed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cf_job import ENV_KEYS, PROFILES, SECRET_RE, UPSTREAM_COMMIT, _proc_stat  # noqa: E402

# sha256 of the upstream files this integration depends on, at UPSTREAM_COMMIT, with line endings normalised
# to LF (a Windows git with core.autocrlf=true checks the same commit out with CRLF)
UPSTREAM_FILES = {
    "inference.py": "210f05bc3437e8057683b61a925c6584f2b7c576bda506dcc810160ad131e8ee",
    "pipeline/causal_inference.py": "9feff55d8b8fe7177c2c437850a51f442ef33b3c7c1837cadfd03c04691a2733",
    "utils/wan_wrapper.py": "5120536c7d58fabc75187245a007f0a1486889372386b698d640fb80b280331d",
    "utils/misc.py": "610bead16d50d99fabdccb2e9ce3c20fa1fa1cd3bdf735d7ff7ffda978bfe195",
    "utils/scheduler.py": "4194ef1324bb06527621749c7f9a82f431ccf011823aa65099cb7bdf116b642e",
    "demo_utils/memory.py": "ca0b31c93019f802e1312d512591ae6bea35b450064cdde19b6202779f50bb7f",
    "wan/modules/causal_model.py": "0566a17f867a7c8ea8ad80d8b79bdfe17aaa4c7859873ffde881a6cdc12c9edd",
    "wan/modules/model.py": "f15214485a5297e6f9e79684bbad80acbcdc88c3272596b256a739a734529449",
    "wan/modules/attention.py": "02ad719a76e6bdec79d18fc62b094ef2ad7c28b2d593c9f395ae1226041b7c9d",
    "wan/modules/t5.py": "8b0cebf3192c542f92a344255a06c203df3ba24160715899a055cc8de0cd930f",
    "wan/modules/vae.py": "a4c1d9362f25a1bef463ac51caed4097317c007d4d87887b02a3a733fdbc48d5",
    "configs/default_config.yaml": "18303f1bacbfbebaa02893cc8e106941753f21a7d1a7109519a90762375ee772",
    "configs/causal_forcing_dmd_framewise_2step.yaml": "4dba99234bef0a9cc7685bccd76c9023ee8938b02ebb9a167757024edc7a0b7d",
    "configs/causal_forcing_dmd_framewise.yaml": "350ace2363afb6539a80dd9eb9a182b468959c479780a0e745af3652701bbd23",
    "configs/causal_forcing_dmd_framewise_1step.yaml": "0b1e43656ba4b0544e85b047be46a79b156d683bfb06eaf99c8817826967623d",
    "configs/causal_forcing_dmd_chunkwise.yaml": "c04ca7933ea3bee5bc11be23b8d3ff1d5c3547986fafe03abdde47841d9dc7d6",
}
# Wan-AI/Wan2.1-T2V-1.3B @ 37ec512624d61f7aa208f7ea8140a131f93afc9a (under models_dir)
MODEL_FILES = {
    "wan_models/Wan2.1-T2V-1.3B/config.json": (249, "ab37994c43740513f94b3ba6233a784035a67b43c8cde83c8f31aa90468c67ce"),
    "wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors": (5676070424, "96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f"),
    "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth": (11361920418, "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"),
    "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth": (507609880, "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"),
    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model": (4548313, "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json": (16837417, "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"),
    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json": (61728, "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"),
    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json": (6623, "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"),
}
# zhuhz22/Causal-Forcing @ 2f8eb8bb6eeb1238da9d13e5420d342a74d634a6 (size, sha256 from the Hugging Face LFS records)
CHECKPOINTS = {
    "cfpp_framewise_2step": ("causal-forcing++/framewise-2step.pt", 5676220819, "8f899a016dffb9539788fac82c6200e43ef6e23b6e61ec6b885bb507a760ce42"),
    "cf_framewise_4step": ("framewise/causal_forcing.pt", 5676220819, "70cae8c3b0b7191fa9840b4f6fe8655ba0cf5c6520c87ba85a46b861c3735cd3"),
    "cfpp_framewise_1step": ("causal-forcing++/framewise-1step.pt", 5676220819, "bdb1b475fc88d528f510158a0990cd457f02c661b527ceb11ca9e4728533e2d0"),
    "cf_chunkwise_4step": ("chunkwise/causal_forcing.pt", 5676282643, "cf75ee5cc6f4e2e336c59c973f5544655d8f0aa481761efe6de1b9cb2eb0cd9d"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_head(repo: Path) -> str | None:
    git = repo / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).is_file():
            return (git / ref).read_text(encoding="utf-8").strip()
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        return None
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: cf_doctor.py <request.json>", file=sys.stderr)
        return 2
    req_path = Path(argv[1]).resolve()
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    me = _proc_stat(os.getpid())
    record = {"pid": os.getpid(), "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None, "starttime": int(me[19]) if me else None}
    (req_path.parent / "runner.pid").write_text(json.dumps(record), encoding="utf-8")
    req = json.loads(req_path.read_text(encoding="utf-8"))
    if not isinstance(req, dict) or req.get("schema_version") != 1:
        print(json.dumps({"error": "unsupported request"}))
        return 2
    env = req.get("env", {})
    if not isinstance(env, dict) or set(env) - ENV_KEYS:
        print(json.dumps({"error": f"env may only contain {sorted(ENV_KEYS)}"}))
        return 2
    for key in list(os.environ):
        if SECRET_RE.search(key):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in env.items()})
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    checks: list[dict] = []

    def add(name: str, status: str, detail) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    report: dict = {"schema_version": 1, "platform": platform.platform(), "python": sys.version.split()[0], "checks": checks}
    upstream = Path(str(req.get("upstream_dir", ""))).resolve()
    head = _git_head(upstream)
    add("upstream_commit", "ok" if head == UPSTREAM_COMMIT else "warn", {"expected": UPSTREAM_COMMIT, "found": head})
    bad = {}
    for rel, digest in UPSTREAM_FILES.items():
        p = upstream / rel
        got = _sha256_text(p) if p.is_file() else None
        if got != digest:
            bad[rel] = "missing" if got is None else "modified"
    add("upstream_files", "fail" if bad else "ok", bad or f"{len(UPSTREAM_FILES)} files match the pinned commit")

    verify = bool(req.get("verify_sha256", False))
    models = Path(str(req.get("models_dir", ""))).resolve()
    problems = {}
    for rel, (size, digest) in MODEL_FILES.items():
        p = models / rel
        if not p.is_file():
            problems[rel] = "missing"
        elif p.stat().st_size != size:
            problems[rel] = f"size {p.stat().st_size} != {size}"
        elif verify and _sha256(p) != digest:
            problems[rel] = "sha256 mismatch"
    add("model_files", "fail" if problems else "ok", problems or ("sizes and sha256 match" if verify else "sizes match (sha256 not checked)"))
    ckpts = req.get("checkpoints", {}) if isinstance(req.get("checkpoints"), dict) else {}
    found = {}
    for profile, path in ckpts.items():
        if profile not in PROFILES or profile not in CHECKPOINTS:
            found[profile] = "unknown profile"
            continue
        _, size, digest = CHECKPOINTS[profile]
        p = Path(str(path))
        if not p.is_file():
            found[profile] = "missing"
        elif p.stat().st_size != size:
            found[profile] = f"size {p.stat().st_size} != {size}"
        elif verify and _sha256(p) != digest:
            found[profile] = "sha256 mismatch"
        else:
            found[profile] = "ok"
    add("checkpoints", "fail" if any(v != "ok" for v in found.values()) or not found else "ok", found)

    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        if not torch.cuda.is_available():
            add("cuda", "fail", "torch.cuda.is_available() is False")
        else:
            cap = torch.cuda.get_device_capability(0)
            free, total = torch.cuda.mem_get_info()
            add(
                "cuda",
                "ok",
                {
                    "device": torch.cuda.get_device_name(0),
                    "capability": f"{cap[0]}.{cap[1]}",
                    "free_gb": round(free / 2**30, 1),
                    "total_gb": round(total / 2**30, 1),
                },
            )
            if total < 40 * 2**30 and not req.get("offload_text_encoder", False):
                add("vram", "warn", "less than 40 GB VRAM: set offload_text_encoder: true (what upstream inference.py does)")
            try:
                import flash_attn

                q = torch.randn(64, 8, 64, device="cuda", dtype=torch.bfloat16)
                cu = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
                o = flash_attn.flash_attn_varlen_func(q, q, q, cu, cu, 64, 64)
                add("flash_attn_kernel", "ok" if bool(torch.isfinite(o).all()) else "fail", flash_attn.__version__)
            except Exception as exc:
                add("flash_attn_kernel", "fail", f"{type(exc).__name__}: {str(exc)[:300]} (upstream cross-attention requires flash-attn 2)")
    except Exception as exc:
        add("torch", "fail", f"{type(exc).__name__}: {exc}")
    for mod in ("diffusers", "transformers", "omegaconf", "av", "einops", "torchvision", "PIL"):
        try:
            m = __import__(mod)
            add(f"import:{mod}", "ok", getattr(m, "__version__", "?"))
        except Exception as exc:
            add(f"import:{mod}", "fail", f"{type(exc).__name__}: {str(exc)[:300]}")

    report["overall"] = "fail" if any(c["status"] == "fail" for c in checks) else ("warn" if any(c["status"] == "warn" for c in checks) else "ok")
    text = json.dumps(report, indent=1, ensure_ascii=False)
    tmp = req_path.parent / "doctor.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(req_path.parent / "doctor.json")
    print(text)
    return 0 if report["overall"] != "fail" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
