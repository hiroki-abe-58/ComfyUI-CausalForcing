"""Run one structured Causal Forcing / Causal Forcing++ generation job (runtime side, never inside ComfyUI).

ComfyUI writes a job JSON into a fresh job directory and starts:

    <runtime python> cf_job.py [--watch-stdin] <job.json>

The upstream code (thu-ml/Causal-Forcing, pinned commit) is imported unmodified. This file mirrors the
official inference.py for one prompt and adds:

- schema/type/limit checks of the job,
- every ``torch.load`` forced to ``weights_only=True`` (upstream loads T5 with weights_only=False),
- explicit selection of ``generator_ema`` / ``generator`` per profile (the official README commands),
  explicit FSDP-prefix mapping and a strict load that must cover every generator tensor,
- the official seeding (``set_seed(seed)`` right before the noise is drawn),
- per-kind generator forward counters (denoising steps, first-frame steps, clean-context KV updates),
  checked against the profile's step lists,
- timings per phase and CUDA allocator peaks, MP4 + metadata confined to the job directory.

Exit codes: 0 success, 2 invalid job, 3 runtime failure, 4 cancelled.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

SCHEMA_VERSION = 1
UPSTREAM_COMMIT = "da3ddf1a590f30c9edc568e11c2f77170995c31c"
# profile id -> upstream config, which checkpoint key the official README uses, frame-wise or not
PROFILES = {
    "cfpp_framewise_2step": {"config": "causal_forcing_dmd_framewise_2step.yaml", "use_ema": True, "framewise": True},
    "cf_framewise_4step": {"config": "causal_forcing_dmd_framewise.yaml", "use_ema": True, "framewise": True},
    "cfpp_framewise_1step": {"config": "causal_forcing_dmd_framewise_1step.yaml", "use_ema": True, "framewise": True},
    "cf_chunkwise_4step": {"config": "causal_forcing_dmd_chunkwise.yaml", "use_ema": False, "framewise": False},
}
LATENT_FRAMES = 21  # 21 latent frames -> 1 + 4 * 20 = 81 RGB frames (Wan VAE temporal stride 4)
LATENT_SHAPE = (16, 60, 104)  # channels, 480/8, 832/8
MAX_VIDEOS = 8
MAX_PROMPT_CHARS = 2000
ENV_KEYS = {"CC", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME", "HF_HOME", "CUDA_VISIBLE_DEVICES"}
SECRET_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|CREDENTIAL|WANDB)", re.I)
FSDP_PREFIX = "model._fsdp_wrapped_module."
IMAGE_NAME = "input.png"


class JobError(ValueError):
    pass


class JobCancelled(RuntimeError):
    pass


_EVENTS_PATH: Path | None = None


def _log(event: str, **fields) -> None:
    """One JSON object per line on stdout and, once the job directory is known, in events.jsonl."""
    line = json.dumps({"event": event, "t": round(time.time(), 3), **fields}, ensure_ascii=False)
    print(line, flush=True)
    if _EVENTS_PATH is not None:
        with open(_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_job(path: Path) -> dict:
    """Parse and validate the job file. Raises JobError."""
    if path.stat().st_size > 256 * 1024:
        raise JobError("job file too large")
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JobError(f"job is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(job, dict) or job.get("schema_version") != SCHEMA_VERSION:
        raise JobError("unsupported job schema")
    allowed = {"schema_version", "job_id", "profile", "mode", "videos", "upstream_dir", "models_dir", "checkpoint", "fps", "env", "offload_text_encoder"}
    extra = set(job) - allowed
    if extra:
        raise JobError(f"unknown job keys: {sorted(extra)}")
    if not isinstance(job.get("job_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", job["job_id"]):
        raise JobError("invalid job_id")
    if job.get("profile") not in PROFILES:
        raise JobError(f"profile must be one of {sorted(PROFILES)}")
    mode = job.get("mode", "t2v")
    if mode not in ("t2v", "i2v"):
        raise JobError("mode must be t2v or i2v")
    if mode == "i2v" and not PROFILES[job["profile"]]["framewise"]:
        raise JobError("image-to-video is only supported by the frame-wise models (official README)")
    videos = job.get("videos")
    if not isinstance(videos, list) or not 1 <= len(videos) <= MAX_VIDEOS:
        raise JobError(f"videos must be a list of 1..{MAX_VIDEOS} items")
    for i, v in enumerate(videos):
        if not isinstance(v, dict) or set(v) - {"prompt", "seed"}:
            raise JobError(f"video {i}: expected {{prompt, seed}}")
        p, s = v.get("prompt"), v.get("seed")
        if not isinstance(p, str) or not p.strip() or len(p) > MAX_PROMPT_CHARS or "\x00" in p:
            raise JobError(f"video {i}: prompt must be 1..{MAX_PROMPT_CHARS} characters")
        if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s <= 2**31 - 1:
            raise JobError(f"video {i}: seed must be an integer in [0, 2^31-1]")
    for key in ("upstream_dir", "models_dir", "checkpoint"):
        if not isinstance(job.get(key), str) or not job[key]:
            raise JobError(f"{key} must be a non-empty string")
    if job.get("fps", 16) not in (16,):
        raise JobError("fps must be 16 (the upstream output rate)")
    env = job.get("env", {})
    if not isinstance(env, dict) or set(env) - ENV_KEYS or not all(isinstance(v, str) for v in env.values()):
        raise JobError(f"env may only contain {sorted(ENV_KEYS)} as strings")
    if not isinstance(job.get("offload_text_encoder", False), bool):
        raise JobError("offload_text_encoder must be a boolean")
    if mode == "i2v":
        img = path.parent / IMAGE_NAME
        if not img.is_file() or img.stat().st_size > 64 * 1024 * 1024:
            raise JobError(f"image-to-video needs {IMAGE_NAME} (at most 64 MiB) in the job directory")
    return job


def _install_cancel_watch(job_dir: Path, watch_stdin: bool) -> None:
    """Exit promptly if a CANCEL file appears or (with --watch-stdin) the parent closes our stdin."""

    def stdin_eof():
        try:
            while os.read(0, 4096):  # raw fd: no buffered-reader lock to trip interpreter shutdown
                pass
        except OSError:
            pass
        if not (job_dir / "result.json").exists():
            _log("cancelled", reason="parent closed stdin")
            _stop_own_group()
            os._exit(4)

    def cancel_file():
        while True:
            if (job_dir / "CANCEL").exists():
                _log("cancelled", reason="cancel file")
                _stop_own_group()
                os._exit(4)
            time.sleep(0.5)

    if watch_stdin:
        threading.Thread(target=stdin_eof, daemon=True).start()
    threading.Thread(target=cancel_file, daemon=True).start()


def _proc_stat(pid: int) -> list[str] | None:
    """Fields of /proc/<pid>/stat after the command name (index 0 = state, 2 = pgrp, 3 = session, 19 = starttime)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    return stat[stat.rfind(")") + 2 :].split()


def _stop_own_group() -> None:
    """SIGKILL every other member of this runner's process group, only when the runner leads its own session."""
    me = os.getpid()
    if not hasattr(os, "getsid") or os.getpgid(0) != me or os.getsid(0) != me or not os.path.isdir("/proc"):
        return
    for entry in os.listdir("/proc"):
        if entry.isdigit() and int(entry) != me:
            st = _proc_stat(int(entry))
            if st and int(st[2]) == me:
                try:
                    os.kill(int(entry), 9)
                except OSError:
                    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # Linux reports KiB
    except Exception:
        return None


def apply_env(env: dict) -> None:
    for key in list(os.environ):
        if SECRET_RE.search(key):
            del os.environ[key]
    os.environ.update(env)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_MODE"] = "disabled"


def become_session_leader(pid_file: Path) -> None:
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    me = _proc_stat(os.getpid())
    record = {"pid": os.getpid(), "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None, "starttime": int(me[19]) if me else None}
    pid_file.write_text(json.dumps(record), encoding="utf-8")


def expected_forwards(config, mode: str) -> dict:
    """Generator forwards the official pipeline makes for one 21-latent-frame video."""
    nfpb = int(config.num_frame_per_block)
    steps = len(config.denoising_step_list)
    first = config.get("denoising_step_list_first_chunk")
    first_steps = len(first) if first is not None else steps
    gen_frames = LATENT_FRAMES - (1 if mode == "i2v" else 0)
    blocks = gen_frames // nfpb
    return {
        "denoise": first_steps + (blocks - 1) * steps,
        "context": blocks + (1 if mode == "i2v" else 0),  # one clean-context KV update per block (+ the input image)
        "blocks": blocks,
        "first_block_steps": first_steps,
        "steps_per_block": steps,
    }


_PATCHED = {"done": False}


class Engine:
    """Everything done once per process: guards, pipeline, weights, device placement, timers."""

    def __init__(self, upstream: Path, models: Path, ckpt: Path, profile: str, offload: bool, on_progress=None):
        for p, what in (
            (upstream / "pipeline" / "causal_inference.py", "upstream checkout"),
            (models / "wan_models" / "Wan2.1-T2V-1.3B" / "config.json", "Wan2.1-T2V-1.3B"),
            (ckpt, "checkpoint"),
        ):
            if not p.is_file():
                raise FileNotFoundError(f"{what} not found")
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        os.chdir(models)  # upstream resolves wan_models/... relative to the working directory
        self.upstream, self.ckpt, self.profile, self.offload = upstream, ckpt, profile, bool(offload)
        self.on_progress = on_progress
        self.should_cancel = None
        self._install_patches()

        import torch
        from omegaconf import OmegaConf
        from pipeline import CausalInferencePipeline

        self.torch = torch
        spec = PROFILES[profile]
        config = OmegaConf.merge(OmegaConf.load(upstream / "configs" / "default_config.yaml"), OmegaConf.load(upstream / "configs" / spec["config"]))
        if not hasattr(config, "denoising_step_list"):
            raise RuntimeError("profile config has no denoising_step_list (few-step pipeline expected)")
        self.config = config
        torch.set_grad_enabled(False)
        device = torch.device("cuda")
        self.device = device
        t0 = time.time()
        pipeline = CausalInferencePipeline(config, device=device)
        meta_left = [n for n, p in list(pipeline.named_parameters()) + list(pipeline.named_buffers()) if p.is_meta]
        if meta_left:
            raise RuntimeError(f"{len(meta_left)} tensors were never loaded, e.g. {meta_left[:3]}")
        state = torch.load(str(ckpt), map_location="cpu")
        key = "generator_ema" if spec["use_ema"] else "generator"
        if key not in state:
            raise RuntimeError(f"checkpoint has no {key!r} (keys: {sorted(state)[:8]})")
        raw = state[key]
        mapped = {(k.replace(FSDP_PREFIX, "model.", 1) if k.startswith(FSDP_PREFIX) else k): v for k, v in raw.items()}
        gen_sd = pipeline.generator.state_dict()
        missing = sorted(set(gen_sd) - set(mapped))
        unexpected = sorted(set(mapped) - set(gen_sd))
        shape_bad = [k for k, v in mapped.items() if k in gen_sd and tuple(gen_sd[k].shape) != tuple(v.shape)]
        if missing or unexpected or shape_bad:
            raise RuntimeError(
                f"checkpoint does not match the generator: missing={missing[:3]} ({len(missing)}), unexpected={unexpected[:3]} ({len(unexpected)}), shape={shape_bad[:3]}"
            )
        probe_key = next(k for k in mapped if k.endswith("blocks.0.self_attn.q.weight"))
        before = gen_sd[probe_key].float().clone()
        result = pipeline.generator.load_state_dict(mapped, strict=True)
        after = pipeline.generator.state_dict()[probe_key].float()
        self.load_report = {
            "checkpoint_bytes": ckpt.stat().st_size,
            "checkpoint_keys": sorted(state)[:8],
            "used_key": key,
            "tensors": len(mapped),
            "generator_tensors": len(gen_sd),
            "parameters": int(sum(v.numel() for v in mapped.values())),
            "fsdp_prefixed_keys_mapped": sum(1 for k in raw if k.startswith(FSDP_PREFIX)),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "probe_tensor": probe_key,
            "probe_changed_from_base": bool(not torch.equal(before, after)),
            "probe_equals_checkpoint": bool(torch.equal(after, mapped[probe_key].float())),
        }
        if not (self.load_report["probe_changed_from_base"] and self.load_report["probe_equals_checkpoint"]):
            raise RuntimeError("checkpoint weights were not applied to the generator")
        del state, raw, mapped
        pipeline = pipeline.to(dtype=torch.bfloat16)
        if self.offload:
            from demo_utils.memory import DynamicSwapInstaller, gpu

            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)  # what inference.py does below 40 GB
        else:
            pipeline.text_encoder.to(device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
        torch.cuda.synchronize()
        self.load_seconds = time.time() - t0
        self.pipeline = pipeline
        self.effective = {
            "config_file": spec["config"],
            "checkpoint_key": key,
            "denoising_step_list": [int(x) for x in config.denoising_step_list],
            "denoising_step_list_first_chunk": [int(x) for x in config.denoising_step_list_first_chunk]
            if config.get("denoising_step_list_first_chunk") is not None
            else None,
            "warped_steps": [round(float(x), 3) for x in pipeline.denoising_step_list],
            "warped_first_chunk": [round(float(x), 3) for x in pipeline.denoising_step_list_first_chunk]
            if pipeline.denoising_step_list_first_chunk is not None
            else None,
            "warp_denoising_step": bool(config.warp_denoising_step),
            "num_frame_per_block": int(config.num_frame_per_block),
            "independent_first_frame": bool(config.independent_first_frame),
            "context_noise": int(config.context_noise),
            "timestep_shift": float(config.model_kwargs.timestep_shift),
            "offload_text_encoder": self.offload,
        }
        self.phase = {}
        self.fwd = {}
        self.current = {"video": 0}
        pipeline.text_encoder.forward = self._timed(pipeline.text_encoder.forward, "text_encode")
        pipeline.generator.forward = self._timed(pipeline.generator.forward, "generator", counter=True)
        pipeline.vae.decode_to_pixel = self._timed(pipeline.vae.decode_to_pixel, "vae_decode")
        pipeline.vae.encode_to_latent = self._timed(pipeline.vae.encode_to_latent, "vae_encode")
        self.videos_generated = 0

    @staticmethod
    def _install_patches() -> None:
        if _PATCHED["done"]:
            return
        import torch

        _orig_load = torch.load

        def _safe_load(*args, **kwargs):
            kwargs["weights_only"] = True  # never unpickle code objects (upstream passes weights_only=False for T5)
            kwargs.setdefault("mmap", True)
            return _orig_load(*args, **kwargs)

        torch.load = _safe_load
        import utils.wan_wrapper as wan_wrapper

        # Upstream builds UMT5-XXL in fp32 on the CPU (~23 GB) and loads the bf16 file into it; the pipeline is
        # cast to bf16 right after. Build it on the meta device and adopt the bf16 tensors: bit-identical weights.
        _orig_umt5 = wan_wrapper.umt5_xxl

        def _umt5_meta(**kwargs):
            kwargs.update(device="meta", dtype=torch.bfloat16)
            model = _orig_umt5(**kwargs)
            model.load_state_dict = lambda sd, strict=True: torch.nn.Module.load_state_dict(model, sd, strict=strict, assign=True)
            return model

        wan_wrapper.umt5_xxl = _umt5_meta
        _PATCHED["done"] = True

    def versions(self) -> dict:
        import flash_attn

        torch = self.torch
        return {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "flash_attn": flash_attn.__version__,
        }

    def _timed(self, fn, key, counter=False):
        torch = self.torch

        def inner(*a, **k):
            if self.should_cancel is not None and self.should_cancel():
                raise JobCancelled(f"cancelled before {key}")
            kind = None
            if counter:
                ts = k.get("timestep")
                kind = "context" if ts is not None and int(ts.flatten()[0]) == int(self.config.context_noise) else "denoise"
                self.fwd[kind] = self.fwd.get(kind, 0) + 1
                if kind == "denoise" and self.fwd.get("context", 0) == self.first_block_context:
                    self.fwd["first_block_denoise"] = self.fwd.get("first_block_denoise", 0) + 1
            torch.cuda.synchronize()
            s = time.time()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            dt = time.time() - s
            self.phase[key] = self.phase.get(key, 0.0) + dt
            if counter:
                self.phase[f"generator_{kind}"] = self.phase.get(f"generator_{kind}", 0.0) + dt
                if kind == "denoise" and self.fwd.get("context", 0) == self.first_block_context:
                    self.phase["first_block_denoise"] = self.phase.get("first_block_denoise", 0.0) + dt
                if self.on_progress is not None:
                    self.on_progress(self.current["video"], self.fwd.get("denoise", 0) + self.fwd.get("context", 0))
            return out

        return inner

    def total_forwards(self, mode: str) -> int:
        e = expected_forwards(self.config, mode)
        return e["denoise"] + e["context"]

    def generate(self, index: int, video: dict, out_dir: Path, fps: int, mode: str = "t2v", image_path: Path | None = None) -> dict:
        import av
        from utils.misc import set_seed

        torch = self.torch
        pipeline = self.pipeline
        self.current["video"] = index
        self.phase = {}
        self.fwd = {}
        # the first generated block comes after the context forward of the input image in I2V
        self.first_block_context = 1 if mode == "i2v" else 0
        torch.cuda.reset_peak_memory_stats()
        initial_latent = None
        s = time.time()
        if mode == "i2v":
            from PIL import Image
            from torchvision import transforms

            transform = transforms.Compose([transforms.Resize((480, 832)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            img = transform(Image.open(image_path).convert("RGB"))  # same transform as the official inference.py
            image = img.unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=torch.bfloat16)
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=self.device, dtype=torch.bfloat16)
        set_seed(video["seed"])  # random, numpy, torch (CPU + CUDA): as the official script, right before the noise
        frames_n = LATENT_FRAMES - (1 if mode == "i2v" else 0)
        noise = torch.randn([1, frames_n, *LATENT_SHAPE], device=self.device, dtype=torch.bfloat16)
        noise_sha = hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        s = time.time()
        try:
            frames, latents = pipeline.inference(noise=noise, text_prompts=[video["prompt"]], return_latents=True, initial_latent=initial_latent)
        finally:
            pipeline.vae.model.clear_cache()
        torch.cuda.synchronize()
        t_inf = time.time() - s
        if not (frames.isfinite().all() and latents.isfinite().all()):
            raise RuntimeError(f"video {index}: non-finite output")
        expect = expected_forwards(self.config, mode)
        got = {"denoise": self.fwd.get("denoise", 0), "context": self.fwd.get("context", 0), "first_block_denoise": self.fwd.get("first_block_denoise", 0)}
        if got["denoise"] != expect["denoise"] or got["context"] != expect["context"] or got["first_block_denoise"] != expect["first_block_steps"]:
            raise RuntimeError(f"generator forward count {got} does not match the profile's schedule {expect}")
        # same conversion as the official script: 255 * video, cast to uint8 by torchvision.write_video (truncation)
        rgb = (255.0 * frames[0]).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        latent_sha = hashlib.sha256(latents.float().cpu().numpy().tobytes()).hexdigest()
        frames_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        del frames, latents
        s = time.time()
        mp4 = out_dir / f"{index:02d}.mp4"
        with av.open(str(mp4), "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = rgb.shape[2], rgb.shape[1], "yuv420p"
            stream.options = {"crf": "18"}
            for f in rgb:
                for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        t_save = time.time() - s
        ph = self.phase
        self.videos_generated += 1
        return {
            "index": index,
            "file": f"videos/{mp4.name}",
            "sha256": _sha256(mp4),
            "mode": mode,
            "seed": video["seed"],
            "prompt_sha256": hashlib.sha256(video["prompt"].encode("utf-8")).hexdigest(),
            "image_sha256": _sha256(image_path) if image_path else None,
            "initial_noise_sha256": noise_sha,
            "latents_sha256": latent_sha,
            "rgb_frames_sha256": frames_sha,
            "latent_frames": LATENT_FRAMES,
            "rgb_frames": int(rgb.shape[0]),
            "height": int(rgb.shape[1]),
            "width": int(rgb.shape[2]),
            "fps": fps,
            "seconds": {
                "inference_total": round(t_inf, 3),
                "text_encode": round(ph.get("text_encode", 0.0), 3),
                "vae_encode_image": round(ph.get("vae_encode", 0.0), 3),
                "generator_forwards": round(ph.get("generator", 0.0), 3),
                "generator_denoise": round(ph.get("generator_denoise", 0.0), 3),
                "generator_context": round(ph.get("generator_context", 0.0), 3),
                "first_block_denoise": round(ph.get("first_block_denoise", 0.0), 3),
                "vae_decode": round(ph.get("vae_decode", 0.0), 3),
                "mp4_write": round(t_save, 3),
            },
            "generator_forwards": {**got, "expected": expect},
            "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_used_bytes_after": int(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]),
        }


def write_result(job_dir: Path, result: dict) -> None:
    tmp = job_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(job_dir / "result.json")


def _run(job: dict, job_dir: Path) -> int:
    t_proc = time.time()
    engine = Engine(
        Path(job["upstream_dir"]).resolve(),
        Path(job["models_dir"]).resolve(),
        Path(job["checkpoint"]).resolve(),
        job["profile"],
        bool(job.get("offload_text_encoder", False)),
    )
    mode = job.get("mode", "t2v")
    total = engine.total_forwards(mode)
    engine.on_progress = lambda video, forwards: _log("progress", video=video, forwards=forwards, forwards_per_video=total)
    _log("loaded", seconds=round(engine.load_seconds, 2), forwards_per_video=total)
    out_dir = job_dir / "videos"
    out_dir.mkdir(exist_ok=False)
    image = job_dir / IMAGE_NAME if mode == "i2v" else None
    records = []
    for idx, video in enumerate(job["videos"]):
        rec = engine.generate(idx, video, out_dir, job.get("fps", 16), mode=mode, image_path=image)
        records.append(rec)
        _log("video_done", index=idx, seconds=rec["seconds"], forwards=rec["generator_forwards"])
    write_result(
        job_dir,
        {
            "schema_version": SCHEMA_VERSION,
            "job_id": job["job_id"],
            "profile": job["profile"],
            "mode": mode,
            "backend": "one-shot",
            "upstream_commit": UPSTREAM_COMMIT,
            "effective_config": engine.effective,
            "weights": engine.load_report,
            "versions": engine.versions(),
            "model_load_seconds": round(engine.load_seconds, 3),
            "process_seconds": round(time.time() - t_proc, 3),
            "host_peak_rss_bytes": _peak_rss_bytes(),
            "videos": records,
        },
    )
    _log("done", job_id=job["job_id"])
    return 0


def main(argv: list[str]) -> int:
    global _EVENTS_PATH
    sys.dont_write_bytecode = True  # never leave __pycache__ in the upstream checkout
    args = argv[1:]
    watch_stdin = "--watch-stdin" in args
    args = [a for a in args if a != "--watch-stdin"]
    if len(args) != 1:
        print("usage: cf_job.py [--watch-stdin] <job.json>", file=sys.stderr)
        return 2
    job_path = Path(args[0]).resolve()
    job_dir = job_path.parent
    _EVENTS_PATH = job_dir / "events.jsonl"
    try:
        job = load_job(job_path)
    except (JobError, OSError) as exc:
        _log("invalid_job", error=str(exc))
        return 2
    apply_env(job.get("env", {}))
    become_session_leader(job_dir / "runner.pid")
    _install_cancel_watch(job_dir, watch_stdin)
    _log("start", job_id=job["job_id"], profile=job["profile"], mode=job.get("mode", "t2v"), videos=len(job["videos"]))
    try:
        return _run(job, job_dir)
    except Exception as exc:
        import traceback

        _log("failed", error_type=type(exc).__name__, error=str(exc)[:2000], traceback=traceback.format_exc()[-4000:])
        return 3


if __name__ == "__main__":
    code = main(sys.argv)
    sys.stdout.flush()
    sys.stderr.flush()
    _stop_own_group()
    os._exit(code)
