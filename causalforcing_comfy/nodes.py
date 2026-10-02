"""ComfyUI nodes. Importing this module loads no CUDA, no model and no upstream code."""

from __future__ import annotations

import json
import time

from . import client
from .config import CONFIG_ENV, CONFIG_FILENAME, ConfigError, config_path, load_runtimes

NO_RUNTIME = "(no runtime configured)"
PROFILE_HELP = {
    "cfpp_framewise_2step": "Causal Forcing++ frame-wise 2-step (official checkpoint causal-forcing++/framewise-2step.pt, EMA): 2 denoising steps per frame, 4 for the first frame.",
    "cf_framewise_4step": "Causal Forcing frame-wise 4-step (official checkpoint framewise/causal_forcing.pt, EMA): the 4-step baseline.",
    "cfpp_framewise_1step": "Causal Forcing++ frame-wise 1-step (official checkpoint causal-forcing++/framewise-1step.pt, EMA): 1 step per frame, 4 for the first frame.",
    "cf_chunkwise_4step": "Causal Forcing chunk-wise 4-step (official checkpoint chunkwise/causal_forcing.pt, non-EMA weights as in the official command): text-to-video only.",
}


def _runtime_ids() -> list[str]:
    try:
        ids = sorted(load_runtimes())
    except ConfigError:
        ids = []
    return ids or [NO_RUNTIME]


def _resolve(runtime_id: str) -> client.Runtime:
    try:
        runtimes = load_runtimes()
    except ConfigError as exc:
        raise ValueError(f"Causal Forcing runtime config is invalid: {exc}") from exc
    if runtime_id not in runtimes:
        where = config_path()
        raise ValueError(
            f"Causal Forcing runtime {runtime_id!r} is not configured. An administrator registers runtimes in "
            f"{where if where else CONFIG_FILENAME} (or the file named by {CONFIG_ENV}); see the README."
        )
    return runtimes[runtime_id]


class CausalForcingRuntime:
    """Select an administrator-registered runtime (WSL2 / Linux environment with the upstream code and weights)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime_id": (_runtime_ids(), {"tooltip": f"Runtimes come from {CONFIG_FILENAME} (administrator config), never from the workflow."}),
            }
        }

    RETURN_TYPES = ("CAUSALFORCING_RUNTIME",)
    RETURN_NAMES = ("runtime",)
    FUNCTION = "select"
    CATEGORY = "CausalForcing"
    DESCRIPTION = "Pick a Causal Forcing runtime registered by the administrator."

    def select(self, runtime_id):
        rt = _resolve(runtime_id)
        return ({"runtime_id": rt.id},)


class CausalForcingGenerate:
    """Autoregressive text-to-video (and frame-wise image-to-video) with the official Causal Forcing (++) checkpoints."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime": ("CAUSALFORCING_RUNTIME",),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Used as written (no prompt rewriting)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
                "model": (list(client.PROFILES), {"default": "cfpp_framewise_2step", "tooltip": " | ".join(f"{k}: {v}" for k, v in PROFILE_HELP.items())}),
                "videos": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Videos in this job (seeds seed, seed+1, ...), one runtime process."}),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional first frame for image-to-video (frame-wise models only). Resized to 832x480 by the runtime, as in the official script."
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate"
    CATEGORY = "CausalForcing"
    DESCRIPTION = "832x480, 81 frames at 16 fps. Runs the official Causal Forcing pipeline in the runtime and returns real VIDEO outputs and a JSON report."

    def generate(self, runtime, prompt, seed, model, videos, image=None):
        import comfy.model_management as mm
        import comfy.utils
        from comfy_api.latest import InputImpl

        if not isinstance(runtime, dict) or "runtime_id" not in runtime:
            raise ValueError("connect a Causal Forcing Runtime node")
        rt = _resolve(runtime["runtime_id"])
        if model not in client.PROFILES:
            raise ValueError(f"model must be one of {client.PROFILES}")
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")
        arr = None
        if image is not None:
            if model not in client.FRAMEWISE:
                raise ValueError("image-to-video is only supported by the frame-wise models")
            import numpy as np

            first = image[0] if image.ndim == 4 else image
            arr = (first.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        specs = [{"prompt": prompt, "seed": int(seed) + i} for i in range(int(videos))]
        if specs[-1]["seed"] > 2**31 - 1:
            raise ValueError("seed + videos - 1 exceeds 2^31-1")
        per_video = client.FORWARDS[(model, "i2v" if arr is not None else "t2v")]
        pbar = comfy.utils.ProgressBar(per_video * len(specs))
        t0 = time.time()
        try:
            outcome = client.generate(
                rt,
                model,
                specs,
                image=arr,
                interrupted=mm.processing_interrupted,
                on_progress=lambda done, tot: pbar.update_absolute(done, tot),
            )
        except client.JobCancelled as exc:
            raise mm.InterruptProcessingException() from exc
        report = _report(outcome, rt, model, time.time() - t0)
        return ([InputImpl.VideoFromFile(str(p)) for p in outcome.videos], json.dumps(report, indent=1, ensure_ascii=False))


def _report(outcome: client.JobOutcome, rt, profile: str, wall_s: float) -> dict:
    r = outcome.result
    return {
        "runtime_id": rt.id,
        "model": profile,
        "model_note": PROFILE_HELP[profile],
        "mode": r.get("mode"),
        "job": outcome.job_dir.name,
        "wall_seconds": round(wall_s, 2),
        "model_load_seconds": r.get("model_load_seconds"),
        "upstream_commit": r.get("upstream_commit"),
        "effective_config": r.get("effective_config"),
        "weights": r.get("weights"),
        "versions": r.get("versions"),
        "videos": [
            {
                k: v[k]
                for k in (
                    "seed",
                    "rgb_frames",
                    "width",
                    "height",
                    "fps",
                    "seconds",
                    "generator_forwards",
                    "cuda_max_allocated_bytes",
                    "sha256",
                    "latents_sha256",
                )
                if k in v
            }
            for v in r.get("videos", [])
        ],
    }


class CausalForcingDoctor:
    """Check a runtime: upstream files at the pinned commit, model and checkpoint files, CUDA and flash-attn."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime_id": (_runtime_ids(),),
                "verify_sha256": ("BOOLEAN", {"default": False, "tooltip": "Hash every model and checkpoint file (tens of GB of reads)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check"
    CATEGORY = "CausalForcing"
    OUTPUT_NODE = True
    DESCRIPTION = "Diagnose a Causal Forcing runtime without generating a video."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def check(self, runtime_id, verify_sha256):
        import comfy.model_management as mm

        where = config_path()
        try:
            runtimes = load_runtimes()
        except ConfigError as exc:
            report = {"overall": "fail", "config": str(where), "error": str(exc)}
            return {"ui": {"text": [f"config error: {exc}"]}, "result": (json.dumps(report, indent=1),)}
        if runtime_id not in runtimes:
            report = {
                "overall": "fail",
                "config": str(where),
                "error": f"no runtime {runtime_id!r}; create {CONFIG_FILENAME} (see examples/ and the README)",
                "configured": sorted(runtimes),
            }
            return {"ui": {"text": [report["error"]]}, "result": (json.dumps(report, indent=1),)}
        try:
            report = client.doctor(runtimes[runtime_id], verify_sha256=verify_sha256, interrupted=mm.processing_interrupted)
        except client.JobCancelled as exc:
            raise mm.InterruptProcessingException() from exc
        lines = [f"{runtime_id}: {report.get('overall')}"] + [f"{c['status']:>4}  {c['name']}" for c in report.get("checks", [])]
        return {"ui": {"text": ["\n".join(lines)]}, "result": (json.dumps(report, indent=1, ensure_ascii=False),)}


NODE_CLASS_MAPPINGS = {
    "CausalForcingRuntime": CausalForcingRuntime,
    "CausalForcingGenerate": CausalForcingGenerate,
    "CausalForcingDoctor": CausalForcingDoctor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CausalForcingRuntime": "Causal Forcing Runtime",
    "CausalForcingGenerate": "Causal Forcing Generate (T2V / I2V)",
    "CausalForcingDoctor": "Causal Forcing Doctor",
}
