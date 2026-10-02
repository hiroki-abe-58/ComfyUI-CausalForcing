"""End-to-end check with a real runtime through ComfyUI's HTTP API (maintainer tool, not part of the node).

It starts its own ComfyUI on 127.0.0.1 (a free port, never 8188), queues API
workflows and checks:

  generate   Causal Forcing generation (2 videos) -> SaveVideo; the saved MP4s
             are fully decoded (81 frames, 832x480, 16 fps) and the report's
             generator forward counts match the model's schedule
  i2v        frame-wise image-to-video from a LoadImage node (optional)
  doctor     the Doctor node reports "ok"
  cancel     POST /interrupt during generation -> the job's runtime processes
             are gone and GPU memory returns to the idle level
  timeout    a runtime with timeout_minutes=1 -> execution error, processes gone
  error      a runtime with a missing checkpoint -> execution error, processes gone

Usage:
  python scripts/gpu_e2e.py --comfyui DIR --python EXE --config RUNTIMES.json
         --runtime ID [--profile cfpp_framewise_2step] [--image PNG] --out report.json
         [--steps generate,i2v,doctor,cancel,timeout,error]

The config must contain RUNTIME; the timeout/error steps derive temporary
runtimes from it (written next to --out). Only processes started by this
script are stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    if port == 8188:
        return free_port()
    return port


def http(base: str, path: str, data: dict | None = None, timeout: float = 30):
    if not base.startswith("http://127.0.0.1:"):
        raise ValueError("only the local test server is contacted")
    req = urllib.request.Request(base + path, data=None if data is None else json.dumps(data).encode(), headers={"Content-Type": "application/json"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - localhost only
        body = r.read()
    return json.loads(body) if body else None


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[0])


def wsl_processes(distro: str, needle: str) -> list[str]:
    """Command lines in the WSL distro that mention `needle` (a job directory name)."""
    out = subprocess.run(
        ["wsl.exe", "-d", distro, "--exec", "/bin/ps", "-eo", "pid,pgid,args"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return [line for line in out.stdout.splitlines() if needle in line and "/bin/ps" not in line]


def prompt_generate(
    runtime_id: str, profile: str, prompt: str, seed: int, videos: int = 1, prefix: str = "causalforcing_e2e", image: str | None = None
) -> dict:
    p = {
        "1": {"class_type": "CausalForcingRuntime", "inputs": {"runtime_id": runtime_id}},
        "2": {"class_type": "CausalForcingGenerate", "inputs": {"runtime": ["1", 0], "prompt": prompt, "seed": seed, "model": profile, "videos": videos}},
        "3": {"class_type": "SaveVideo", "inputs": {"video": ["2", 0], "filename_prefix": f"{prefix}/{profile}", "format": "mp4", "format.codec": "h264"}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 1]}},
    }
    if image:
        p["5"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        p["2"]["inputs"]["image"] = ["5", 0]
    return p


def prompt_doctor(runtime_id: str) -> dict:
    return {
        "1": {"class_type": "CausalForcingDoctor", "inputs": {"runtime_id": runtime_id, "verify_sha256": False}},
        "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
    }


class Comfy:
    def __init__(self, comfyui: Path, python: str, config: Path, workdir: Path):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.workdir = workdir
        env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY"))}
        env["COMFYUI_CAUSALFORCING_CONFIG"] = str(config)
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for sub in ("output", "temp", "user"):
            (workdir / sub).mkdir(exist_ok=True)
        self.log = open(workdir / "comfyui.log", "wb")  # noqa: SIM115
        argv = [python, str(comfyui / "main.py"), "--listen", "127.0.0.1", "--port", str(self.port), "--cpu", "--disable-auto-launch"]
        argv += ["--output-directory", str(workdir / "output"), "--temp-directory", str(workdir / "temp"), "--user-directory", str(workdir / "user")]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(argv, cwd=str(comfyui), stdout=self.log, stderr=subprocess.STDOUT, env=env, creationflags=flags)
        for _ in range(240):
            try:
                http(self.base, "/system_stats", timeout=5)
                return
            except (urllib.error.URLError, OSError):
                if self.proc.poll() is not None:
                    raise RuntimeError("ComfyUI exited during start-up; see comfyui.log") from None
                time.sleep(1)
        raise RuntimeError("ComfyUI did not start")

    def queue(self, prompt: dict) -> str:
        pid = str(uuid.uuid4())
        res = http(self.base, "/prompt", {"prompt": prompt, "client_id": "causalforcing-e2e", "prompt_id": pid})
        if res.get("node_errors"):
            raise RuntimeError(f"validation failed: {res}")
        return res["prompt_id"]

    def wait(self, pid: str, timeout: float) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            h = http(self.base, f"/history/{pid}")
            if h and pid in h and h[pid].get("status", {}).get("completed") is not None:
                st = h[pid]["status"]
                if st.get("status_str") in ("success", "error") or st.get("completed"):
                    return h[pid]
            time.sleep(1)
        raise TimeoutError(f"prompt {pid} did not finish in {timeout} s")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        self.log.close()


def newest_job_dir(jobs_root: Path, after: float) -> Path | None:
    """Newest job folder under jobs_root (ComfyUI nests its temp folder, so search a few levels)."""
    if not jobs_root.is_dir():
        return None
    cands = [p for pat in ("job-*", "*/job-*", "*/*/job-*") for p in jobs_root.glob(pat) if p.is_dir() and p.stat().st_mtime >= after]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def decode_check(path: Path, python_with_av: str) -> dict:
    code = (
        "import av,json,sys\n"
        "c=av.open(sys.argv[1]);s=c.streams.video[0];n=sum(1 for _ in c.decode(s))\n"
        "print(json.dumps({'frames':n,'width':s.codec_context.width,'height':s.codec_context.height,'rate':float(s.average_rate),'codec':s.codec_context.name}))"
    )
    out = subprocess.run([python_with_av, "-c", code, str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui", required=True, type=Path)
    ap.add_argument("--python", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--profile", default="cfpp_framewise_2step")
    ap.add_argument("--image", type=Path, help="PNG for the i2v step (copied into the input folder of the test ComfyUI)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", default="generate,i2v,doctor,cancel,timeout,error")
    args = ap.parse_args()
    steps = args.steps.split(",")
    work = args.out.parent / f"e2e-{time.strftime('%Y%m%d-%H%M%S')}"
    work.mkdir(parents=True)
    base_cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rt = base_cfg["runtimes"][args.runtime]
    jobs_root = Path(rt["jobs_dir"]) if rt.get("jobs_dir") else work / "temp"
    cfg = {"schema_version": 1, "runtimes": {args.runtime: rt}}
    cfg["runtimes"]["e2e-timeout"] = {**rt, "timeout_minutes": 1}
    cfg["runtimes"]["e2e-missing-checkpoint"] = {**rt, "checkpoints": {k: v + ".missing" for k, v in rt["checkpoints"].items()}}
    cfg_path = work / "causalforcing.runtimes.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    report: dict = {"port": None, "steps": {}}
    idle = gpu_used_mib()
    report["gpu_idle_mib_before"] = idle
    comfy = Comfy(args.comfyui, args.python, cfg_path, work)
    report["port"] = comfy.port
    try:
        info = http(comfy.base, "/object_info")
        report["nodes_registered"] = {n: n in info for n in ("CausalForcingRuntime", "CausalForcingGenerate", "CausalForcingDoctor")}
        report["runtime_choices"] = info["CausalForcingRuntime"]["input"]["required"]["runtime_id"][0]

        if "generate" in steps:
            t0 = time.time()
            pid = comfy.queue(
                prompt_generate(
                    args.runtime,
                    args.profile,
                    "A paper boat drifting down a narrow stream in a sunlit forest, gentle ripples, close-up tracking shot.",
                    7,
                    videos=2,
                )
            )
            h = comfy.wait(pid, 3600)
            outs = h.get("outputs", {})
            saved = [
                Path(work / "output" / v["subfolder"] / v["filename"])
                for o in outs.values()
                for v in o.get("images", []) + o.get("videos", [])
                if "filename" in v
            ]
            rep = json.loads(outs["4"]["text"][0]) if "4" in outs else None
            dec = [decode_check(p, args.python) for p in saved]
            fwd = [v["generator_forwards"] for v in rep["videos"]] if rep else []
            ok_fwd = bool(fwd) and all(
                f["denoise"] == f["expected"]["denoise"]
                and f["context"] == f["expected"]["context"]
                and f["first_block_denoise"] == f["expected"]["first_block_steps"]
                for f in fwd
            )
            report["steps"]["generate"] = {
                "status": h["status"].get("status_str"),
                "wall_s": round(time.time() - t0, 1),
                "saved": [p.name for p in saved],
                "decoded": dec,
                "decode_ok": len(dec) == 2 and all(d["frames"] == 81 and (d["width"], d["height"]) == (832, 480) and d["rate"] == 16.0 for d in dec),
                "forwards_ok": ok_fwd,
                "forwards": fwd,
                "report_excerpt": {k: rep.get(k) for k in ("model", "mode", "model_load_seconds", "wall_seconds", "upstream_commit")} if rep else None,
                "per_video_seconds": [v["seconds"] for v in rep["videos"]] if rep else None,
            }

        if "i2v" in steps and args.image:
            import shutil

            (args.comfyui / "input").mkdir(exist_ok=True)
            shutil.copyfile(args.image, args.comfyui / "input" / args.image.name)
            prompt = "The scene comes alive: gentle wind, drifting clouds, slow camera push-in."
            pid = comfy.queue(prompt_generate(args.runtime, args.profile, prompt, 21, image=args.image.name, prefix="causalforcing_e2e_i2v"))
            h = comfy.wait(pid, 3600)
            outs = h.get("outputs", {})
            saved = [
                Path(work / "output" / v["subfolder"] / v["filename"])
                for o in outs.values()
                for v in o.get("images", []) + o.get("videos", [])
                if "filename" in v
            ]
            rep = json.loads(outs["4"]["text"][0]) if "4" in outs else None
            dec = [decode_check(p, args.python) for p in saved]
            report["steps"]["i2v"] = {
                "status": h["status"].get("status_str"),
                "mode": rep and rep.get("mode"),
                "decoded": dec,
                "forwards": [v["generator_forwards"] for v in rep["videos"]] if rep else None,
                "seconds": [v["seconds"] for v in rep["videos"]] if rep else None,
            }

        if "doctor" in steps:
            pid = comfy.queue(prompt_doctor(args.runtime))
            h = comfy.wait(pid, 1800)
            doc = json.loads(h["outputs"]["2"]["text"][0]) if "2" in h.get("outputs", {}) else None
            report["steps"]["doctor"] = {
                "status": h["status"].get("status_str"),
                "overall": doc and doc.get("overall"),
                "checks": doc and [(c["name"], c["status"]) for c in doc["checks"]],
            }

        for step, rid, expect in (("cancel", args.runtime, "interrupted"), ("timeout", "e2e-timeout", "error"), ("error", "e2e-missing-checkpoint", "error")):
            if step not in steps:
                continue
            t0 = time.time()
            pid = comfy.queue(
                prompt_generate(
                    rid,
                    "cf_framewise_4step",
                    "A lighthouse on a cliff at dusk, waves below, slow pan.",
                    11,
                    videos=4 if step != "error" else 1,
                    prefix=f"causalforcing_e2e_{step}",
                )
            )
            job = None
            if step == "cancel":
                end = time.time() + 900
                while time.time() < end:
                    job = newest_job_dir(jobs_root, t0)
                    if job and (job / "events.jsonl").exists() and b'"progress"' in (job / "events.jsonl").read_bytes():
                        break
                    time.sleep(1)
                http(comfy.base, "/interrupt", {})
            h = comfy.wait(pid, 1800)
            job = job or newest_job_dir(jobs_root, t0)
            time.sleep(5)
            left = wsl_processes(rt["distro"], job.name) if job and rt.get("kind") == "wsl" else []
            msgs = [m for m in h["status"].get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
            gpu_after = gpu_used_mib()
            report["steps"][step] = {
                "status": h["status"].get("status_str"),
                "messages": [m[0] for m in msgs],
                "error_excerpt": next((m[1].get("exception_message", "")[:300] for m in msgs if m[0] == "execution_error"), None),
                "wall_s": round(time.time() - t0, 1),
                "job_dir": job.name if job else None,
                "stop_report": json.loads((job / "stop_report.json").read_text(encoding="utf-8")) if job and (job / "stop_report.json").exists() else None,
                "leftover_processes": left,
                "gpu_used_mib_after": gpu_after,
                "gpu_back_to_idle": gpu_after <= idle + 600,
                "expected": expect,
            }
    finally:
        comfy.stop()
        report["comfyui_stopped"] = comfy.proc.poll() is not None
        report["gpu_idle_mib_after"] = gpu_used_mib()
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
