"""Stand-in for runtime/cf_job.py used by the CPU tests (no torch, no GPU).

It accepts the same command line and job file, validates the job with the real
runner's ``load_job``, and then behaves according to the prompt:

- "MODE:ok"      progress events, an 81-frame 832x480 MP4 per video, result.json
- "MODE:fail"    a "failed" event and exit code 3
- "MODE:hang"    (after one progress event) starts a grandchild and sleeps forever
- "MODE:badfile" a result.json whose video path points outside videos/
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
REAL = Path(__file__).resolve().parents[2] / "runtime"
sys.path.insert(0, str(REAL))
from cf_job import _proc_stat, _stop_own_group, load_job  # noqa: E402


def log(job_dir: Path, event: str, **fields) -> None:
    line = json.dumps({"event": event, "t": time.time(), **fields})
    print(line, flush=True)
    with open(job_dir / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_mp4(path: Path, seed: int) -> None:
    import av
    import numpy as np

    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=16)
        s.width, s.height, s.pix_fmt = 832, 480, "yuv420p"
        s.options = {"crf": "30", "preset": "ultrafast"}
        for i in range(81):
            img = np.full((480, 832, 3), (seed * 37 + i * 3) % 256, dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--watch-stdin"]
    job_path = Path(args[0]).resolve()
    job_dir = job_path.parent
    job = load_job(job_path)
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:  # already a session leader (the client used start_new_session)
            pass
    pgid = os.getpgid(0) if hasattr(os, "getpgid") else os.getpid()
    st = _proc_stat(os.getpid())
    (job_dir / "runner.pid").write_text(json.dumps({"pid": os.getpid(), "pgid": pgid, "starttime": int(st[19]) if st else None}), encoding="utf-8")
    # Like the real runner, which always runs on Linux. On a Windows host a thread blocked in a pipe read
    # stalls the encoder's own I/O, so there the fake relies on the CANCEL file alone.
    if "--watch-stdin" in sys.argv and os.name == "posix":

        def eof():
            try:
                while os.read(0, 4096):
                    pass
            except OSError:
                pass
            log(job_dir, "cancelled", reason="parent closed stdin")
            _stop_own_group()
            os._exit(4)

        threading.Thread(target=eof, daemon=True).start()
    mode = job["videos"][0]["prompt"].split()[0]
    log(job_dir, "start", job_id=job["job_id"], profile=job["profile"], videos=len(job["videos"]))
    if mode == "MODE:fail":
        log(job_dir, "failed", error_type="RuntimeError", error="simulated failure")
        return 3
    if mode == "MODE:hang":
        log(job_dir, "progress", video=0, forwards=1, forwards_per_video=35)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])  # noqa: S603
        (job_dir / "grandchild.pid").write_text(str(child.pid), encoding="utf-8")
        time.sleep(3600)
        return 0
    (job_dir / "videos").mkdir()
    records = []
    for i, v in enumerate(job["videos"]):
        for k in range(1, 36):
            if k % 7 == 0:
                log(job_dir, "progress", video=i, forwards=k, forwards_per_video=35)
        mp4 = job_dir / "videos" / f"{i:02d}.mp4"
        write_mp4(mp4, v["seed"])
        rel = f"videos/{i:02d}.mp4" if mode != "MODE:badfile" else "../job.json"
        records.append(
            {
                "index": i,
                "file": rel,
                "sha256": hashlib.sha256(mp4.read_bytes()).hexdigest(),
                "seed": v["seed"],
                "rgb_frames": 81,
                "height": 480,
                "width": 832,
                "fps": 16,
                "seconds": {"inference_total": 0.1},
                "mode": job.get("mode", "t2v"),
                "generator_forwards": {"denoise": 44, "context": 21, "first_block_denoise": 4},
            }
        )
    result = {"schema_version": 1, "job_id": job["job_id"], "profile": job["profile"], "mode": job.get("mode", "t2v"), "videos": records, "fake_runtime": True}
    (job_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    log(job_dir, "done", job_id=job["job_id"])
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    _stop_own_group()
    os._exit(code)
