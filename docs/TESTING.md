# Testing

## CPU tests (CI and local)

```sh
python -m pip install pytest==9.1.1 ruff==0.16.10
COMFYUI_PATH=/path/to/ComfyUI python -m pytest -q -rs
```

No GPU and no weights: `tests/fake_runtime/cf_job.py` stands in for the
upstream pipeline. It validates jobs with the real runner's `load_job`, writes
real 81-frame 832x480 MP4 files and can simulate failures and hangs (with a
grandchild process). Covered:

- runtime config validation (unknown keys, relative paths, `..`, distro names,
  environment allowlist, per-model checkpoint paths) and the config location;
- argv construction (fixed list, `wsl.exe --exec`, no shell), launcher
  environment without secrets (`WSLENV` emptied);
- job creation and the runner's own validation (tampered jobs rejected,
  image-to-video only for frame-wise models and only with the image file);
- the forward counts the runner enforces, derived from the official step
  lists (including the 4-step first frame of the 1/2-step models);
- result validation (path confinement, sha256, geometry);
- real subprocesses: success with progress, failure reporting, cancel and
  timeout. On Linux these also prove that the runner's process group,
  including a grandchild, is gone afterwards, that the kill helper refuses a
  foreign pid, stops orphans and stops a Doctor run;
- inside a real ComfyUI checkout (CPU): node registration without heavy
  imports, `validate_prompt`, `VIDEO` outputs decoded back (81 frames,
  16 fps, 832x480), the image input written as a PNG for image-to-video, and
  ComfyUI interrupts mapped to a job stop.

Tests that need a ComfyUI checkout are marked `comfy` and fail (not skip) when
`COMFYUI_PATH` is missing. Three process tests that read `/proc` are
POSIX-only; on Windows hosts that code runs inside WSL, and the tests run on
the Ubuntu CI leg (and were run in a WSL2 venv locally).

## GPU end-to-end (maintainer, real runtime)

`scripts/gpu_e2e.py` starts its own ComfyUI on a free `127.0.0.1` port and
uses the HTTP API with a real runtime:

| Step | Pass criteria |
| --- | --- |
| generate | 2 videos with the selected model -> `SaveVideo`; both MP4 files fully decoded (81 frames, 832x480, 16 fps); generator forward counts equal the model's schedule |
| i2v | frame-wise image-to-video from a `LoadImage` node; output decoded; forward counts include the image context pass |
| doctor | Doctor reports `ok` |
| cancel | `/interrupt` during generation -> `execution_interrupted`, no process of the job left in WSL, GPU memory back to idle |
| timeout | runtime with `timeout_minutes: 1` -> `execution_error`, nothing left |
| error | runtime with missing checkpoints -> `execution_error` with the runner's message, nothing left |

Last run for v0.1.0 (2026-10-03, RTX 5090, ComfyUI v0.38.0 on Windows 11,
node installed from `git archive`, 2-step model): all six steps passed, see
`docs/results/gpu_e2e_report.json`. The first Doctor run reported
`upstream_files: fail` because one pinned hash had been taken from a CRLF
checkout; the hash was fixed (all hashes are now compared with LF line
endings and were checked against the upstream commit) and the Doctor re-run
reported `ok`.

## Official reference

The published comparison also ran the official `inference.py` (unchanged)
for one prompt per model, with three numerics-neutral guards
(`weights_only=True`, the bf16 meta-device T5 build, and capturing the latents
and float frames it writes). The runtime of this package must produce the same
latents and frames for the same prompt and seed; see `docs/BENCHMARKS.md`.

## GUI check

`scripts/export_gui_workflows.py` (needs Playwright) loads each API workflow
into the real ComfyUI frontend, checks that every node type is known and that
`graphToPrompt()` reproduces it, and exports the UI-format files in
`workflows/`.
