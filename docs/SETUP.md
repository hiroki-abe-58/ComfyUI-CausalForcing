# Setting up a runtime

The nodes run the official Causal Forcing pipeline in a separate Python
environment (the *runtime*), so nothing is installed into ComfyUI's own
Python.

| ComfyUI host | Runtime | Config `kind` | Tested with real weights |
| --- | --- | --- | --- |
| Windows 10/11 | a WSL2 Linux distribution on the same PC | `wsl` | yes (Windows 11 + WSL2 Ubuntu 24.04, RTX 5090) |
| Linux | a venv on the same machine | `posix` | CPU tests only |

Native Windows runtimes are not supported (the upstream cross-attention needs
flash-attn 2, which this project has not verified on Windows).

Below, `/opt/causalforcing` is an example location on the Linux side.

## 1. GPU

`nvidia-smi` must work inside the distribution (on WSL2 this comes from the
Windows NVIDIA driver). The reference run used driver 595.95 and CUDA 12.8
wheels on an RTX 5090 (compute capability 12.0).

## 2. Python environment

Python 3.10 (the upstream code pins `numpy==1.24.4`). The upstream README says
the inference environment is the one of Self-Forcing; the reference runtime
used torch 2.8 because official flash-attn 2.8.3 wheels exist for it:

```sh
uv venv --python 3.10 /opt/causalforcing/venv
uv pip install --python /opt/causalforcing/venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128
# flash-attn 2.8.3: download the wheel from the GitHub release listed in requirements-reference.txt,
# check its sha256, install the local file
uv pip install --python /opt/causalforcing/venv/bin/python ./flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
uv pip install --python /opt/causalforcing/venv/bin/python -r requirements-reference.txt
```

Only the packages needed for inference are listed (the upstream
`requirements.txt` also covers training: wandb, lmdb, onnx, ...).

## 3. Upstream code (pinned)

```sh
git clone https://github.com/thu-ml/Causal-Forcing /opt/causalforcing/Causal-Forcing
git -C /opt/causalforcing/Causal-Forcing checkout da3ddf1a590f30c9edc568e11c2f77170995c31c
```

Do not `pip install` it; the runner imports it from this folder unmodified.

## 4. Weights (pinned revisions)

```sh
cd /opt/causalforcing/models
hf download Wan-AI/Wan2.1-T2V-1.3B --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --local-dir wan_models/Wan2.1-T2V-1.3B \
  config.json diffusion_pytorch_model.safetensors models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth \
  google/umt5-xxl/special_tokens_map.json google/umt5-xxl/spiece.model \
  google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json LICENSE.txt README.md
hf download zhuhz22/Causal-Forcing --revision 2f8eb8bb6eeb1238da9d13e5420d342a74d634a6 --local-dir checkpoints \
  causal-forcing++/framewise-2step.pt framewise/causal_forcing.pt
```

About 17 GB for Wan (only the 1.3B model is needed for inference; the
upstream README also downloads the 14B model, which is used only for
training) and 5.7 GB per Causal Forcing checkpoint. Run the Doctor node with
`verify_sha256` once to compare every file with the hashes in
`runtime/cf_doctor.py`.

## 5. Register the runtime in ComfyUI

Copy `examples/causalforcing.runtimes.example.json` to
`<ComfyUI>/user/causalforcing.runtimes.json` (or point the environment
variable `COMFYUI_CAUSALFORCING_CONFIG` to it before starting ComfyUI), keep
one entry and fill in your paths:

- `distro`: the name shown by `wsl -l -v` (kind `wsl` only);
- `python`, `upstream_dir`, `models_dir` (the folder that contains
  `wan_models/`): Linux paths;
- `checkpoints`: one path per model you downloaded; only those models can be
  selected (`cfpp_framewise_2step`, `cf_framewise_4step`,
  `cfpp_framewise_1step`, `cf_chunkwise_4step`);
- `env`: only `CC`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `XDG_CACHE_HOME`, `HF_HOME`, `CUDA_VISIBLE_DEVICES` are accepted;
- `offload_text_encoder`: `true` on GPUs with less than 40 GB (what upstream
  `inference.py` does automatically);
- `timeout_minutes`, optional `jobs_dir`, optional `wsl_mount_root`.

Restart ComfyUI, add **Causal Forcing Doctor**, pick the runtime and queue it.

## Memory

The runner builds the 11 GB UMT5-XXL encoder directly in bf16 from a
memory-mapped file instead of upstream's fp32 copy (bit-identical weights), so
a 31 GB WSL2 memory limit is enough. Observed peaks are in
`docs/BENCHMARKS.md`.
