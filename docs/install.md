# Install Qwen3.8 Next 5090 Lab

This page was modified by Qwen3.8 Next 5090 Lab contributors in 2026. The alpha
is not published to PyPI and never uses FreeToken's distribution name. A
CPython 3.12/Linux x86-64 wheel is attached to the companion GitHub prerelease.

## Requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- Python >= 3.10 for source installs; the binary wheel requires CPython 3.12
- CUDA toolkit 13.3 inside WSL2 for runtime JIT compilation

## Install the companion wheel

The binary is intentionally narrow: WSL2 Ubuntu 24.04, Linux x86-64, CPython
3.12, RTX 5090/SM120. It is not a portable manylinux wheel and does not bundle
the 135 GB checkpoint, CUDA, PyTorch, or the SGLang/FlashInfer acceleration
packages. Install into an isolated environment:

```bash
python3.12 -m venv "$HOME/q38lab-venv"
source "$HOME/q38lab-venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r \
  https://raw.githubusercontent.com/wimi321/qwen38-next-5090-lab/wheel-v0.2.0-alpha.1.post1/requirements-wheel-cu130.txt
python -m pip install \
  https://github.com/wimi321/qwen38-next-5090-lab/releases/download/wheel-v0.2.0-alpha.1.post1/qwen38_next_5090_lab-0.2.0a1.post1-cp312-cp312-linux_x86_64.whl
python -m pip check
q38lab doctor --profile rtx5090-wsl2-256k-image
```

The requirements file pins the exact CPython 3.12/CUDA 13 acceleration
artifacts by URL and SHA256. The installed wheel supports `doctor`, `download`,
`serve`, and `smoke`. `q38lab bench` intentionally stays in the source checkout
because it is the maintainer's evidence harness, not a normal serving command.

## Install from source

```bash
git clone https://github.com/wimi321/qwen38-next-5090-lab.git
cd qwen38-next-5090-lab
uv sync --locked --extra accel
source .venv/bin/activate
```

CUDA kernels are JIT-compiled on first use and need a CUDA 13 toolkit with
`nvcc` on `PATH`. Use a dedicated environment because this downstream retains
the `freetoken` import namespace and cannot be co-installed with upstream.
The tracked lock is the verified Linux x86_64/CUDA 13 dependency resolution;
contributors can add the test extra with `uv sync --locked --extra accel --extra dev`.

## Verify

```bash
source .venv/bin/activate
q38lab doctor
q38lab download --accept-qwen-license
q38lab serve --profile rtx5090-wsl2
```

Then run `q38lab smoke` and read the
[Qwen3.8 reproducibility record](qwen4-exp.md).
