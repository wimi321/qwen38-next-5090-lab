# Install the downstream source

This page was modified by Qwen3.8 Next 5090 Lab contributors in 2026. The alpha
is source-only; it is not published under FreeToken's PyPI name.

## Requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)

## Install from source

```bash
git clone https://github.com/wimi321/qwen38-next-5090-lab.git
cd qwen38-next-5090-lab
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[accel]"
```

CUDA kernels are JIT-compiled on first use and need a CUDA 13 toolkit with
`nvcc` on `PATH`. Use a dedicated environment because this downstream retains
the `freetoken` import namespace and cannot be co-installed with upstream.

## Verify

```bash
source .venv/bin/activate
q38lab doctor
q38lab download --accept-qwen-license
q38lab serve --profile rtx5090-wsl2
```

Then run `q38lab smoke` and read the
[Qwen3.8 reproducibility record](qwen4-exp.md).
