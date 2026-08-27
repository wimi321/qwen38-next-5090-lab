# Qwen3.8 Next 5090 Lab

[简体中文](README.zh-CN.md) · [Reproducibility record](docs/qwen4-exp.md) · [Model status](docs/models.md) · [Security](SECURITY.md)

> [!IMPORTANT]
> This is an **unofficial, experimental downstream of
> [FreeToken](https://github.com/FlashML-org/FreeToken)**. It is not affiliated
> with or endorsed by Qwen, RadixArk, NVIDIA, FlashML, or their contributors.

Run the **text tower** from the complete 135 GB
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
checkpoint on one 32 GB RTX 5090. The runtime combines sparse PLE row streaming,
QSA Triton kernels, four-stream gated residuals, and heterogeneous MoE offload
behind an OpenAI-compatible API.

This repository is a source-only developer preview. It does not redistribute
model weights, publish the `freetoken` package name, or claim to be the first or
fastest implementation.

![Qwen3.8 Next 5090 Lab architecture](docs/assets/q38lab-architecture.svg)

## Verified alpha contract

| Area | `v0.1.0-alpha.1` contract |
|---|---|
| Input / output | Text input and text output only |
| Hardware | One RTX 5090 (32 GB), TP=1, WSL2/Ubuntu 24.04 |
| Scheduling | One running request, 8,192 total tokens |
| Cache / graphs | Naive cache, radix prefix reuse off, CUDA graph off |
| Expert execution | Packed NVFP4 routed weights with BF16 activations (**W4A16 compatibility**) |
| Checkpoint | Full 135,253,622,894-byte directory at revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Not included | Vision, video, audio, MTP, multi-request batching, 32K/262K claims, native W4A4 parity |

The checkpoint declares a routed-expert W4A4 recipe. This alpha preserves the
packed NVFP4 weights but does not consume the checkpoint's activation-scale
contract; measurements here therefore describe W4A16 compatibility, not W4A4
quality or numerical parity.

## Why it fits

- **PLE stays sparse.** The approximately 51 GB FP8 PLE table is read-only
  safetensors mmap. Only matched rows are copied and decoded to FP32 on the CPU,
  scaled, converted into pinned BF16 staging, and then transferred to the GPU.
- **QSA has its own cache path.** Twelve sparse-attention layers select from
  four-token blocks and keep main K/V plus index-key state; the other 36 layers
  use gated delta recurrence.
- **Experts span the memory hierarchy.** The 512 routed experts per layer use
  top-10 routing, pageable CPU layers, pinned host banks, PCIe transfers, and a
  GPU expert cache. The shared expert remains part of every layer.
- **Evidence is a release artifact.** Environment, resolved configuration,
  request timings, RSS/VRAM/page-fault/PCIe samples, tests, and checksums are
  stored together. README figures are generated from the recorded summary.

## Install from source

The verified path is Linux x86-64 inside WSL2. Keep both the repository and
checkpoint in the distribution's ext4 filesystem, not under `/mnt/c` or
`/mnt/d`. Install the CUDA 13 toolkit inside WSL, but do not install a Linux
display driver there; WSL uses the Windows NVIDIA driver.

```bash
git clone https://github.com/wimi321/qwen38-next-5090-lab.git
cd qwen38-next-5090-lab

uv sync --locked --extra accel
source .venv/bin/activate

q38lab doctor
```

The source distribution retains `import freetoken` and the legacy `ft` command
for compatibility. Install it in a dedicated virtual environment: it cannot be
co-installed with the upstream `freetoken` distribution.

## Download the pinned checkpoint

Read the [model-license guide](MODEL_LICENSES.md) before downloading. The command
requires explicit acknowledgement, pins the revision, and refuses to switch to
a different checkpoint if that revision disappears.

```bash
q38lab download --accept-qwen-license

# Optional: re-read and hash the complete local checkpoint.
q38lab download --accept-qwen-license --full-verify
```

The validated copy contains 419 non-cache files, 206 safetensors files, and
135,253,622,894 bytes. Model files are never added to this repository or its
release assets.

The pinned downloader writes an external verification receipt after Hugging
Face has resolved the exact commit. `serve` rechecks every control-file hash
and the complete file-stat fingerprint against that receipt. The optional
`--full-verify` mode additionally reads all 135 GB and must be used for release
evidence; a shape-only directory is never accepted by `serve`.

## Serve the model

```bash
q38lab serve --profile rtx5090-wsl2
```

The profile resolves to the audited launch envelope: `memory-ratio=0.89`,
`num-tokens=max-seq=max-prefill=8192`, `max-running-requests=1`, naive cache,
`qsa_triton`, graph disabled, MoE offload, and automatic expert-cache sizing.
CLI flags override `Q38LAB_*` environment variables, which override profile
defaults.

The unauthenticated server binds to `127.0.0.1` by default. `q38lab` rejects an
unauthenticated non-loopback bind unless the caller supplies
`--unsafe-allow-non-loopback`. That flag adds no authentication or TLS; do not
expose this development server directly to a network.

Check it with the bundled smoke suite:

```bash
q38lab smoke
```

Or call it with the OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1919/v1", api_key="local-only")
response = client.chat.completions.create(
    model="qwen3.8-flash-next-nvfp4",
    messages=[{"role": "user", "content": "Reply with exactly: q38lab-ok"}],
    temperature=0,
    max_tokens=32,
)
print(response.choices[0].message.content)
```

For a streaming request, add `stream=True` and iterate over the returned chunks.
The smoke suite also covers both thinking modes and one schema-constrained tool
call; model output is not a security boundary, so applications must still
validate tool names and arguments.

## Reproduce the evidence

```bash
q38lab bench --out results/rtx5090-YYYY-MM-DD
```

This is the authoritative release harness, not a quick microbenchmark. It
re-hashes the full checkpoint, runs the non-slow tests, measures the prompt/API
gates, executes 100 sequential requests, then performs a separate 30-minute
soak with periodic generation and continuous resource sampling. It consumes
the live launch attestation written by `q38lab serve` and refuses a dirty or
mismatched runtime checkout.

Do not compare numbers without the adjacent environment and resolved-config
records. TTFT is client-observed, and “effective prefill” includes fixed request
and CPU-staging overhead rather than measuring a kernel in isolation.

<!-- BEGIN GENERATED BENCHMARK SUMMARY -->
| Gate | Recorded alpha result |
|---|---:|
| Non-slow tests | 1,454 passed, 9 skipped, 11 deselected |
| 8,176-token prompt peak VRAM | 31,542 MiB (30.803 GiB) |
| Peak WSL RSS | 67.895 GiB |
| Sequential stability | 100/100 requests; p50 2.345 s; p95 2.363 s |
| 8,176-token client TTFT p50 | 7.225 s |
| 8,176-token effective prefill p50 | 1,130.83 tok/s |
<!-- END GENERATED BENCHMARK SUMMARY -->

The validated WSL instance had `swap=0`. The Windows host already had a 128 GiB
pagefile with aggregate system use, so this project does **not** claim that host
paging was absent. Short seven-token completions are not presented as a
steady-state decode benchmark; the release gate uses a separate 256–512-token
decode measurement.

See [the full procedure and caveats](docs/qwen4-exp.md) before quoting any
result.

## Project status

This alpha is intentionally narrow. The next milestones are:

1. End-to-end reference parity and a native SM120 W4A4 activation path.
2. PLE telemetry/hot-row caching and verified CUDA graph capture/replay.
3. Image support with a processor, media transport, vision prefill, mRoPE, and
   safe cache semantics.

Media is rejected today. A multimodal checkpoint does not make this server
multimodal: the online request layer, vision tower, placeholder expansion, and
media-aware cache behavior must all be implemented and tested first.

## Provenance, contributing, and citation

This downstream preserves the full FreeToken history and is based on audited
upstream commit `9ef3651309fe4058672f2cc92069238dea06be1b`. See
[MODIFICATIONS.md](MODIFICATIONS.md) for the downstream changes and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained attribution.

Contributions must be understood and tested by a human maintainer; model weights,
private logs, fabricated benchmarks, and unreviewed agent-only submissions are
not accepted. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or
pull request.

For academic use, cite both this software release via [CITATION.cff](CITATION.cff)
and the upstream FreeToken work that provides the serving-engine foundation.

## Licenses

- Repository code and documentation: [Apache License 2.0](LICENSE), including
  retained FreeToken and third-party notices.
- Qwen model artifacts: separate
  [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
- RadixArk checkpoint: its model card points users to the source-model terms and
  describes the checkpoint as a candidate release.

The repository license does not grant model rights or trademark rights. Nothing
here is legal advice; users are responsible for reviewing all terms that apply
to their use and deployment.
