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

The diagram shows the unreleased v0.2 candidate. The verified v0.1 path omits
the vision branch and uses mmap/CPU PLE decoding as described below.

## Verified v0.1 contract

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

## v0.2 candidate: 256K plus images

The current development branch contains the `0.2.0a1` source candidate for
`v0.2.0-alpha.1`. It is **not a released or hardware-validated support claim**.
The verified support matrix remains the v0.1 table above until one evidence run
passes every v0.2 gate on the pinned full checkpoint.

| Candidate area | Unverified `v0.2.0-alpha.1` target (hardware harness pending) |
|---|---|
| Context accounting | 262,144 total tokens, including rendered text, expanded image tokens, and output |
| Boundary proof | Exactly 261,120 input tokens plus 1,024 generated tokens, once for text and once with a real image |
| Hardware / scheduling | One RTX 5090 (32 GB), WSL2, TP=1, one running request, 512-token prefill chunks |
| Cache / graphs | Naive cache, radix prefix reuse off, CUDA graph off |
| Images | Structured OpenAI `image_url`; HTTPS or base64 data URL; up to four images |
| Still excluded | Video, audio, MTP, radix cache, TP>1, multi-request scheduling, and context beyond 262,144 |

The candidate profile is:

```bash
q38lab doctor --profile rtx5090-wsl2-256k-image
q38lab serve --profile rtx5090-wsl2-256k-image
q38lab smoke --images
```

Do not use this section as evidence that the commands have passed the full
model gates. Release requires the exact text and image boundary requests,
Needle-in-a-Haystack and deterministic image cases, 100/100 mixed sequential
requests, a 30-minute soak, TTFT at or below 15 minutes, steady decode at or
above 5 tok/s, peak VRAM below 31 GiB, WSL RSS below 105 GiB, and WSL swap at
zero. No v0.2 performance number is published before those raw records exist.

## Execution paths

- **PLE stays sparse.** The verified v0.1 path keeps the approximately 51 GB
  FP8 table in a read-only safetensors mmap and decodes only matched rows on the
  CPU. The v0.2 candidate instead requires native Linux `io_uring` plus
  `O_DIRECT`, a globally bounded 4 GiB native LRU, 4 KiB-aligned reads, queue
  depth 512, and batches of at most 4,096 pages. Double-buffered pinned FP8 rows
  are transferred and decoded/scaled to BF16 on the GPU; the mmap path remains
  only a test/debug fallback for this profile.
- **QSA has its own cache path.** Twelve sparse-attention layers select from
  four-token blocks; the v0.2 candidate stores one persistent index key per
  four tokens plus a request-local tail ring and mRoPE coordinates. Its full
  256K main K/V and compressed-index budget is 6.1875 GiB. The SM120 selector
  uses a bounded 128 MiB FP32 workspace and top-512 kernel, with the Torch
  oracle and original Triton path retained as correctness fallbacks.
- **Experts span the memory hierarchy.** The 512 routed experts per layer use
  top-10 routing, pageable CPU layers, pinned host banks, PCIe transfers, and a
  GPU expert cache. The shared expert remains part of every layer. The v0.2
  candidate additionally enables route-aware native-NVFP4 prefill movement:
  after the current layer's router runs, a bounded 512-entry mask identifies
  the selected raw expert IDs. Coalesced selected rows are copied for banks
  whose per-expert row is at least 256 KiB; smaller banks are copied as one
  whole-layer entry so the batched CUDA copy remains asynchronous. Raw IDs and
  the full `[E]` double-buffer layout are preserved--there is no expert-ID
  compaction and no reduction in the reserved GPU buffer size. Registered
  banks use direct DMA when available, while LOCKED/PAGEABLE layers use the
  fixed pair of 32 MiB pinned bounce slabs.
- **Images enter before residual replication.** The candidate loads the 27-layer
  vision tower, expands image placeholders with the pinned Transformers
  processor, constructs three-axis interleaved mRoPE, and injects merged visual
  embeddings before copying the four gated-residual streams. Each image is
  encoded once; its CPU-resident BF16 embedding is sliced and transferred only
  where a 512-token prefill chunk intersects the image span.
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

The unreleased 256K/image candidate uses a separate profile and refuses to
start if native `io_uring`/`O_DIRECT` or the memory budget is unavailable:

```bash
q38lab serve --profile rtx5090-wsl2-256k-image
```

That profile resolves to `max-seq-len=num-tokens=262144`,
`max-prefill-length=512`, `max-running-requests=1`, `qsa_triton_sm120`, naive
cache, graph disabled, native PLE streaming, vision loading, and route-aware
native-NVFP4 MoE prefill. The sparse MoE path requires the existing
double-buffered prefill cache and refuses any non-native NVFP4 bank layout. It
deliberately waits for the current layer's routing decision instead of eagerly
prefetching the next full expert layer, and it disables the separate prefill
hit-D2D path. Those tradeoffs are why its net performance remains an evidence
question, not a source-level claim. The profile reserves QSA cache, selector
workspace, vision weights, and runtime headroom before automatically sizing the
GPU expert cache; it fails closed if the fixed geometry cannot fit the resolved
0.89 planning budget. The separate evidence gate, not this arithmetic alone,
enforces the strict peak-VRAM result below 31 GiB.

Exercise the unverified candidate test surface with a maintainer-selected
public HTTPS fixture; this smoke command is not release evidence:

```bash
HTTPS_IMAGE_URL='https://replace-with-your-public-fixture.example/chart.png'
q38lab smoke --images --https-image-url "$HTTPS_IMAGE_URL"
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

The v0.2 source candidate contains an unverified image-input path through the
same API. This example shows only the request shape, not a published validation
result or a supported input contract:

```python
response = client.chat.completions.create(
    model="qwen3.8-flash-next-nvfp4",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.org/chart.png"}},
            {"type": "text", "text": "What is the chart's highest value?"},
        ],
    }],
    temperature=0,
    max_tokens=128,
)
```

The candidate implementation is limited to HTTPS and base64 image data URLs;
these limits are not hardware-validation results. It caps a
request at four images, 20 MiB per image and 40 MiB total, with a ten-second
fetch deadline. It rejects HTTP/local URLs, loopback, private, link-local or
reserved addresses, unsafe redirects, DNS rebinding, unsupported MIME types,
audio, and video. Requests containing images cannot enter the shared text
prefix cache, and media bytes, visual embeddings, and mRoPE state are released
with the request.

`Q38LAB_DOH_FALLBACK=1` is an explicit compatibility opt-in only for WSL/TUN
setups whose transparent fake-IP DNS returns non-global addresses for every
public host; it is disabled by default. The fallback connects to fixed
Cloudflare DoH public IPs `1.1.1.1` and `1.0.0.1` while authenticating
`cloudflare-dns.com` with TLS SNI, and it still rejects every non-global target
answer. Mixed public/non-public system answers and numeric IP literals never
use the fallback. `doctor --json` and release evidence record whether the
opt-in was enabled and that libc `getaddrinfo` has only deadline-bounded,
four-slot soft cancellation—there is no portable hard cancel for a lookup
already running in libc. This compatibility path does not make the v0.2
candidate verified or relax any evidence gate.

For a streaming request, add `stream=True` and iterate over the returned chunks.
The smoke suite also covers both thinking modes and one schema-constrained tool
call; model output is not a security boundary, so applications must still
validate tool names and arguments.

## Reproduce the evidence

```bash
q38lab bench --profile rtx5090-wsl2 \
  --out results/rtx5090-YYYY-MM-DD
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
| Verified field | Result |
|---|---:|
| Runtime commit | `643bcd6a82eb` |
| Checkpoint revision | `7b719225242a` |
| GPU | NVIDIA GeForce RTX 5090 |
| Executed expert path | W4A16 compatibility |
| Peak VRAM | 30.926 GiB |
| Peak WSL RSS | 72.234 GiB |
| WSL swap | 0 MiB |
| 8176 rendered-token TTFT (p50 / p95) | 7283.0 / 7355.5 ms |
| 8176 effective prefill (p50) | 1122.6 tok/s |
| 256-token steady decode | 14.0 tok/s |
| Sequential requests | 100/100 |
| Continuous run | 30.0 min |
| Tests | 1530 passed, 0 failed |
<!-- END GENERATED BENCHMARK SUMMARY -->

The validated WSL instance had `swap=0`. The Windows host already had a
pagefile, so this project does **not** claim that host paging was absent. Short
seven-token completions are not presented as a
steady-state decode benchmark; the release gate uses a separate 256–512-token
decode measurement.

See [the full procedure and caveats](docs/qwen4-exp.md) before quoting any
result.

The v0.2 harness is profile-aware and additionally records bounded selector
workspace use, PLE bytes/cache/wait/page-fault telemetry, image-token and vision
latency, per-chunk prefill timing, and route-aware MoE prefill counters. The
`q38lab.moe_prefill` snapshot reports unique active rows summed over layer
calls, the corresponding 512-row opportunities, bytes actually scheduled for
copy, and the hypothetical full-bank bytes. Because sub-256-KiB banks move as
whole-layer entries, its byte fraction is intentionally distinct from its row
fraction; these are movement-accounting counters, not invented hardware PCIe
measurements. The launch attestation also carries a native PLE checkpoint
probe: every one of the 128 shard boundaries plus eight deterministic
bigram/trigram hash rows must match independent safetensors slices after GPU
FP8 decode. A v0.2 evidence directory is not
release-compatible unless it contains both 261,120 + 1,024 boundary proofs and
passes every resource, throughput, mixed-request, and soak gate. Until such a
reviewed directory exists, the generated benchmark block above intentionally
remains the v0.1 result.

The candidate evidence entry point requires both a deterministic local image
and a stable public HTTPS fixture:

```bash
q38lab bench --profile rtx5090-wsl2-256k-image \
  --image-file "$HOME/q38lab-fixtures/chart.png" \
  --https-image-url "https://example.org/q38lab-chart.png" \
  --decode-tokens 1024 \
  --out results/rtx5090-256k-image-YYYY-MM-DD
```

Both image arguments are mandatory for this profile. The 30-minute soak keeps
alternating text and image requests so image-only leaks cannot hide behind a
later text-only plateau. The decode budget may be
256–1,024 tokens; the harness rejects a missing gate, dirty launch attestation,
or telemetry mismatch instead of emitting release-compatible evidence. When a
transparent fake-IP environment requires the DoH opt-in, use the same
`Q38LAB_DOH_FALLBACK=1` setting for `doctor`, `serve`, and `bench`; its value and
the system-resolver soft-cancellation limitation are preserved in the evidence.

## Project status

The verified v0.1 alpha is intentionally narrow. The unreleased v0.2 source
contains the 256K/image candidate paths described above, but they remain
unverified and outside the support matrix until the full hardware gate passes.
Later milestones are:

1. End-to-end reference parity and a native SM120 W4A4 activation path.
2. PLE hot-row optimization and verified QSA/PLE CUDA graph capture/replay.
3. Video/audio evaluation, MTP, radix-cache semantics, and contexts beyond the
   single-request 262,144-token candidate.

The v0.1 profile continues to reject media. Source code alone also does not make
the v0.2 profile supported: the online image path, vision tower, placeholder
expansion, mRoPE, chunk scatter, cleanup, and security controls must pass the
recorded full-checkpoint gates before the support matrix changes.

## Provenance, contributing, and citation

This downstream preserves the full FreeToken history and is based on audited
upstream commit `9ef3651309fe4058672f2cc92069238dea06be1b`. See
[MODIFICATIONS.md](MODIFICATIONS.md) for the downstream changes and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained attribution.
The v0.2 SM120 top-512 specialization is an Apache-2.0 adaptation from
[`yhfgyyf/sglang-qwen38-flash-next-sm120`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120)
at exact source commit `30edf3503961a471b25150aa890f8166031b5738`.
The design review also references SGLang's
[Qwen3.8 integration PR #36497](https://github.com/sgl-project/sglang/pull/36497),
[PLE NVMe PR #36567](https://github.com/sgl-project/sglang/pull/36567), and
[SM120 QSA PR #36556](https://github.com/sgl-project/sglang/pull/36556).
Those projects and results do not endorse this downstream, and their 96 GB,
MTP, or CUDA Graph settings and measurements are not results of this project.

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
