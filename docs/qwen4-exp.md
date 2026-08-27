# Qwen3.8-Flash-Next (Qwen4-Exp) reproducibility record

This document was substantially modified by Qwen3.8 Next 5090 Lab contributors
in 2026; see [`../MODIFICATIONS.md`](../MODIFICATIONS.md). It describes an
unofficial FreeToken downstream, not an upstream FreeToken support claim.

This page records two deliberately separate milestones for exactly
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
at revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

The released `v0.1.0-alpha.1` milestone has recorded hardware evidence for a
deliberately narrow contract:

- text only, one RTX 5090, tensor parallel size 1;
- the complete HF checkpoint, read directly without FTW conversion;
- an 8,192-token total context budget and the naive cache;
- MTP, vision, video, audio and radix prefix reuse disabled.

The `0.2.0a1` / `v0.2.0-alpha.1` milestone now has reviewed RTX 5090/WSL2
hardware evidence for a deliberately narrow image-capable alpha. Its profile
combines a 262,144-token total budget with image input. Total tokens include
rendered text, expanded image tokens, and output; the validated boundary is
261,120 input plus 1,024 output, once for text and once with a real image. The
formal record is
[`results/rtx5090-2026-08-28-v02-alpha1-757872a-run7/`](../results/rtx5090-2026-08-28-v02-alpha1-757872a-run7/).

Both profiles disable CUDA graphs and radix prefix reuse. The complete PLE
auxiliary table must never be materialized on the GPU. The v0.2 alpha adds
native direct-I/O streaming and compressed QSA state; its support statement is
bounded by the reviewed target-GPU full-checkpoint evidence rather than source
code or unit tests alone.

The pinned checkpoint declares a ModelOpt **W4A4** recipe for routed experts.
FreeToken's existing NVFP4 expert backends preserve its packed E2M1 weights,
block scales and global scales, but currently execute them with BF16
activations (**W4A16 compatibility**); the per-projection activation scales are
not consumed. The loader emits a prominent warning when it detects this
metadata. This is sufficient only for the experimental hardware bring-up: it
is not checkpoint-numerically equivalent and does not inherit the checkpoint's
published W4A4 quality claims. Formal support requires a W4A4 execution path
and reference/GPU parity.

## v0.2 validated alpha architecture

The hardware-validated `rtx5090-wsl2-256k-image` alpha profile configures TP=1,
one running request, naive cache, graph disabled, a 262,144-token total limit,
and 512-token prefill chunks. It is specific to one 32 GB RTX 5090 under WSL2
and fails closed if the fixed cache geometry plus minimum expert residency
cannot fit the resolved 0.89 planning budget. The formal evidence independently
enforces the strict measured peak below 31 GiB.

### Compressed QSA and SM120 selection

Only the 12 QSA layers allocate sparse-attention cache. Main BF16 K/V remains
available for selected attention, while index keys are persisted once per
four-token block rather than once per token. Each request also keeps a
four-token pending tail and the original three-axis positions required for
mRoPE. At 262,144 tokens the combined main K/V and compressed index-key budget
is 6.1875 GiB.

`qsa_triton_sm120` retains the Triton scorer but bounds its FP32 selector
workspace at 128 MiB. A CUDA top-512 specialization is adapted under
Apache-2.0 from
[`yhfgyyf/sglang-qwen38-flash-next-sm120`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120),
exact commit `30edf3503961a471b25150aa890f8166031b5738`, file
`python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh`. The wrapper and
specialization were changed for FreeToken and a wide threshold-bin rescan was
added; the source header and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) retain the attribution.
The PyTorch oracle and original `qsa_triton` path remain correctness fallbacks,
not alternate release configurations.

### Native PLE streaming

The validated alpha rejects startup unless the checkpoint resides on Linux/WSL
ext4 and a read-only probe proves native `io_uring` plus `O_DIRECT`. The backend
uses 4 KiB-aligned reads, queue depth 512, at most 4,096 pages per batch, and
one globally bounded 4 GiB C++ LRU shared across the physical shards. The next
prefill chunk is prefetched while later GPU layers process the current chunk.

Only hashed rows are read. Double-buffered pinned FP8 staging is copied to the
GPU, where a bit-exact lookup and checkpoint scale produce BF16 rows for the
PLE projection/convolution. The existing mmap/CPU decode path remains useful
for unit tests and debugging, but it is not accepted as a production fallback
for this profile. Runtime evidence must include bytes read, LRU hits/misses,
wait time, page faults, staged bytes, and GPU-decoded rows.

### Route-aware native-NVFP4 MoE prefill

The alpha profile sets `FREETOKEN_MOE_PREFILL_SPARSE=1`. Startup accepts
that setting only with the double-buffered MoE prefill cache and native
`nvfp4` banks; a missing overlap buffer or a Marlin/b12x/other bank layout is a
hard error. The verified v0.1 profile does not enable this path.

This implementation is deliberately **not** compact-expert routing. After the
current layer's top-10 router has produced its IDs, a fixed 512-entry device
mask is copied to bounded pinned host storage and synchronized once. The host
derives sorted, coalesced runs of active raw expert IDs. Those raw IDs remain
the row positions in the existing `[E]` GPU double buffer, and the grouped
NVFP4 kernel still receives the original IDs and `num_experts=512`. Inactive
rows may retain stale bytes, but the current routing cannot address them. The
full double-buffer reservation is unchanged, so this path reduces expert-bank
movement only; it does not reduce the MoE cache's GPU-memory budget.

For a bank whose per-expert row is at least 256 KiB, only the coalesced active
runs are copied. For this checkpoint that covers the large packed gate/up and
down weight rows. Banks below 256 KiB per expert--including the smaller scale
and global-scale banks--still move as one whole-layer entry. This avoids the
CUDA runtime behavior in which mixing small entries into a batched registered-
memory copy can make the call host-synchronous. CUDA-registered sources use
direct asynchronous copies when the batched runtime entry point is available;
LOCKED/PAGEABLE layers copy the same selected raw-ID runs through two 32 MiB
pinned bounce slabs.

The active set is unavailable until the layer's router completes. Sparse mode
therefore does not eagerly prefetch the next full expert layer and does not use
the separate prefill hit-D2D optimization. It also introduces one bounded
device-to-host synchronization per MoE layer. These are explicit scheduling
tradeoffs: unit parity and reduced planned bytes do not establish a speedup.
The reviewed full-checkpoint evidence, rather than those implementation facts,
is the authority for the v0.2 latency gate.

The terminal runtime snapshot exports `q38lab.moe_prefill` with:

- `active_rows`: unique raw expert rows summed over MoE prefill layer calls;
- `possible_rows`: 512 rows for each corresponding layer call;
- `bytes_copied`: bytes scheduled by the sparse movement plan, including full
  copies of every small bank;
- `full_bytes`: the bytes that full-layer movement would have scheduled; and
- `row_fraction` / `byte_fraction`: ratios derived independently from the two
  pairs above.

Because small banks remain whole-layer copies, `byte_fraction` is not expected
to equal `row_fraction`. These are internal movement-accounting counters, not
substitutes for profiler or driver PCIe measurements.

### Image-only multimodal path

Transformers `5.16.1` is pinned for `AutoProcessor`/`Qwen3VLProcessor`. The
validated alpha implementation maps the Qwen4-Exp 27-layer vision tower,
patch projection, position interpolation, merger, and its BF16 weights. It
expands image placeholders,
validates `image_grid_thw`, and builds three-axis interleaved mRoPE plus rope
delta. Merged visual embeddings replace placeholder embeddings before the
hidden state is copied into four gated-residual streams.

Each image is encoded once. The complete merged BF16 embedding is then held in
pageable CPU memory; only the rows intersecting the active 512-token prefill
chunk are transferred and scattered. Image requests never enter the shared
text prefix cache. Request cleanup drops source bytes, processor tensors,
visual embeddings, image spans, and mRoPE state. Video, audio and MTP remain
explicitly unsupported.

The architecture review was informed by SGLang's
[Qwen3.8 integration PR #36497](https://github.com/sgl-project/sglang/pull/36497),
[PLE NVMe PR #36567](https://github.com/sgl-project/sglang/pull/36567), and
[SM120 QSA PR #36556](https://github.com/sgl-project/sglang/pull/36556). This
project does not copy their 96 GB launch envelope, MTP/CUDA Graph configuration,
or benchmark values, and it makes no performance comparison with them.

## Reproducible WSL environment

FreeToken is a Linux runtime. On Windows, use an Ubuntu 24.04 WSL2 distribution
whose VHD and Linux files live on an NTFS drive, but whose repository and model
are stored *inside* the distribution's ext4 filesystem. Do not serve the model
from `/mnt/d`.

The WSL install is a manual host-administration step. In an **Administrator
PowerShell**, first ensure the current Store version of WSL is installed and
that `wsl --help` lists `--location`:

```powershell
wsl --update
```

Before creating the distribution, put this in `%UserProfile%\.wslconfig` so a
new VHD has a 750GB logical limit and Windows retains about 16GB of RAM:

```ini
[wsl2]
memory=112GB
processors=32
swap=0
defaultVhdSize=750GB
```

Run `wsl --shutdown` after saving the file, then install the distribution from
**Administrator PowerShell**:

```powershell
wsl --shutdown
wsl --install --distribution Ubuntu-24.04 --location D:\WSL\Ubuntu-24.04
```

Follow the reboot prompt and complete the Linux username/password setup on the
first launch. See Microsoft's [WSL install
guide](https://learn.microsoft.com/windows/wsl/install) and [command
reference](https://learn.microsoft.com/windows/wsl/basic-commands).

`defaultVhdSize` affects newly created VHDs; do not attempt to shrink an
existing larger VHD. The setting is documented in Microsoft's
[advanced WSL configuration](https://learn.microsoft.com/windows/wsl/wsl-config).

Inside Ubuntu, install build tools, `uv`, and the CUDA **13 toolkit only**. Do
not install an NVIDIA Linux display driver in WSL: GPU access comes from the
Windows host driver. Follow NVIDIA's [CUDA on WSL
guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), then verify the
host/device and compiler before building FreeToken:

```bash
nvidia-smi
nvcc --version
test "$(swapon --show --noheadings | wc -l)" -eq 0
```

The expected machine for this record is an RTX 5090 with Windows driver
591.86, `nvcc` 13.x, 32 logical processors and no WSL swap.

Install the ordinary build prerequisites and `uv` before creating the source
environment:

```bash
sudo apt update
sudo apt install -y build-essential curl git pkg-config
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

## Source and checkpoint

Keep both paths below in the WSL ext4 filesystem:

```bash
mkdir -p "$HOME/src" "$HOME/models"
cd "$HOME/src"
git clone https://github.com/wimi321/qwen38-next-5090-lab.git
cd qwen38-next-5090-lab

uv sync --locked --extra accel --extra dev
source .venv/bin/activate
```

Every evidence directory records the checked-out downstream commit. The audited
FreeToken merge base is
`9ef3651309fe4058672f2cc92069238dea06be1b`; it establishes provenance but is
not a substitute for the downstream release commit.

Download only the pinned revision. If that commit is unavailable, stop and
audit the replacement rather than silently selecting another revision:

```bash
q38lab download --accept-qwen-license --full-verify
```

The downloader writes its verification receipt next to, not inside, the model
directory. It refuses a destination inside any Git worktree, checks remaining
disk capacity before network I/O, and never falls back to another revision.
The full mode reads every checkpoint byte; the release harness repeats that
same shared verifier and does not depend on an undocumented external manifest.

The 2026-08-27 validation copy contained 419 non-cache files
(135,253,622,894 bytes), including 206 safetensors files.  Every Hugging Face
entry verified against the pinned revision. The canonical GNU-format manifest
digest is `6cc22b628ca575785e5dfdcab3c7056e79a7eac798969a145341ed1530c2a3a8`.

## RTX 5090 launch

The release profile is the preferred entry point:

```bash
q38lab serve --profile rtx5090-wsl2
```

The equivalent low-level command below is retained so the resolved configuration
can be audited. These are parser-backed FreeToken flags; the
explicit sequence limit keeps the first milestone honest, while `--num-tokens`
sets the total KV capacity rather than the checkpoint's advertised context:

```bash
source "$HOME/src/qwen38-next-5090-lab/.venv/bin/activate"
MODEL_DIR="$HOME/models/qwen38-flash-next-nvfp4-7b71922"

ft serve \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.8-flash-next-nvfp4 \
  --gpu 0 \
  --tp-size 1 \
  --max-running-requests 1 \
  --max-seq-len-override 8192 \
  --max-prefill-length 8192 \
  --num-tokens 8192 \
  --memory-ratio 0.89 \
  --cache-type naive \
  --attention-backend qsa_triton \
  --graph 0 \
  --moe-backend offload \
  --moe-cache-auto \
  --nvfp4-backend auto
```

The `0.89` ratio is the validated RTX 5090 setting for this checkpoint. The
Qwen4-Exp auto planner also deducts a model-scoped 512 MiB runtime guard before
choosing expert-cache geometry. Do not round the ratio back to the CLI default
or remove the guard; the authoritative peak is read from the release evidence,
and must remain below the strict `<31 GiB` threshold.

Do not force the original four-layer estimate. This checkpoint's expert banks
occupy about 63.46 GiB, while a 112 GiB WSL instance receives a FreeToken pin
budget of about 44.02 GiB. Four CPU layers would still leave about 58.17 GiB to
pin and can fail in `cudaHostRegister`. With `--moe-cpu-layers` omitted, the WSL
budget resolver selects the minimum safe head-and-tail set (15 layers on this
configuration: 0-7 and 41-47), leaving about 43.63 GiB pinned. Keep the first
startup on auto and record the resolved layer list from the log. Passing the
count `--moe-cpu-layers 15` is not equivalent because count syntax distributes
layers evenly through the model.

The checkpoint's PLE bank is separate from the MoE expert cache. The validated
host already had a Windows pagefile, which was not created or resized for this
run. Do not add WSL swap, alter the host pagefile, or rely on paging to make a
failing configuration appear to pass.
`--graph 0` is intentional for this bring-up; do not change it until the PLE
staging test and the QSA kernels both pass real capture/replay parity on the
target CUDA/Triton stack.

For this model `moe_intermediate_size=640`; on an SM120 GPU the current
`--nvfp4-backend auto` policy therefore selects FreeToken's Triton W4A16
inline-dequant backend (the FlashInfer b12x crossover defaults to 1024).
`--nvfp4-backend` selects among existing W4A16 kernels; forcing FlashInfer does
not turn the run into W4A4. Preserve the startup compatibility warning in the
test record.

### Validated 256K/image alpha launch

Run the extended doctor before starting the alpha profile:

```bash
q38lab doctor --profile rtx5090-wsl2-256k-image --json
q38lab serve --profile rtx5090-wsl2-256k-image
```

The resolved profile must show all of these values without local overrides:

| Field | Required alpha value |
|---|---:|
| `max-seq-len` / `num-tokens` | 262,144 |
| `max-prefill-length` | 512 |
| `max-running-requests` / TP | 1 / 1 |
| `memory-ratio` | 0.89 |
| cache / graph | naive / 0 |
| attention backend | `qsa_triton_sm120` |
| PLE backend | native `io_uring` + `O_DIRECT` |
| PLE LRU / queue / batch | 4 GiB / 512 / 4,096 pages |
| selector workspace | at most 128 MiB |
| QSA full-context cache budget | 6.1875 GiB |
| MoE prefill movement | route-aware native-NVFP4 raw-ID rows; full `[E]` double buffers |
| vision | enabled |

`doctor` must prove that the source and checkpoint are on the WSL ext4
filesystem, aligned direct reads work against a real checkpoint shard,
`io_uring` is enabled for the current process, the native extension imports,
and the locked-memory/host/GPU budgets are viable. The startup planner reserves
QSA cache, selector workspace, vision weights, and runtime headroom before
sizing the MoE expert cache. Do not relax the budget, enable a pagefile inside
WSL, or switch to mmap to turn a refusal into an apparent pass.

For a transparent fake-IP DNS environment only, rerun `doctor`, `serve`, and
the evidence harness with the explicit `Q38LAB_DOH_FALLBACK=1` opt-in; it is
disabled by default. Doctor reports both that setting and the system resolver's
soft-cancellation limitation: CPython/libc provides no portable hard cancel for
an in-progress `getaddrinfo`, so the request deadline-bounds a daemon helper and
caps potentially stuck lookups at four slots instead.

## API smoke checks

Wait for the ready log, then check lifecycle and resolved runtime metrics:

```bash
curl -fsS http://127.0.0.1:1919/health
curl -fsS http://127.0.0.1:1919/v1/models
ft ctl --json stats
```

Non-streaming greedy request:

```bash
curl -fsS http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-flash-next-nvfp4",
    "messages":[{"role":"user","content":"Reply with exactly: freetoken-ok"}],
    "temperature":0,
    "max_tokens":32,
    "stream":false
  }'
```

Streaming form of the same request (it must finish with `[DONE]` and assemble
to the same greedy text):

```bash
curl -fsSN http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-flash-next-nvfp4",
    "messages":[{"role":"user","content":"Reply with exactly: freetoken-ok"}],
    "temperature":0,
    "max_tokens":32,
    "stream":true,
    "stream_options":{"include_usage":true}
  }'
```

Exercise both chat-template thinking modes with `reasoning_effort` set to
`none` and `high`. Exercise tool calling with a real function schema and verify
that both streaming and non-streaming responses return the same function name
and JSON arguments:

```bash
curl -fsS http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-flash-next-nvfp4",
    "messages":[{"role":"user","content":"Explain why the sky is blue in one sentence."}],
    "reasoning_effort":"high",
    "temperature":0,
    "max_tokens":128
  }'
```

```bash
curl -fsS http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-flash-next-nvfp4",
    "messages":[{"role":"user","content":"What is the weather in Shanghai?"}],
    "reasoning_effort":"none",
    "tools":[{"type":"function","function":{"name":"get_weather","description":"Read weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice":"auto",
    "temperature":0,
    "max_tokens":128
  }'
```

For the v0.2 image alpha, the OpenAI request keeps image and text
parts structured:

```bash
HTTPS_IMAGE_URL=https://example.org/chart.png
curl -fsS http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\":\"qwen3.8-flash-next-nvfp4\",
    \"messages\":[{
      \"role\":\"user\",
      \"content\":[
        {\"type\":\"image_url\",\"image_url\":{\"url\":\"$HTTPS_IMAGE_URL\"}},
        {\"type\":\"text\",\"text\":\"What is the highest value in this chart?\"}
      ]
    }],
    \"temperature\":0,
    \"max_tokens\":128
  }"
```

The validated alpha API accepts HTTPS and `data:image/...;base64,...` image
inputs. It permits no more than four images, 20 MiB per image and 40 MiB total,
and applies one ten-second request deadline. It rejects local files, HTTP,
credentials in URLs, loopback, private/link-local/reserved peers, any hostname
with a non-public DNS answer, rebinding between validation and connect, unsafe
redirect hops, invalid MIME types, audio, and video. TLS certificate
verification and SNI use the original hostname while the TCP connection is
pinned to an already audited public IP.

The DoH fallback does not weaken those rules. It is considered only when all
system DNS answers are non-global (the transparent fake-IP case), never for a
mixed public/non-public answer set or a numeric literal. It connects directly
to fixed Cloudflare DoH public IPs `1.1.1.1` and `1.0.0.1` while TLS
authenticates `cloudflare-dns.com`, follows a bounded CNAME chain, and accepts
only global target addresses. Every redirect is resolved and audited again.

Exercise data URL, caller-selected public HTTPS, four-image, streaming-image,
and rejection cases with:

```bash
q38lab smoke --images --https-image-url "$HTTPS_IMAGE_URL"
```

This command is a convenience smoke surface, not a substitute for the formal
v0.2 evidence. The authoritative harness owns its image fixtures and combines
a real image with the 261,120-input-token boundary. An over-limit request
returns OpenAI-style `context_length_exceeded` detail with separate
rendered-text, expanded-image, requested-output, and 262,144-total counts.

For the 1, 128, 2K and 8K boundary cases, count the fully rendered chat prompt
with the checkpoint tokenizer. Prompt tokens plus requested output tokens must
not exceed 8,192; an "8K prompt" therefore leaves explicit headroom for chat
template tokens and decoding.

For the sequential stability gate, reuse a short deterministic payload and
fail at the first HTTP error:

```bash
SMOKE_PAYLOAD='{"model":"qwen3.8-flash-next-nvfp4","messages":[{"role":"user","content":"Reply with OK"}],"temperature":0,"max_tokens":8}'
for _ in $(seq 1 100); do
  curl -fsS http://127.0.0.1:1919/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data "$SMOKE_PAYLOAD" > /dev/null || exit 1
done
ft ctl --json stats
```

## Measurement record

The released v0.1 profile remains reproducible with:

```bash
q38lab bench --profile rtx5090-wsl2 \
  --out results/rtx5090-YYYY-MM-DD
```

After starting the alpha profile, its fail-closed harness is invoked with
maintainer-owned deterministic image fixtures:

```bash
q38lab bench --profile rtx5090-wsl2-256k-image \
  --image-file "$HOME/q38lab-fixtures/chart.png" \
  --https-image-url "https://example.org/q38lab-chart.png" \
  --decode-tokens 256 \
  --out results/rtx5090-256k-image-YYYY-MM-DD
```

For the v0.2 profile, both image arguments are mandatory, the HTTPS URL must be
public and stable for the run, and `--decode-tokens` may be 256 through 1,024.
The reviewed run used 256 tokens for this steady-decode contract.
The harness reads the exact clean-checkout launch attestation, independently
tokenizes all boundaries, and fails rather than creating release-compatible
evidence when a gate or telemetry cross-check is missing. Its
`environment.json` records whether `Q38LAB_DOH_FALLBACK` was enabled and that
system `getaddrinfo` hard cancellation is unavailable; these audit fields do
not count as a passed image or release gate.

Do not hand-maintain benchmark numbers in this document. The authoritative v0.2
record is
[`results/rtx5090-2026-08-28-v02-alpha1-757872a-run7/`](../results/rtx5090-2026-08-28-v02-alpha1-757872a-run7/);
the English README table is generated from its `summary.json`, and CI rejects a
stale table. That evidence directory contains exactly 10 tracked UTF-8 files:
`environment.json`, `resolved-config.json`, `summary.json`, `requests.jsonl`,
`latency.csv`, `resource-samples.csv`, `pytest.txt`,
`runtime-telemetry.json`, `ple-checkpoint-probe.json`, and exhaustive
`SHA256SUMS`. The v0.1 record retains its original evidence format.

The release harness re-hashes the complete checkpoint, runs the non-slow test
suite, verifies the four rendered prompt boundaries and API semantics, completes
100 sequential requests, and then runs a separate 30-minute soak. The v0.1
steady-decode contract is 256--512 tokens; the v0.2 contract is 256--1,024
tokens, and the reviewed v0.2 run used 256. Short seven-token prompt
completions are never used to infer sustained decode throughput. TTFT is
client-observed, and "effective prefill" includes request and CPU-staging
overhead rather than being a pure kernel measurement.

README resource values are the formal API acceptance-window maxima, covering
the first recorded API gate through the final soak request. The retained raw
`resource-samples.csv` begins earlier and therefore also contains
pre-acceptance preflight and pytest samples; those rows remain auditable but do
not contribute to the README maxima.

WSL must have no configured swap. A pre-existing Windows pagefile is recorded
but never modified, and zero host paging is not claimed. Portable
`nvidia-smi` sampling does not expose reliable PCIe RX/TX counters, so the
evidence marks them unavailable rather than inventing a value. The reviewed
v0.2 record has `source.release_compatible=true` and passes the strict raw-data,
checksum, runtime-commit, and release-input validation.

The v0.2 record additionally requires 8K regression, 32K, 128K, and 261,120
rendered-input cases; exact 1,024-token completion at the text and real-image
boundary; Needle-in-a-Haystack at multiple depths; deterministic OCR, object,
and chart answers; one cold plus three warm PLE measurements; and selector,
PLE, vision, prefill-chunk, and MoE-prefill telemetry. The server snapshot must
prove that the selector workspace never exceeded 128 MiB, warm PLE lookups
produce real cache hits rather than a reused client-side result, and the MoE
movement counters agree with the raw evidence window.

Use `ft ctl --json stats` for FreeToken's TTFT, throughput, request count and
VRAM fields. Cross-check process RSS/page faults through `/proc/<pid>/status`
or `pidstat -r`, GPU memory through `nvidia-smi`, and PCIe traffic through an
NVIDIA profiler or `nvidia-smi dmon`. Sample throughout the run rather than
only before and after it. For example, this records framebuffer memory and
PCIe receive/transmit throughput once per second:

For the alpha profile, `ft ctl --json stats` exposes the last terminal
`q38lab.moe_prefill` snapshot. Check `0 <= active_rows <= possible_rows`,
`0 <= bytes_copied <= full_bytes`, and independently recompute both fractions.
The snapshot describes route-aware prefill movement, not decode-cache hit/miss
rate; continue to mark an unexported decode hit rate as unavailable rather than
inferring it from PCIe traffic.

```bash
mkdir -p "$HOME/qwen4-exp-metrics"
nvidia-smi dmon -i 0 -s mt -d 1 -o DT \
  -f "$HOME/qwen4-exp-metrics/nvidia-dmon.log"
```

## Acceptance

### Recorded v0.1 boundary

The reviewed run satisfies the first text-only hardware milestone: the pinned
full checkpoint, 8K, one request, TP=1, naive cache, graph disabled, W4A16
compatibility, 100/100 sequential requests, and a 30-minute run within the
recorded resource gates. It does not validate image input or context beyond
8K. The WSL `swap=0` condition is proven, but a stronger claim that no host
pagefile existed or was touched is not: the pre-existing Windows pagefile had
non-zero aggregate system usage. Keep that caveat with every quoted v0.1
result.

### Recorded v0.2 alpha boundary

The reviewed run in
[`results/rtx5090-2026-08-28-v02-alpha1-757872a-run7/`](../results/rtx5090-2026-08-28-v02-alpha1-757872a-run7/)
passes the formal acceptance gates for the pinned full checkpoint and this
narrow profile:

- the non-slow suite, CPU oracles, native direct-I/O checks, target-GPU QSA/cache
  checks, visual loader mapping, and independent PLE shard/hash-row probes pass;
- 8K regression, 32K, 128K, and 261,120 rendered-input cases pass, including
  Needle-in-a-Haystack at every required depth;
- both text and real-image requests complete the exact
  `261,120 + 1,024 = 262,144` boundary, while a one-token excess is rejected as
  `context_length_exceeded` rather than silently truncated;
- deterministic OCR, object, and chart fixtures pass, as do streaming parity,
  thinking, tool calling, data URL, public HTTPS, four-image, and streaming-image
  cases;
- the formal hardware rejection cases for HTTP image URLs, loopback, local
  files, and audio pass, while the broader media-security behavior remains
  covered by the zero-failure test gate;
- client-observed 261K TTFT satisfies the 15-minute ceiling, and the exact
  256-token steady decode used by this run satisfies the 256--1,024-token
  contract, the 5 tok/s floor, and `finish_reason=length`;
- all 100 mixed short text/image sequential requests pass, followed by the
  required alternating 30-minute soak without OOM, crash, stale media state, or
  detected monotonic WSL RSS growth;
- peak VRAM, WSL RSS, and zero-WSL-swap gates pass, with the pre-existing Windows
  pagefile caveat retained; and
- cold/warm PLE, selector, vision, chunk, route-aware MoE movement, page-fault,
  and resource telemetry cross-check against the raw files, whose exhaustive
  checksums also validate.

This alpha supports only one RTX 5090 under Ubuntu 24.04 WSL2, TP=1, one running
request, naive cache, CUDA graphs disabled, W4A16 compatibility, and at most
262,144 total tokens. Audio, video, MTP, radix prefix cache, TP>1, concurrent
requests, contexts beyond 262,144, and native W4A4/reference parity remain
unsupported. Graph mode may only be considered after separate fixed-address
QSA/PLE capture and replay parity on the target stack.
