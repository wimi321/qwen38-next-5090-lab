# Qwen3.8-Flash-Next (Qwen4-Exp) text-only reproducibility record

This document was substantially modified by Qwen3.8 Next 5090 Lab contributors
in 2026; see [`../MODIFICATIONS.md`](../MODIFICATIONS.md). It describes an
unofficial FreeToken downstream, not an upstream FreeToken support claim.

This page is the reproducibility record for the first Qwen4-Exp milestone. It
targets exactly
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
at revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

The integration is experimental until every item in [Acceptance](#acceptance)
has a recorded result. Its first milestone is deliberately narrow:

- text only, one RTX 5090, tensor parallel size 1;
- the complete HF checkpoint, read directly without FTW conversion;
- an 8,192-token total context budget and the naive cache;
- MTP, vision, video, audio and radix prefix reuse disabled.

This is not a claim of 32K or 262K support. The PLE auxiliary table remains a
host-mapped, row-on-demand bank; never load the complete table onto the GPU.
Decode CUDA graphs are also disabled during the initial bring-up: PLE row
staging now has a fixed-address replay seam, but the new QSA Triton kernels
still need real capture/replay parity on the target RTX 5090 before graph mode
is a valid optimization.

The pinned checkpoint declares a ModelOpt **W4A4** recipe for routed experts.
FreeToken's existing NVFP4 expert backends preserve its packed E2M1 weights,
block scales and global scales, but currently execute them with BF16
activations (**W4A16 compatibility**); the per-projection activation scales are
not consumed. The loader emits a prominent warning when it detects this
metadata. This is sufficient only for the experimental hardware bring-up: it
is not checkpoint-numerically equivalent and does not inherit the checkpoint's
published W4A4 quality claims. Formal support requires a W4A4 execution path
and reference/GPU parity.

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

The `0.89` ratio is the validated RTX 5090 setting for this checkpoint.  A
ratio of `0.90` reached 31,847 MiB after an 8,176-token prompt and missed the
strict `<31 GiB` acceptance threshold; `0.89` kept the same request at 31,542
MiB.  Do not round this value back to the CLI default in the reproducibility
record.

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

The following result was measured on 2026-08-27.  The runtime code was
`a0c07783cdd45b9f4b31b4a938c2bff535f07137`; the later documentation-only
commit does not change the measured tree.  Each prompt class used three
warm-ups followed by ten measured SSE requests with greedy sampling and an
eight-token output cap.

| Field | Record |
|---|---|
| Upstream base / checkpoint revision | `9ef3651` / `7b71922` (the evidence manifest records the downstream release commit) |
| Driver / CUDA toolkit / torch / Triton | 591.86 / 13.3 / 2.11.0+cu130 / 3.6.0 |
| Checkpoint quant recipe / executed expert path | W4A4 / W4A16 compatibility |
| Cold-start time (3 runs) | approximately 66.0 s (log resolution), 107.272 s, 162.509 s |
| Peak GPU memory | 31,567 MiB (30.827 GiB), sampled once per second |
| Peak WSL RSS / WSL swap | 71,192,992 kB (67.895 GiB) / 0 kB |
| Windows pagefile | pre-existing `D:\pagefile.sys`, 131,072 MiB allocated; 814 MiB system-wide current usage after the run; configuration was not changed |
| Final scheduler page faults | 19,100,768 minor / 1,673 major |
| PCIe receive/transmit | peak 60,710 / 9,970 MB/s; 300-s sampled sums 4,184,741 / 573,790 MB |
| Expert/cache geometry | CPU layers 0-7 and 41-47 (15, pageable); MoE cache 6,558 slots; resolver pages 8,273; allocated KV 8,192 tokens |
| Stability gate | 100/100 on runtime HEAD in 234.858 s; p50/p95/max 2.345/2.363/2.555 s; first/last-ten mean 2.369/2.344 s |
| Test suite | `python -m pytest -m 'not slow'`: 1,454 passed, 9 skipped, 11 deselected in 95.99 s |

The one-token case below means one user-content token; the rendered chat prompt
contains 13 tokens.  TTFT is client-observed time to the first non-empty SSE
content delta.  "Effective prefill" is rendered prompt tokens divided by that
end-to-end TTFT, so it includes fixed request and CPU-staging overhead and is
not a pure kernel benchmark.  Decode throughput uses the remaining six output
tokens after TTFT.

| Rendered prompt / completion tokens | TTFT p50 / p95 | Total p50 / p95 | Effective prefill p50 | Decode p50 |
|---|---:|---:|---:|---:|
| 13 / 7 | 2.238 / 2.245 s | 2.653 / 2.673 s | 5.80 tok/s | 14.31 tok/s |
| 128 / 7 | 2.295 / 2.309 s | 2.712 / 2.725 s | 55.76 tok/s | 14.34 tok/s |
| 2,048 / 7 | 3.411 / 3.419 s | 3.831 / 3.851 s | 600.29 tok/s | 14.26 tok/s |
| 8,176 / 7 | 7.225 / 7.238 s | 7.643 / 7.667 s | 1,130.83 tok/s | 14.24 tok/s |

The final 100-request run left `VmHWM` unchanged and moved `VmRSS` by only
about 0.5 MiB between its pre-run and midpoint/final snapshots.  WSL had no
configured swap.  The Windows host already had a 128 GiB pagefile; it was not
created, resized or selected as a workaround for this run.  WMI reports only
aggregate host usage, so the observed 814 MiB cannot prove zero host paging by
the WSL VM.  Streaming and non-streaming returned the same deterministic text,
`reasoning_effort=high` produced `reasoning_content`, and a required tool
request produced `get_weather({"city": "Paris"})` with
`finish_reason=tool_calls`.

Use `ft ctl --json stats` for FreeToken's TTFT, throughput, request count and
VRAM fields. Cross-check process RSS/page faults through `/proc/<pid>/status`
or `pidstat -r`, GPU memory through `nvidia-smi`, and PCIe traffic through an
NVIDIA profiler or `nvidia-smi dmon`. Sample throughout the run rather than
only before and after it. For example, this records framebuffer memory and
PCIe receive/transmit throughput once per second:

The current control API does not expose the internal MoE cache hit/miss
counters, so do not report a hit rate for this milestone. Record the resolved
cache geometry from startup and mark hit rate as unavailable until telemetry is
added rather than inferring it from PCIe traffic.

```bash
mkdir -p "$HOME/qwen4-exp-metrics"
nvidia-smi dmon -i 0 -s mt -d 1 -o DT \
  -f "$HOME/qwen4-exp-metrics/nvidia-dmon.log"
```

## Acceptance

Do not mark this model supported until all of the following are true on the
pinned full checkpoint:

- CPU/reference parity, tiny-model prefill/decode parity, weight-map and cache
  reuse tests pass, followed by `python -m pytest -m 'not slow'` and the GPU
  QSA/cache tests.
- Before CUDA graphs are enabled, PLE row staging, request-local token/conv
  state, and QSA all pass fixed-address capture/replay parity on the target
  GPU; the initial `--graph 0` run does not satisfy that later optimization
  milestone by itself.
- The loader skips `model.visual.*` and `mtp.*`, maps the routed NVFP4 experts
  to expert offload, and never materializes the full PLE table on the GPU.
- A real W4A4 expert path consumes the checkpoint's activation-scale contract
  and passes CPU-oracle plus SM120 prefill/decode parity. Until then, W4A16
  compatibility measurements are experimental evidence only and this model
  must not be listed as formally supported.
- Greedy 1, 128, 2K and 8K-boundary requests pass; thinking on/off, tool calls,
  streaming and non-streaming agree.
- Peak VRAM is below 31GiB, WSL RSS below 105GiB, `swapon --show` stays empty,
  and no Windows pagefile is used as an escape hatch.
- One hundred sequential requests or a continuous 30-minute run finishes with
  no crash, OOM or monotonic memory growth.
- Three cold starts and three warm-ups plus ten measured requests have recorded
  TTFT, prefill/decode throughput, PCIe traffic, page faults, RSS and VRAM.

Vision/image support belongs to a later branch. Until its processor, media
transport, vision prefill and cache semantics exist, media requests are outside
this integration's supported input contract and no media result is valid.

The recorded run satisfies the first text-only hardware milestone, including
the 8K and stability resource gates.  It remains experimental rather than a
formal supported-model claim because W4A4 execution, graph-mode parity, MTP and
multimodal serving are deliberately outside this result.

The WSL `swap=0` condition is proven.  A stronger claim that no host pagefile
existed or was touched is not proven because the pre-existing Windows pagefile
had non-zero aggregate system usage; keep that caveat with any published
benchmark result.
