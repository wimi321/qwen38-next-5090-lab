# Qwen3.8-Flash-Next (Qwen4-Exp) text-only bring-up

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
git clone https://github.com/FlashML-org/FreeToken.git
cd FreeToken
git switch --create codex/qwen4-exp-text \
  9ef3651309fe4058672f2cc92069238dea06be1b
# Apply the reviewed Qwen4-Exp text implementation commits to this branch.

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[accel]'
```

The implementation commit is part of the result and must also be recorded;
the SHA above is its audited upstream merge base, not a substitute for the
Qwen4-Exp changes.

Download only the pinned revision. If that commit is unavailable, stop and
audit the replacement rather than silently selecting another revision:

```bash
MODEL_DIR="$HOME/models/qwen38-flash-next-nvfp4-7b71922"
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --local-dir "$MODEL_DIR"
```

Create a local checksum manifest once, preserve it with the test record, and
verify it before subsequent runs:

```bash
cd "$MODEL_DIR"
find . -type f ! -path './.cache/*' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum > ../qwen38-flash-next-nvfp4-7b71922.sha256
sha256sum -c ../qwen38-flash-next-nvfp4-7b71922.sha256
```

## RTX 5090 launch

Run from the source environment. These are parser-backed FreeToken flags; the
explicit sequence limit keeps the first milestone honest, while `--num-tokens`
sets the total KV capacity rather than the checkpoint's advertised context:

```bash
source "$HOME/src/FreeToken/.venv/bin/activate"
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
  --memory-ratio 0.90 \
  --cache-type naive \
  --attention-backend qsa_triton \
  --graph 0 \
  --moe-backend offload \
  --moe-cache-auto \
  --nvfp4-backend auto
```

Do not force the original four-layer estimate. This checkpoint's expert banks
occupy about 63.46 GiB, while a 112 GiB WSL instance receives a FreeToken pin
budget of about 44.02 GiB. Four CPU layers would still leave about 58.17 GiB to
pin and can fail in `cudaHostRegister`. With `--moe-cpu-layers` omitted, the WSL
budget resolver selects the minimum safe head-and-tail set (15 layers on this
configuration: 0-7 and 41-47), leaving about 43.63 GiB pinned. Keep the first
startup on auto and record the resolved layer list from the log. Passing the
count `--moe-cpu-layers 15` is not equivalent because count syntax distributes
layers evenly through the model.

The checkpoint's PLE bank is separate from the MoE expert cache. Do not enable
a Windows pagefile or WSL swap to make a failing configuration appear to pass.
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

Record three cold starts, then perform three warm-up requests followed by ten
measured requests for each prompt size. Preserve the server log, the checksum
manifest and a table containing:

| Field | Record |
|---|---|
| Implementation HEAD / upstream base / checkpoint revision | / `9ef3651...` / `7b71922...` |
| Driver / CUDA toolkit / torch | |
| Checkpoint quant recipe / executed expert path | W4A4 / W4A16 compatibility |
| Cold-start time (3 runs) | |
| Prompt tokens / output tokens | |
| TTFT and prefill/decode token/s | |
| Peak GPU memory | |
| Peak WSL RSS and major/minor page faults | |
| PCIe receive/transmit traffic | |
| Expert-cache geometry | |
| Test duration / completed requests | |

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
  reuse tests pass, followed by `pytest -m 'not slow'` and the GPU QSA/cache
  tests.
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
