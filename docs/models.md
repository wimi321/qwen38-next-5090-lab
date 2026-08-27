# Model status

This page was substantially modified by Qwen3.8 Next 5090 Lab contributors in
2026; see [`../MODIFICATIONS.md`](../MODIFICATIONS.md). It reports only what this
downstream release has validated. For FreeToken's broader upstream model list,
consult the [upstream documentation](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md).

## Hardware-validated experimental integration

| Model | Checkpoint | Validated contract | Status |
|---|---|---|---|
| Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | Full 135,253,622,894-byte directory; text only; RTX 5090 32 GB; WSL2; TP=1; max-running=1; 8K; naive cache; graph=0 | Alpha, W4A16 compatibility |

“Hardware-validated experimental” is narrower than formal model support. The
recorded full-checkpoint 8K/stability run passed its resource gates, but the
runtime does not yet consume the checkpoint's routed-expert activation scales
and has no complete native W4A4 reference parity. See
[`qwen4-exp.md`](qwen4-exp.md) for commands, measurements, and caveats.

## Capability matrix

| Capability | Alpha status |
|---|---|
| Text prefill/decode | Validated within the profile above |
| OpenAI chat completions | Streaming and non-streaming smoke-tested |
| Thinking modes / tool-call parsing | Smoke-tested; applications must validate tool output |
| Query-Selective Attention | Dedicated Torch oracle/vectorized/Triton path; 12-layer QSA cache |
| PLE | Sparse mmap row reads; CPU FP8 decode/scale; pinned BF16 staging; GPU projection/convolution |
| Routed MoE | 512 experts, top-10; heterogeneous host/CPU/GPU offload |
| Native checkpoint W4A4 activation path | Not implemented |
| CUDA graph | Disabled; QSA/PLE capture-replay parity is pending |
| Radix prefix cache | Disabled for this model |
| Images, video, audio | Rejected by the online text-only API |
| Vision tower / MTP | Not loaded |
| Multi-request / TP>1 | Not claimed |
| 32K / 262K context | Not claimed |

The source checkpoint is multimodal. That metadata does not make this online
server multimodal: a processor, structured media transport, vision prefill,
placeholder expansion, mRoPE, and media-aware cache semantics are all missing
from the alpha contract.

## Checkpoint policy

Use the project downloader so a missing revision cannot silently fall back:

```bash
q38lab download --accept-qwen-license
q38lab download --accept-qwen-license --full-verify
```

The first command records exact pinned-download provenance outside the model
directory and hashes all configuration/tokenizer/control files. The second
also reads every weight byte and records the audited aggregate manifest digest;
the release harness always repeats that full verification.

Weights stay outside the repository. Code is Apache-2.0; model artifacts are
separately governed by the [Qwen Community License 1.0](../MODEL_LICENSES.md).

## Inherited runtime architectures

Because this repository retains the FreeToken engine, inherited adapters for
other model families remain in the source tree. They were not revalidated as
part of `v0.1.0-alpha.1`, are not covered by the RTX 5090 Qwen3.8 evidence, and
must not be inferred from the table above to be supported by this downstream
release. Compatibility fixes for those paths should normally be coordinated
with upstream FreeToken.
