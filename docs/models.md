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

## Unreleased candidate (not hardware validated)

Source version `0.2.0a1` adds a separate
`rtx5090-wsl2-256k-image` candidate profile for 262,144 total tokens and image
input. It is not listed in the hardware-validated table because the exact text
and real-image `261,120 input + 1,024 output` gates, throughput/resource limits,
100 mixed requests, and 30-minute soak have not yet produced a reviewed
release-compatible evidence directory. Do not infer v0.2 support from the
presence of code, profiles, or unit tests.

## Capability matrix

| Capability | Verified v0.1 | Unreleased v0.2 candidate |
|---|---|---|
| Text prefill/decode | Validated to 8,192 total tokens | Code/evidence gates target 262,144 total; full run pending |
| OpenAI chat completions | Streaming and non-streaming smoke-tested | Adds long-context and mixed image gates; pending full run |
| Thinking / tools | Smoke-tested; applications must validate output | Image combinations added to the pending gate |
| Query-Selective Attention | Torch/vectorized/Triton; 12-layer QSA cache | One index key per four tokens, 6.1875 GiB 256K cache, bounded SM120 top-512; full run pending |
| PLE | Sparse mmap rows; CPU FP8 decode/scale; pinned BF16 transfer | Required native `io_uring` + `O_DIRECT`, 4 GiB LRU, pinned FP8 and GPU decode/scale; full run pending |
| Routed MoE | 512 experts, top-10; heterogeneous host/CPU/GPU offload | Same W4A16 compatibility; route-aware native-NVFP4 prefill preserves raw IDs, copies active rows only for large banks, and keeps small banks/full `[E]` buffers; full run pending |
| Images | Rejected | HTTPS/data URL, image tower, mRoPE and chunk scatter implemented; not yet validated for release |
| Video / audio / MTP | Rejected / not loaded | Rejected / not loaded |
| Native checkpoint W4A4 | Not implemented | Not implemented |
| CUDA graph / radix cache | Disabled | Disabled |
| Multi-request / TP>1 | Not claimed | Not claimed |
| Context beyond 262,144 | Not claimed | Not claimed |

The source checkpoint is multimodal. The v0.1 profile remains text-only. The
v0.2 candidate includes the processor, structured media transport, vision
prefill, placeholder expansion, mRoPE and media-aware cache behavior, but those
paths do not become a supported input contract until their full-checkpoint
evidence passes every documented release gate.

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
part of `v0.1.0-alpha.1` or the v0.2 candidate, are not covered by the RTX 5090
Qwen3.8 evidence, and
must not be inferred from the table above to be supported by this downstream
release. Compatibility fixes for those paths should normally be coordinated
with upstream FreeToken.
