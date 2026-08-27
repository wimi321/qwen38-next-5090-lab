# Model status

This page was substantially modified by Qwen3.8 Next 5090 Lab contributors in
2026; see [`../MODIFICATIONS.md`](../MODIFICATIONS.md). It reports only what this
downstream release has validated. For FreeToken's broader upstream model list,
consult the [upstream documentation](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md).

## Hardware-validated experimental integration

| Model | Checkpoint | Validated contract | Status |
|---|---|---|---|
| Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | Full 135,253,622,894-byte directory; text only; RTX 5090 32 GB; WSL2; TP=1; max-running=1; 8K; naive cache; graph=0 | Alpha, W4A16 compatibility |
| Qwen3.8-Flash-Next (`v0.2.0-alpha.1`) | Same pinned checkpoint and revision | Full directory; text and images through HTTPS/data URLs; RTX 5090 32 GB; WSL2; TP=1; max-running=1; 262,144 total tokens; naive cache; graph=0; native `io_uring` + `O_DIRECT` | Hardware verified, W4A16 compatibility |

“Hardware-validated experimental” is narrower than formal model support. The
v0.1 full-checkpoint 8K/stability run remains the text-only baseline. The v0.2
full-checkpoint 262,144-token and image run passed its documented release
gates; its exact evidence is
[`results/rtx5090-2026-08-28-v02-alpha1-757872a-run7`](../results/rtx5090-2026-08-28-v02-alpha1-757872a-run7/).
The runtime does not yet consume the checkpoint's routed-expert activation
scales and has no complete native W4A4 reference parity. See
[`qwen4-exp.md`](qwen4-exp.md) for commands, measurements, and caveats.

## v0.2.0-alpha.1 validated scope

The `rtx5090-wsl2-256k-image` profile is verified only for RTX 5090/WSL2,
TP=1, one running request, naive cache, CUDA graph disabled, and W4A16
compatibility. The 262,144-token ceiling includes text, expanded image tokens,
and output. Image transport supports HTTPS and data URLs. Native `io_uring` +
`O_DIRECT` is mandatory for this profile. Audio, video, MTP, radix cache, TP>1,
concurrent requests, and totals above 262,144 tokens are unsupported.

## Capability matrix

| Capability | Verified v0.1 | Hardware-validated v0.2.0-alpha.1 |
|---|---|---|
| Text prefill/decode | Validated to 8,192 total tokens | Validated to 262,144 total tokens, including expanded image tokens and output |
| OpenAI chat completions | Streaming and non-streaming smoke-tested | Streaming/non-streaming, long-context, and mixed image gates validated |
| Thinking / tools | Smoke-tested; applications must validate output | Text and image combinations validated; applications must still validate output |
| Query-Selective Attention | Torch/vectorized/Triton; 12-layer QSA cache | One index key per four tokens and bounded SM120 top-512 path used by the validated run |
| PLE | Sparse mmap rows; CPU FP8 decode/scale; pinned BF16 transfer | Native `io_uring` + `O_DIRECT`, 4 GiB LRU, pinned FP8 staging, and GPU decode/scale used by the validated run |
| Routed MoE | 512 experts, top-10; heterogeneous host/CPU/GPU offload | Same W4A16 compatibility; route-aware native-NVFP4 prefill preserves raw IDs, copies active rows only for large banks, and keeps small banks/full `[E]` buffers |
| Images | Rejected | HTTPS/data URL, image tower, mRoPE, and chunk scatter hardware-validated |
| Video / audio / MTP | Rejected / not loaded | Unsupported |
| Native checkpoint W4A4 | Not implemented | Not implemented |
| CUDA graph / radix cache | Disabled | Disabled / unsupported |
| Multi-request / TP>1 | Not claimed | Unsupported |
| Context beyond 262,144 | Not claimed | Unsupported; over-limit requests are rejected |

The source checkpoint is multimodal. The v0.1 profile remains text-only. The
v0.2 profile supports images within the validated contract through its
processor, structured media transport, vision prefill, placeholder expansion,
mRoPE, and media-aware cache behavior. It does not support audio or video.

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
part of `v0.1.0-alpha.1` or `v0.2.0-alpha.1`, are not covered by the RTX 5090
Qwen3.8 evidence, and
must not be inferred from the table above to be supported by this downstream
release. Compatibility fixes for those paths should normally be coordinated
with upstream FreeToken.
