# Changelog

All notable downstream changes are documented here. Upstream FreeToken history
before the audited base remains available in Git; see
[`MODIFICATIONS.md`](MODIFICATIONS.md) for provenance.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-alpha.1] - 2026-08-28

This developer preview is hardware-validated only for the narrow
`rtx5090-wsl2-256k-image` contract described below. The reviewed full-checkpoint
record is tracked at
`results/rtx5090-2026-08-28-v02-alpha1-757872a-run7/`.

### Added

- `rtx5090-wsl2-256k-image` profile with a 262,144-token total
  context budget, 512-token prefill chunks, TP=1, one running request, naive
  cache, graph disabled, and fail-closed memory planning.
- Compressed QSA index-key persistence at one row per four tokens, request-local
  tail state and mRoPE positions, and a 6.1875 GiB full-context QSA cache budget.
- `qsa_triton_sm120` with a bounded 128 MiB FP32 score workspace and an
  Apache-2.0 top-512 CUDA specialization adapted from
  `yhfgyyf/sglang-qwen38-flash-next-sm120` at commit
  `30edf3503961a471b25150aa890f8166031b5738`; Torch and Triton correctness
  fallbacks remain available to v0.1/debug paths, while the v0.2 profile
  requires the native kernel and fails closed on any fallback.
- Native Linux `io_uring` + `O_DIRECT` PLE reads with 4 KiB alignment, queue
  depth 512, batches up to 4,096 pages, one globally bounded 4 GiB native LRU,
  next-chunk prefetch, and double-buffered pinned FP8 staging for GPU
  decode/scale to BF16.
- Route-aware native-NVFP4 MoE prefill movement for the v0.2 profile. A
  bounded expert mask discovers selected raw IDs after each layer's router;
  coalesced active rows move for banks with per-expert rows of at least 256 KiB,
  while smaller banks still move as one whole-layer entry. The implementation
  preserves raw IDs and full `[E]` GPU double buffers, uses direct registered
  copies or the fixed 64 MiB bounce allocation according to host residency, and
  exports active/possible-row plus copied/full-byte telemetry. The reviewed run
  validates this execution path, not a general performance advantage.
- Opt-in Qwen4-Exp image tower and weight mapping, pinned Transformers processor,
  placeholder expansion, `image_grid_thw`, three-axis interleaved mRoPE, and
  visual embedding injection before four-stream residual replication.
- Per-chunk visual embedding scatter from a CPU-resident BF16 plan, with image
  requests excluded from shared text prefix reuse and all media state released
  at request completion.
- OpenAI structured `image_url` input for HTTPS and base64 data URLs, bounded at
  four images, 20 MiB each, 40 MiB total, and a ten-second deadline. Local/HTTP,
  loopback/private/link-local/reserved endpoints, rebinding, unsafe redirects,
  invalid MIME types, audio, and video fail with explicit client errors.
- v0.2 evidence schemas and gates for 32K/128K/261,120 prompts, exact text and
  image `261,120 input + 1,024 output` boundaries,
  selector/PLE/vision/chunk/MoE-prefill telemetry, mixed-request stability/soak,
  cold/warm PLE measurements, and a native all-shard-boundary plus deterministic
  hash-row parity probe against independent safetensors slices.

### Changed

- `q38lab doctor` now reports QSA, PLE, vision, MoE, disk, ext4,
  `O_DIRECT`, `io_uring`, and locked-memory budget details for the new profile.
- Context overflow errors account separately for rendered text tokens, expanded
  image tokens, requested output, and the 262,144 total limit.
- Release attribution records the SGLang Qwen3.8 integration, PLE NVMe, and
  SM120 QSA design references without importing their 96 GB, MTP, CUDA Graph,
  or performance claims.

### Hardware validation

- The pinned 135,253,622,894-byte checkpoint completed both exact
  `261,120 input + 1,024 output` boundaries, including one with a real image;
  a one-token excess returned `context_length_exceeded`.
- 8K, 32K, 128K, and 261,120-token Needle-in-a-Haystack cases all passed. The
  261K client-observed TTFT remained below 15 minutes and a 256-token steady
  decode exceeded 5 tok/s.
- Peak resources during the formal API acceptance window remained below 31 GiB
  VRAM and 105 GiB WSL RSS, with WSL swap at zero. Raw resource samples also
  include pre-acceptance pytest/preflight activity and are retained for audit.
- Deterministic OCR/object/chart, image thinking and tool calling, streaming
  parity, 100/100 mixed sequential requests, and a 30-minute alternating
  text/image soak passed without a monotonic memory leak.
- The full non-slow suite recorded 1,667 passed and zero failed tests; the PLE
  probe covered real rows across all 128 shards with native direct I/O and GPU
  FP8 decode attestation.

## [0.1.0-alpha.1] - 2026-08-27

### Added

- Text-only Qwen4-Exp integration for the pinned full
  `RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint.
- QSA reference, vectorized, and Triton paths with dedicated main K/V and
  index-key caching for the model's 12 sparse-attention layers.
- Four-stream gated residuals and a final mixer across 36 GDN and 12 QSA layers.
- Sparse safetensors mmap PLE row bank with CPU FP8 decode/scale, pinned BF16
  staging, GPU projection/convolution, and request-local state.
- Top-10 routed MoE support and WSL-aware heterogeneous expert offload for a
  single 32 GB RTX 5090.
- `q38lab doctor`, `download`, `serve`, `smoke`, and `bench` source CLI.
- Source-only CI/release evidence, SBOM/checksum generation, bilingual project
  documentation, and downstream license/provenance records.

### Changed

- Project identity, package metadata, repository links, contribution policy,
  and release automation now identify this unofficial downstream rather than an
  official FreeToken distribution.
- The alpha profile fixes TP=1, max-running=1, 8K, naive cache, graph disabled,
  memory ratio 0.89, MoE offload, and automatic expert-cache/pageable-layer
  sizing.
- uv installs on Linux x86_64 use the official SGLang v0.4.5 cu130 release
  wheel pinned by SHA-256, avoiding an ambiguous duplicate-hash index entry.
- `q38lab doctor` accepts the measured WDDM display reservation while still
  requiring at least 30,000 MiB free before the fixed RTX 5090 profile starts.
- OpenAI model metadata, FastAPI documentation, and server help identify this
  downstream instead of presenting it as the official FreeToken service.
- The smoke tool-call gate disables thinking and uses the same required-tool
  contract as the release harness, so its 128-token budget measures parsing
  instead of being consumed by hidden reasoning.

### Known limitations

- Expert activations are BF16; this is W4A16 compatibility for a checkpoint
  whose routed experts declare W4A4, not native W4A4 parity.
- Vision, image/video/audio requests, MTP, radix prefix reuse, CUDA graph,
  TP>1, multi-request scheduling, and context beyond 8K are not supported or
  claimed by this alpha.
- Formal end-to-end Transformers/reference logits parity is pending.
- WSL swap was zero in the recorded run, but a pre-existing Windows pagefile
  prevents a claim that host paging was absent.

[0.2.0-alpha.1]: https://github.com/wimi321/qwen38-next-5090-lab/releases/tag/v0.2.0-alpha.1
[0.1.0-alpha.1]: https://github.com/wimi321/qwen38-next-5090-lab/releases/tag/v0.1.0-alpha.1
