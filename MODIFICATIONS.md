# Downstream modifications

Qwen3.8 Next 5090 Lab is a modified distribution of
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken), provided
under the Apache License 2.0. It is maintained independently at
[`wimi321/qwen38-next-5090-lab`](https://github.com/wimi321/qwen38-next-5090-lab).

The audited upstream baseline is:

```text
9ef3651309fe4058672f2cc92069238dea06be1b
```

The repository retains the upstream history. Files changed from that baseline
carry a modification notice or are new downstream files; this inventory gives
the corresponding high-level description. The Apache-2.0 license remains in
[`LICENSE`](LICENSE), and retained source-level notices remain authoritative.

## Material changes

### Query-Selective Attention

- Added an explicit QSA attention type, configuration group, and dedicated
  Torch/reference, vectorized CUDA, and Triton paths.
- Added four-token block selection, bounded top-k token expansion, partial RoPE,
  tail handling, and a cache that stores both main K/V and QSA index keys.
- Generalized cache-pool sizing so storage is allocated only for the 12 QSA
  layers in the pinned model.
- The v0.2 alpha persists one compressed index key per four tokens, with a
  request-local pending tail and the original three-axis positions. Its
  262,144-token main K/V plus index budget is 6.1875 GiB.
- Added `qsa_triton_sm120`, which bounds selector scores to a 128 MiB FP32
  workspace and uses a top-512 CUDA specialization. That specialization is an
  Apache-2.0 adaptation of
  `python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh` from
  [`yhfgyyf/sglang-qwen38-flash-next-sm120`](https://github.com/yhfgyyf/sglang-qwen38-flash-next-sm120)
  at exact commit `30edf3503961a471b25150aa890f8166031b5738`.
- Retained the PyTorch oracle and original `qsa_triton` backend as visible
  correctness fallbacks. The new CUDA selector has independent boundary and
  adversarial wide-row tests; the reviewed full-model v0.2 evidence complements
  rather than replaces those checks.
- CUDA graph capture/replay remains disabled for QSA until it passes a real
  target-GPU parity gate.

### Qwen4-Exp text tower

- Added a distinct `qwen4_exp` model package and checkpoint configuration
  parser for the Qwen3.8-Flash-Next text architecture.
- Added four-stream gated residual reads/writes and final mixing; parameterized
  the inherited GDN output gate so this model can use `sigmoid` without changing
  the default behavior of older models.
- Added the 48-layer hybrid layout (36 GDN and 12 QSA layers), 512 routed
  experts with softmax top-10 routing, and the shared expert path.
- Added loader mapping for `model.language_model.*`. The v0.1 profile skips
  visual and MTP weights and remains text-only.
- The v0.2 alpha loads the Qwen4-Exp vision configuration and
  `model.visual.*` weights, including the 27-layer tower, patch projection,
  position interpolation, and merger. Visual embeddings replace expanded
  image placeholders before the hidden state is copied into four residual
  streams. MTP remains skipped.
- Added processor-generated `image_grid_thw`, structured image spans, three-axis
  interleaved mRoPE plus rope delta, and a shape/dtype-preserving tensor wire
  format for tokenizer-to-scheduler messages.

### PLE auxiliary bank

- Added an auxiliary-bank lifecycle separate from ordinary model state so the
  approximately 51 GB PLE table is never materialized as one tensor or copied
  wholesale to the GPU.
- Added checkpoint-defined bigram/trigram hashing, 16 n-gram embedding heads,
  the layer-2 PLE projection/convolution, and request-local history/state.
- Added read-only safetensors mmap row banks. Requested FP8 rows are decoded and
  scaled on the CPU, converted to pinned BF16 staging, and transferred to the
  GPU for PLE projection and convolution.
- Added a separate v0.2 production backend using the Linux `io_uring` UAPI with
  `O_DIRECT`: 4 KiB-aligned reads, queue depth 512, at most 4,096 pages per
  batch, one global 4 GiB native LRU split across physical shards, and
  next-prefill-chunk prefetch. The profile fails closed if the native extension
  or direct-I/O filesystem contract is unavailable.
- For the v0.2 production path, requested FP8 rows use double-buffered pinned
  staging and are decoded/scaled on the GPU before conversion to BF16. The mmap
  and synchronous direct readers remain test/debug fallbacks and are not an
  accepted substitute for the 256K profile.
- Added telemetry for storage bytes, cache hits/misses and eviction, queued
  prefetch, wait time, staged bytes, GPU-decoded rows, and page faults. The
  release harness cross-checks server telemetry against raw evidence instead
  of inferring I/O from request latency.
- Explicitly rejects unsupported cache/state combinations instead of silently
  reusing invalid PLE state.

### Image request and prefill path

- Extended OpenAI chat messages to retain structured text/image content and to
  carry validated image bytes to the tokenizer without forwarding source URLs.
- Added HTTPS acquisition with TLS hostname validation against a pre-audited
  public IP, DNS-rebinding and redirect checks, plus base64 data URL decoding.
  Requests are limited to four images, 20 MiB each, 40 MiB total, and ten
  seconds; HTTP, local files, loopback/private/link-local/reserved destinations,
  audio, and video are rejected.
- Added pinned Transformers `5.16.1` processor integration, image placeholder
  expansion, image-only mRoPE construction, and processor-output validation.
- Images are encoded once. The full merged BF16 embedding is retained in
  pageable CPU memory, while only rows intersecting the current 512-token
  prefill chunk are transferred and scattered. Image requests bypass shared
  text prefix reuse, and media bytes, visual plans, and mRoPE state are cleared
  when the request is freed.
- Added OpenAI-style `context_length_exceeded` reporting that separates text,
  expanded image, and requested output tokens against the 262,144 total.

### MoE and RTX 5090 integration

- Added non-power-of-two/wide top-k handling needed by the model's top-10
  router.
- Added WSL-aware host pin-budget resolution and an automatic pageable-layer
  selection for the pinned 135 GB checkpoint.
- Added an opt-in, route-aware prefill movement path for native NVFP4 banks.
  The current layer's router produces a bounded 512-entry active mask; raw
  expert IDs remain the GPU-buffer row indices and the existing full `[E]`
  double buffers remain allocated. Banks with per-expert rows of at least
  256 KiB copy only coalesced active-ID runs, while smaller banks copy the full
  layer to avoid the CUDA runtime's synchronous mixed-small-entry behavior.
- Registered banks use direct asynchronous copies when the batched runtime path
  is available. LOCKED/PAGEABLE layers copy the same selected raw-ID runs
  through two 32 MiB pinned bounce slabs. Sparse prefill disables eager
  next-layer full-bank prefetch and the separate prefill hit-D2D path because
  the active set is not known until the current router completes.
- Added `q38lab.moe_prefill` telemetry for summed active and possible expert
  rows, copied and hypothetical full-bank bytes, and their derived fractions.
  The counters preserve the whole-layer contribution of small banks, so byte
  fraction is not inferred from row fraction. The v0.2 profile fails
  closed unless double-buffered overlap and the native `nvfp4` layout are
  available; none of these source changes is a hardware-performance claim.
- Added a model-declared 512 MiB runtime guard to automatic expert-cache sizing;
  the public profile remains at `memory-ratio=0.89`, while late LRU fill retains
  headroom for the preview's strict peak-VRAM gate.
- Registered the text-only model and added the `qsa_triton` launch surface.
- Added the opt-in `qsa_triton_sm120` and image tower launch surfaces. The v0.2
  planner reserves 6.1875 GiB QSA cache, 128 MiB selector workspace, vision
  weights, and runtime headroom before shrinking the GPU expert cache; it
  refuses a fixed geometry that cannot fit the resolved planning budget. The
  separate release evidence enforces the absolute peak-VRAM envelope.
- Preserves packed NVFP4 routed weights while executing BF16 activations. This
  is W4A16 compatibility, not native checkpoint W4A4 parity.

### Product and release surface

- Added the `q38lab` source-only CLI for environment diagnosis, pinned model
  download/verification, the RTX 5090 profile, API smoke tests, and evidence
  collection.
- Added the `rtx5090-wsl2-256k-image` alpha profile and corresponding
  doctor, image/security smoke, long-context, runtime-telemetry, and evidence
  contracts. The original `rtx5090-wsl2` profile and v0.1 evidence are retained
  unchanged.
- Replaced upstream publication automation with hosted validation and
  source-only release workflows. This downstream does not publish to the
  upstream `freetoken` PyPI name and does not redistribute model weights.
- Removed the inherited wheel installer, wheel/PyPI build helpers, prebuilt
  kernel-cache packaging project, official logo assets, and community artwork;
  these remain available in the preserved upstream history but are not a
  downstream distribution surface.
- Replaced upstream-facing branding, setup, contribution, security, and model
  documentation with downstream-specific material while preserving attribution.
- Replaced the ambiguous SGLang cu130 simple-index source (which advertised two
  hashes for one wheel URL) with the official v0.4.5 x86_64 GitHub release asset
  pinned to its GitHub-reported SHA-256 digest.

## Validation boundary

The `v0.1.0-alpha.1` evidence applies only to:

- the complete `RadixArk/Qwen3.8-Flash-Next-NVFP4` directory at revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`;
- one RTX 5090 with TP=1 and one running request;
- 8,192 total tokens, naive cache, graph disabled, and W4A16 compatibility;
- text input/output only.

It does not establish W4A4 numerical parity, multimodal serving, MTP, radix
prefix reuse, 32K/262K context, multi-request stability, or absence of Windows
host paging. See [`docs/qwen4-exp.md`](docs/qwen4-exp.md).

The reviewed `0.2.0a1` full-checkpoint run is tracked at
`results/rtx5090-2026-08-28-v02-alpha1-757872a-run7/`. It validates the native
262,144-token and image paths only for one RTX 5090 under WSL2, TP=1, one
running request, naive cache, graph disabled, and W4A16 compatibility. The run
proved all of the following:

- exact `261,120 input + 1,024 output` completion for both a text request and a
  request containing a real image;
- 8K regression plus 32K and 128K, Needle-in-a-Haystack, deterministic
  OCR/object/chart results, stream/non-stream consistency, thinking, and tool
  calls combined with images;
- 261K TTFT below 15 minutes, a 256-token steady decode above 5 tok/s,
  acceptance-window peak VRAM below 31 GiB, WSL RSS below 105 GiB, and WSL
  swap zero;
- 100/100 mixed text/image sequential requests and at least 30 minutes without
  a crash or monotonic leak; and
- recorded bounded selector workspace, cold plus three warm PLE measurements,
  PLE I/O/LRU/wait/page-fault telemetry, vision latency, image tokens, prefill
  chunk timing, and internally consistent route-aware MoE active-row and
  copied-byte counters;
- a release-qualified native PLE probe covering every shard's first/last row
  and deterministic bigram/trigram hash hits, with exact GPU-decoded parity to
  independent safetensors slices; and
- a 30-minute soak that continues alternating text and image requests rather
  than switching to a text-only leak window.

The release evidence contains ten files, exhaustive checksums, and raw resource
samples spanning both pre-acceptance pytest/preflight work and the formal API
acceptance window. README resource figures are explicitly the acceptance-window
maxima. During human review, the 13 ordinary 8K regression records were
namespaced from `prompt-8176` to `regression-prompt-8176` to distinguish them
from the NIAH request; only the case labels and resulting checksums changed.
Video, audio, MTP, radix cache, TP>1, multi-request scheduling, native W4A4
parity, and contexts beyond 262,144 remain outside the v0.2 contract.

## Publication review checklist

The following controls remain mandatory when the release tag is prepared:

- [ ] Record the final public release commit after the downstream-only history
      rewrite; preserve the private pre-rewrite bundle and mapping outside the
      public release assets.
- [ ] Regenerate all evidence, SBOM, source archive, and checksums from that
      final release commit; confirm the generated README block is unchanged in
      `--check` mode.
- [ ] Record exact source revisions for any legacy FreeToken modules whose
      inline notices say they were adapted from third-party repositories. The
      audited FreeToken baseline did not provide a single consolidated revision
      manifest, so this document does not invent one.
- [ ] Verify every modified upstream file still carries a prominent downstream
      modification notice and all pre-existing copyright/attribution text.
- [ ] Review the source-model and checkpoint terms at their pinned URLs; stop if
      either is withdrawn or materially changed.
- [ ] Complete a human license/trademark review before distributing binaries or
      offering any hosted service. This repository's documentation is not legal
      advice.
