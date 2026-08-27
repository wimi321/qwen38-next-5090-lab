# Changelog

All notable downstream changes are documented here. Upstream FreeToken history
before the audited base remains available in Git; see
[`MODIFICATIONS.md`](MODIFICATIONS.md) for provenance.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.1] - Unreleased

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

### Known limitations

- Expert activations are BF16; this is W4A16 compatibility for a checkpoint
  whose routed experts declare W4A4, not native W4A4 parity.
- Vision, image/video/audio requests, MTP, radix prefix reuse, CUDA graph,
  TP>1, multi-request scheduling, and context beyond 8K are not supported or
  claimed by this alpha.
- Formal end-to-end Transformers/reference logits parity is pending.
- WSL swap was zero in the recorded run, but a pre-existing Windows pagefile
  prevents a claim that host paging was absent.

[0.1.0-alpha.1]: https://github.com/wimi321/qwen38-next-5090-lab/releases/tag/v0.1.0-alpha.1
