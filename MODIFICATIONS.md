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
- Added loader mapping for `model.language_model.*`; visual and MTP weights are
  intentionally skipped. The online API remains text-only.

### PLE auxiliary bank

- Added an auxiliary-bank lifecycle separate from ordinary model state so the
  approximately 51 GB PLE table is never materialized as one tensor or copied
  wholesale to the GPU.
- Added checkpoint-defined bigram/trigram hashing, 16 n-gram embedding heads,
  the layer-2 PLE projection/convolution, and request-local history/state.
- Added read-only safetensors mmap row banks. Requested FP8 rows are decoded and
  scaled on the CPU, converted to pinned BF16 staging, and transferred to the
  GPU for PLE projection and convolution.
- Explicitly rejects unsupported cache/state combinations instead of silently
  reusing invalid PLE state.

### MoE and RTX 5090 integration

- Added non-power-of-two/wide top-k handling needed by the model's top-10
  router.
- Added WSL-aware host pin-budget resolution and an automatic pageable-layer
  selection for the pinned 135 GB checkpoint.
- Registered the text-only model and added the `qsa_triton` launch surface.
- Preserves packed NVFP4 routed weights while executing BF16 activations. This
  is W4A16 compatibility, not native checkpoint W4A4 parity.

### Product and release surface

- Added the `q38lab` source-only CLI for environment diagnosis, pinned model
  download/verification, the RTX 5090 profile, API smoke tests, and evidence
  collection.
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

## Publication review checklist

The following values cannot be known until release preparation runs and must
be reviewed rather than guessed:

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
