# Model licenses and download policy

The Apache-2.0 license in [`LICENSE`](LICENSE) applies to this repository's code
and documentation. It does **not** grant rights to Qwen model weights,
tokenizers, configuration files, model outputs, model names, or trademarks.

Qwen3.8 Next 5090 Lab does not redistribute model artifacts. Its downloader is
only a reproducibility helper that transfers a pinned Hugging Face revision
directly to the user's machine after explicit acknowledgement.

## Source model

- Model: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Terms: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
- License identifier shown by the model repository:
  `qwen-community-1.0`

Read the complete license before use. Among other conditions, its current text
contains attribution/display requirements for certain large commercial
products and requires a separate Qwen license for certain commercial “Model as
a Service” or “AI Work Assistant” uses. Those defined terms, thresholds,
exceptions, and all other conditions must be read from the license itself; this
summary is not a substitute.

Passing `--accept-qwen-license` means only that the local user confirms they
reviewed the linked terms. It is not a license grant by this project, does not
record consent on Qwen's behalf, and does not determine whether a proposed use
is permitted.

## Quantized checkpoint used by this project

- Checkpoint:
  [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- Pinned revision:
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`
- Expected validated directory: 419 non-cache files,
  135,253,622,894 bytes, including 206 safetensors files
- Local manifest SHA256 from the validated copy:
  `6cc22b628ca575785e5dfdcab3c7056e79a7eac798969a145341ed1530c2a3a8`

At the time this release was prepared, the checkpoint model card described it
as a **private candidate release**, said it was a quantized version of the Qwen
source model, and directed users to the source model for license terms. It also
described routed-expert quantization with an NVIDIA Model Optimizer W4A4 recipe.

This runtime currently executes those packed routed-expert weights with BF16
activations (W4A16 compatibility). Neither the checkpoint terms nor its W4A4
quality claims imply that this different execution path has matching numerical
behavior.

## Reproducible download behavior

```bash
q38lab download --accept-qwen-license
q38lab download --accept-qwen-license --full-verify
```

The downloader must:

1. request exactly the repository and revision listed above;
2. store files outside the source repository by default;
3. stop if the revision is unavailable or its inventory/checksums do not match;
4. never silently select `main`, a newer commit, another quantization, or an
   unpinned mirror;
5. keep weights out of Git, source archives, SBOM attachments, and GitHub
   Releases.

The ordinary check verifies the pinned Hugging Face inventory and expected
size. `--full-verify` re-reads the complete local directory against its
manifest; it is intentionally much slower.

## User responsibilities

Before use or distribution, review:

- the exact source-model license and checkpoint model card at the time of use;
- any additional terms presented by Hugging Face, a mirror, or a hosting
  provider;
- the licenses of any other checkpoint, tokenizer, calibration data, or
  application component you substitute;
- privacy, export, safety, and intellectual-property rules applicable to inputs,
  outputs, and the intended deployment.

If the checkpoint is withdrawn or its revision/terms change, stop and audit the
replacement. Do not edit the pin merely to make the downloader succeed.

This document is a technical provenance record, not legal advice. For a
commercial hosted service or AI work-assistant product, obtain advice suitable
for that deployment and contact the model licensor when required by its terms.
