# Third-party notices and provenance

This file supplements, and does not replace, the copyright and attribution
notices in individual source files. Qwen3.8 Next 5090 Lab is a derivative work
of FreeToken. Its downstream modifications are described in
[`MODIFICATIONS.md`](MODIFICATIONS.md).

The audited FreeToken baseline did not contain a root `NOTICE` file. Its root
README and source headers nevertheless identify the following projects as
design influences or sources of adapted/borrowed code; those notices are
retained here and inline where the original project placed them.

## FreeToken

- Project: [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
- Audited downstream base:
  `9ef3651309fe4058672f2cc92069238dea06be1b`
- Copyright notice retained from the distribution: Copyright 2026 FreeToken
  Authors
- License: Apache License 2.0; the complete text is in [`LICENSE`](LICENSE)
- Upstream paper/citation:
  [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://arxiv.org/abs/2608.16157)

Qwen3.8 Next 5090 Lab is not an official FreeToken release. FlashML names and
marks are used only to identify the origin of this derivative work.

## Source-derived and adapted components

| Project | Provenance retained by FreeToken source | Terms / bundled text |
|---|---|---|
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | `python/freetoken/kernel/fla/` identifies the project as the GDN/linear-attention source adapted through the SGLang fork. Downstream Qwen4-Exp reuses that inherited GDN implementation. | MIT; [`licenses/flash-linear-attention-LICENSE`](licenses/flash-linear-attention-LICENSE) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | FreeToken's GGUF Q4_0 and CPU/CUDA MoE files identify llama.cpp/ggml kernel and layout sources inline. | MIT; [`licenses/llama.cpp-LICENSE`](licenses/llama.cpp-LICENSE) |
| [NVIDIA NCCL](https://github.com/NVIDIA/nccl) | `python/freetoken/kernel/csrc/include/freetoken/nccl227.h` vendors the NCCL 2.27.1 public header and retains NVIDIA's copyright notice. | Combined Apache-2.0/BSD terms; [`licenses/NCCL-LICENSE.txt`](licenses/NCCL-LICENSE.txt) |
| [SGLang](https://github.com/sgl-project/sglang) | Multiple inherited FreeToken kernels and serving paths identify specific SGLang implementations in their module headers or comments. | See the upstream project license and retained inline notices. |
| [vLLM](https://github.com/vllm-project/vllm) | Inherited NVFP4 Marlin, parser/API, sampling, and sparse-attention modules identify their vLLM sources inline. | Apache-2.0; the repository-wide Apache text is in [`LICENSE`](LICENSE); retained upstream copyright notices still apply. |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | The inherited NVFP4 b12x backend identifies FlashInfer as its source and is also loaded as an optional dependency. | Apache-2.0; see the upstream project and package metadata. |
| [LightLLM](https://github.com/ModelTC/lightllm) | Inherited rotary and function-call parsing modules identify the specific LightLLM files they adapt. | See the upstream project license and retained inline notices. |

The exact donor revision is not stated for every legacy adapted module in the
audited FreeToken baseline. This project preserves the available file-level
links and notices and flags exact-revision reconstruction as a release-review
item rather than inventing commit identifiers.

## Design acknowledgements

The upstream FreeToken README states that FreeToken was deeply inspired by
[`mini-sglang`](https://github.com/sgl-project/mini-sglang) and learned from
SGLang, vLLM, FlashInfer, flash-linear-attention, LightLLM, and llama.cpp.
Listing a project here does not imply its authors endorse this downstream.

## Runtime dependencies

Python/CUDA dependencies installed from package indexes remain under their own
licenses. They are not copied into this source repository merely because they
appear in `pyproject.toml`. Release preparation generates an SBOM from the
resolved environment; consult that release artifact for the exact dependency
versions and license metadata used in a particular build.

For the supported Linux x86_64 profile, uv resolves `sglang-kernel` from the
official SGLang wheel repository's v0.4.5 GitHub release asset and verifies the
SHA-256 digest published by GitHub. This pin is a supply-chain integrity record,
not a redistribution of the wheel.

## Model artifacts are separate

No Qwen or RadixArk weights are included in this repository. Their terms are
separate from the Apache-2.0 code terms; see [`MODEL_LICENSES.md`](MODEL_LICENSES.md).

## Reporting a missing notice

If an attribution or license text appears incomplete, do not remove the
affected source. Open a license/attribution issue with the file path, suspected
upstream URL, and supporting history, or follow the private process in
[`SECURITY.md`](SECURITY.md) if public disclosure would expose a vulnerability.
