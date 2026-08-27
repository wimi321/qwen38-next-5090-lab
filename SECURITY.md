# Security policy

Qwen3.8 Next 5090 Lab is an experimental local inference runtime, not a
hardened multi-tenant service. The `v0.1.0-alpha.1` release supports the alpha
branch only; older pre-release snapshots may not receive security fixes.

## Reporting a vulnerability

Use GitHub's private vulnerability-reporting flow for
[`wimi321/qwen38-next-5090-lab`](https://github.com/wimi321/qwen38-next-5090-lab/security/advisories/new).
If that flow is unavailable before the repository is public, contact the
maintainer privately through the verified contact method on the
[`wimi321`](https://github.com/wimi321) profile. Do not include exploit code,
credentials, private prompts, tokens, model-license acceptance records, or
unredacted logs in a public issue.

Include the release/commit, OS and WSL versions, driver/toolkit versions, GPU,
the exact checkpoint revision, the affected command/API, impact, and a minimal
reproduction. Expect an acknowledgement within seven days; timelines for a fix
or coordinated disclosure depend on severity and whether the issue is inherited
from FreeToken or another upstream.

For inherited vulnerabilities, the maintainer may coordinate privately with
[FreeToken](https://github.com/FlashML-org/FreeToken/security) or the relevant
dependency. Please do not report Qwen model behavior as a code vulnerability
unless it crosses a concrete software security boundary.

## Default security posture

- The unauthenticated API binds to `127.0.0.1` by default.
- `q38lab serve` rejects an unauthenticated non-loopback bind unless the caller
  uses `--unsafe-allow-non-loopback`. That acknowledgement does not add
  authentication, TLS, authorization, rate limiting, or tenant isolation.
- Do not port-forward the server, place it on a public interface, or expose it
  through a tunnel without a separately maintained authenticated TLS proxy and
  network controls.
- Run the project as an unprivileged WSL user. It does not need administrator or
  root privileges after host/toolkit setup.
- Keep repositories and checkpoints in WSL ext4. Treat checkpoint and tokenizer
  files as untrusted input until the pinned inventory is verified.
- Use a dedicated virtual environment. This downstream retains the
  `freetoken` import namespace and cannot be safely co-installed with the
  upstream distribution.

## Data and tool-call risks

Prompts, generated text, thinking content, tool schemas, arguments, and request
logs may contain secrets. The server and evidence tools should be used only with
data the operator is authorized to process. Review evidence archives before
publication; redaction must remove prompt/output content, user paths, usernames,
hostnames, IP addresses, tokens, and unrelated process data while retaining the
measurements needed for reproducibility.

Model output is untrusted. Applications must validate tool names, JSON arguments,
paths, URLs, shell fragments, and authorization independently. Never execute a
tool call solely because the model emitted it. Thinking/reasoning content is not
an audit log and may be incomplete or inaccurate.

## Resource exhaustion

The alpha profile is intentionally restricted to one running request and 8,192
total tokens. Raising concurrency, sequence length, output limits, cache size,
or memory ratio can exhaust GPU memory, host RAM, pinned-memory limits, disk
space, or PCIe bandwidth. The profile is a tested operating envelope, not a
sandbox against malicious clients.

The downloader may consume approximately 135 GB plus Hugging Face cache and
temporary space. Verify the destination before downloading. It must never write
weights into the source tree or accept an unpinned fallback.

## Media and remote fetches

This release is text-only and rejects image, video, and audio payloads. It does
not fetch user-supplied media URLs. Any future multimodal implementation must
add scheme/size/time limits and SSRF defenses for loopback, private, link-local,
redirected, and local-file targets before remote media is accepted.

## Dependency and release integrity

Use source archives, evidence bundles, SBOMs, and `SHA256SUMS` from the same
GitHub Release. Verify checksums before use. A release never contains model
weights. Dependency packages, CUDA kernels, the Windows driver, WSL, and the
checkpoint have their own security update channels; review them independently.

Do not treat a successful checksum as proof that a model or dependency is safe.
It proves only that bytes match the recorded artifact.
