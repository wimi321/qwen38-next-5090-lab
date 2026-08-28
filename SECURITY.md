# Security policy

Qwen3.8 Next 5090 Lab is an experimental local inference runtime, not a
hardened multi-tenant service. Security fixes target the current alpha release;
older pre-release snapshots may not receive them.

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

Both hardware-validated profiles are restricted to TP=1 and one running
request. `rtx5090-wsl2` is limited to 8,192 total tokens;
`rtx5090-wsl2-256k-image` is limited to 262,144 total tokens and requires its
native direct-I/O checks. Raising concurrency, sequence length, output limits,
cache size, or memory ratio can exhaust GPU memory, host RAM, pinned-memory
limits, disk space, or PCIe bandwidth. These profiles are tested operating
envelopes, not sandboxes against malicious clients.

The downloader may consume approximately 135 GB plus Hugging Face cache and
temporary space. Verify the destination before downloading. It must never write
weights into the source tree or accept an unpinned fallback.

## Media and remote fetches

The v0.1 profile remains text-only. The v0.2 profile accepts images only through
base64 `data:` URLs or HTTPS. It permits at most four images, 20 MiB per image,
40 MiB total, and a ten-second fetch deadline. It rejects local files, plain
HTTP, URL credentials, loopback, private, link-local or reserved addresses,
mixed public/non-public DNS answers, DNS rebinding, unsafe redirects, invalid
MIME types, audio, and video. Each redirect target is resolved and audited
again, and HTTPS retains certificate verification and SNI for the original
hostname while connecting to an already validated public address.

These controls reduce SSRF exposure but do not make the unauthenticated local
server suitable for hostile multi-tenant traffic. Operators should prefer data
URLs for sensitive images, review outbound network policy, and avoid exposing
the service beyond loopback.

## Dependency and release integrity

Use source archives, evidence bundles, SBOMs, and `SHA256SUMS` from the same
GitHub Release. The optional binary companion has a separate
`WHEEL-SHA256SUMS` and provenance record bound to its tag and commit. Verify
checksums before use. A release never contains model weights. Dependency
packages, CUDA kernels, the Windows driver, WSL, and the checkpoint have their
own security update channels; review them independently.

Do not treat a successful checksum as proof that a model or dependency is safe.
It proves only that bytes match the recorded artifact.
