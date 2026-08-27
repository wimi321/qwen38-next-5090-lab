# Contributing to Qwen3.8 Next 5090 Lab

This guide replaces the upstream FreeToken contribution guide for this modified
downstream distribution. Contributions to FreeToken itself must follow
[FreeToken's current policy](https://github.com/FlashML-org/FreeToken/blob/main/CONTRIBUTING.md)
and be submitted to that project separately.

## Before opening an issue

- Read the [alpha contract](README.md#verified-alpha-contract),
  [model status](docs/models.md), and [reproducibility record](docs/qwen4-exp.md).
- Search existing issues. Do not report a capability as a regression while the
  versioned [model status](docs/models.md) still marks it out of scope.
- Use the matching issue form. Include text logs rather than screenshots and
  remove secrets, prompts, outputs, usernames, host paths, and tokens.
- Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).

## Human accountability and AI assistance

AI-assisted contributions are welcome, but the submitting human is responsible
for every line, claim, test, and benchmark. A maintainer must be able to explain
why the change is correct without asking an agent to reconstruct it.

Unreviewed agent-only submissions are not accepted. Do not submit invented APIs,
source revisions, license claims, benchmark results, or hardware validation. If
AI materially assisted the change, disclose the tool in the pull request and
include an `Assisted-by:` trailer in downstream commits where appropriate.

## Development setup

Use Linux x86-64, preferably the documented WSL2/Ubuntu 24.04 environment:

```bash
git clone https://github.com/wimi321/qwen38-next-5090-lab.git
cd qwen38-next-5090-lab
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[accel,dev]'
q38lab doctor
```

Keep a checkpoint outside the repository. Never add model shards, Hugging Face
caches, API tokens, local manifests containing user paths, or raw request logs
to a commit.

## Pull-request shape

- Start substantive features with an issue. Keep one concern per pull request.
- Explain the behavior before and after, why the chosen layer is responsible,
  and what remains unsupported.
- Add a regression test that fails before the fix and passes after it.
- Preserve all upstream copyright and provenance text. Modified upstream files
  must carry the downstream modification notice and new third-party-derived code
  must identify its source and license.
- Update `MODIFICATIONS.md`, `THIRD_PARTY_NOTICES.md`, model status, and CLI docs
  when the public contract or provenance changes.
- Do not weaken the pinned-checkpoint, loopback-bind, license-acknowledgement,
  evidence-redaction, or generated-results checks to make a test pass.

Use Conventional Commits:

```text
<type>(<scope>): <imperative subject>
```

Common types are `feat`, `fix`, `perf`, `refactor`, `build`, `ci`, `docs`,
`test`, and `chore`. The subject is lowercase and has no trailing period.

## Testing

Run the smallest relevant test first, then the non-slow suite:

```bash
python -m pytest tests/path/to/test_file.py
python -m pytest -m 'not slow'
```

QSA, PLE, model mapping, cache-state, and top-k changes require their CPU oracles
and tiny synthetic parity tests. GPU kernel changes require the relevant CUDA
tests on the hardware/shape they claim to support. A CPU-only CI success does
not establish RTX 5090 correctness.

For changes intended for the alpha release, also run:

```bash
q38lab smoke
q38lab bench --out results/local-review
```

Do not commit a local evidence directory until it has passed the repository's
redaction/validation tooling and the maintainer has approved it as a release
snapshot.

## Performance evidence

Performance pull requests need A/B results from the same commit base, checkpoint
revision, machine, prompt/token budgets, profile, warm-up count, and measurement
method. Include:

- GPU, VRAM, CPU, RAM, OS/WSL, Windows driver, CUDA toolkit, Torch, Triton, and
  power/clock policy;
- exact commands and resolved configuration;
- median and p95 TTFT/latency, steady 256–512-token decode throughput, RSS,
  VRAM, page faults, and PCIe samples;
- failures and variance, not only the best run;
- whether WSL swap was zero and the truthful Windows pagefile caveat.

Never call client-observed effective prefill a kernel benchmark, infer long-run
decode speed from a seven-token completion, or claim “no swap” when only WSL
swap was measured.

## Upstreaming

Downstream work may later be proposed to FreeToken, but this repository does not
automatically open or submit upstream pull requests. Coordinate in
[FreeToken issue #214](https://github.com/FlashML-org/FreeToken/issues/214),
split QSA, auxiliary-bank lifecycle, residuals, loader integration, and API work
into reviewable changes, and follow the upstream maintainers' current process.

The human author must understand, run, and present each upstream proposal.
Downstream branding, release tooling, RTX-specific profiles, and model-license
acknowledgement are generally not upstream engine changes.

## License

Unless explicitly stated otherwise, contributions intentionally submitted to
this repository are provided under the [Apache License 2.0](LICENSE). By
submitting a contribution, you confirm that you have the right to do so and
that required third-party notices are included.
