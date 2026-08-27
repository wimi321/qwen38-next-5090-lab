# Release evidence

`results/rtx5090-YYYY-MM-DD[-N]/` directories are sanitized, reviewable
hardware-validation records. They contain measurements, never prompts, generated
text, usernames, hostnames, local paths, credentials, or model weights.

Create a candidate bundle on the private RTX 5090 host:

```bash
q38lab bench \
  --model-dir "$HOME/models/qwen38-flash-next-nvfp4-7b71922" \
  --out "results/rtx5090-$(date -u +%F)"
```

`q38lab bench` reads the PID, exact runtime commit, model path, resolved
profile, and process start time from the live `q38lab serve` attestation. It
refuses a manually guessed PID or a modified host/port/profile.

Then review every file, set `source.release_compatible` only after confirming
that the measured runtime tree is represented by the release, and run:

```bash
python scripts/release/evidence.py checksums results/rtx5090-YYYY-MM-DD
python scripts/release/evidence.py validate results/rtx5090-YYYY-MM-DD --release
python scripts/release/evidence.py table results/rtx5090-YYYY-MM-DD --readme README.md
```

The GitHub tag workflow selects the newest directory that passes the strict
release validator. `example-synthetic/` demonstrates the wire format only; its
`status` makes it permanently ineligible for a release.

Schemas in `schema/` document the public JSON interface. The standard-library
validator is authoritative for cross-file, checksum, privacy, and threshold
rules that JSON Schema cannot express.
