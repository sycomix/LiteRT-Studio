# Artifact packaging

`package-artifact` creates a deterministic `.litertstudio` ZIP archive from an
artifact directory containing `manifest.json`. It verifies every artifact
against the size and SHA-256 recorded in that manifest before packaging.

```bash
litert-studio package-artifact artifacts/export \
  --report runs/reference-suite.json \
  --report runs/litert-runtime-report.json \
  --output releases/model.litertstudio

litert-studio verify-bundle releases/model.litertstudio
```

The archive contains:

- `bundle.json`, which hashes every packaged entry;
- the original `manifest.json`;
- files explicitly listed by the artifact manifest under `artifacts/`;
- reports explicitly supplied on the command line under `reports/`.

Entries are sorted and assigned fixed timestamps and permissions. Identical
inputs therefore produce byte-identical archives. Training datasets, caches,
checkpoints, logs, and secrets are not discovered or included automatically.
The verifier checks the bundle format, rejects duplicate or unmanifested
entries, and validates every recorded size and hash.
