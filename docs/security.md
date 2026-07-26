# Security and privacy

- SafeTensors is required for weight import; pickle-based checkpoints are not
  loaded by default.
- Treat model repositories, tokenizer templates, and datasets as untrusted
  input. Do not execute repository code unless the user opts into a sandboxed
  runner.
- Bind the future local API to loopback, require an unpredictable session
  secret, and apply origin checks.
- Resolve and authorize filesystem paths on the server. The web client receives
  opaque project asset IDs.
- Never place tokens in plans, manifests, logs, subprocess arguments, or UI
  state. Use environment-backed secret references.
- Network access is off by default for local jobs and explicit for registry
  downloads or remote runners.
- Artifact packages include hashes and provenance but never embed training
  records.
- Release bundles include only files named by the artifact manifest plus
  validation reports explicitly selected by the operator. Absolute paths and
  path traversal are rejected, and verification rejects unmanifested content.
