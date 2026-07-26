# Execution model

## Training boundary

The application service never imports the training stack to construct a plan.
It materializes a versioned request containing canonical local paths and
numerical recipe settings. An isolated worker then:

1. parses the request;
2. validates model and dataset paths;
3. fingerprints the complete JSONL dataset;
4. reports installed package versions;
5. refuses unsupported methods or overwrite-shaped paths;
6. exits before loading weights when `--preflight` is used.

The model-loading training loop remains disabled until the fixture-scale
Transformers/PEFT implementation and checkpoint tests are present.

## Process safety

Workers launch from argument arrays with `shell=False`. Standard input is
disabled, output is redirected to a project log, and only an explicit
environment allowlist crosses the process boundary. Tokens must be resolved by
a future secret broker rather than placed in requests, arguments, or logs.

Cancellation first requests termination, waits for a bounded interval, and
then kills the worker if necessary. The durable job state machine records
queued, running, succeeded, failed, and cancelled outcomes separately.

## Conversion validation

An adapter cannot move from `research` to `supported` based on successful file
creation alone. It must emit:

- numerical parity over pinned logits with absolute and relative tolerances;
- exact or explicitly approved token-generation parity;
- artifact hashes and byte sizes;
- tool versions, settings, source fingerprints, and license references.

The current parity module operates on framework-neutral sequences so both the
PyTorch reference worker and LiteRT-LM validation runner can produce the same
report schema.
