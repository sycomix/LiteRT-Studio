# Initial implementation plan

## Workstream A — application core

Completed: versioned configuration schemas, SQLite jobs/events, atomic state
transitions, a shell-free subprocess launcher, environment allowlisting, and
artifact manifest contracts.

Remaining:

1. Add log redaction and environment-version capture.
2. Expose repositories and events through the loopback API.
3. Generate TypeScript types from the published schemas.

Acceptance: CLI and API produce byte-equivalent normalized plans and a stopped
process cannot report success.

## Workstream B — reference training backend

Completed: deterministic request materialization, JSONL validation/fingerprints,
dependency preflight, and the isolated worker boundary.

Remaining:

1. Select a small model/dataset fixture with CI-compatible licensing.
2. Implement formatting and token-stat previews.
3. Implement the pinned Transformers + PEFT LoRA training loop.
4. Resume from checkpoints and emit structured metric events.
5. Save adapter and merged SafeTensors variants with manifests.

Acceptance: a tiny LoRA run completes in CI, resumes, and produces a loadable
adapter with deterministic smoke-test output.

## Workstream C — first architecture adapter

Completed: Gemma selection, official `export_hf` integration research,
SafeTensors header/schema audit, PEFT merge, versioned export requests, Linux
preflight, and PyTorch reference-suite capture.

Remaining:

1. Execute the float `.litertlm` baseline in a pinned Linux environment.
2. Add a LiteRT-LM runner that emits candidate logits and tokens.
3. Compare source/exported results through the parity schema.
4. Add dynamic int8 only after float baseline parity.

Acceptance: pinned prompts meet documented tolerances and generate matching
greedy tokens before packaging.

## Workstream D — local Studio

1. Add loopback-only API bootstrap and filesystem authorization.
2. Implement project/model/data pages.
3. Add plan editors, compatibility explanations, and estimates.
4. Stream job state, metrics, and logs; support cancellation.
5. Add artifact inspection and export.

Acceptance: a user can perform the reference workflow without editing JSON,
while exporting a CLI-replayable plan.

## Suggested first issues

- Define `ArchitectureAdapter` protocol and capability descriptor.
- Add full-content optional hashing with progress events.
- Publish JSON Schema for conversion/training configurations.
- Add dataset fingerprinting and redaction tests.
- Research and record the reference model’s officially supported LiteRT path.
- Build the first model inspector UI from fixture data.
