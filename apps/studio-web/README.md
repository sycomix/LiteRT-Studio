# Studio web client

The first local interface is shipped with the Python service under
`src/litert_studio/server/static`. It is dependency-light, responsive, and
wired directly to the versioned local API.

The implemented workspaces include:

- model registry import and SafeTensors compatibility inspection;
- dataset validation and fingerprinting;
- LoRA, QLoRA, and full-training planning/execution;
- LiteRT conversion planning/execution and adapter merge;
- reference capture and LiteRT-LM runtime validation;
- deterministic packaging and bundle verification;
- durable background job history and events.

The client remains a presentation layer. Validation rules and pipeline state
transitions belong to the Python application service so CLI and UI runs behave
identically. A future TypeScript client can replace this presentation layer
without changing the API.
