# Roadmap

## M0 — Foundation (complete)

- Project/config schema and deterministic plans
- SafeTensors layout inspection
- CLI dry runs and test fixtures
- Architecture, UI, security, and contribution contracts
- Versioned schemas, capability protocols, SQLite jobs, and runner lifecycle

## M1 — First end-to-end model (in progress)

- Gemma selected as the first research adapter family
- Isolated worker request, preflight, process launcher, parity reports, and
  artifact manifests complete
- Fixture-scale Transformers/PEFT LoRA run verified with checkpoints and metrics
- Gemma/Gemma 2 SafeTensors schema audit and adapter merge implemented
- Versioned LiteRT Torch export worker and WSL execution verified
- Float and dynamic-int8 `.litertlm` export complete
- LiteRT-LM CPU token parity complete for float and dynamic int8
- Deterministic evaluation splitting and explicit holdout datasets complete
- Expose and compare raw runtime logits; greedy decoding parity is complete
- Deterministic, checksummed release artifact packaging complete

## M2 — Local Studio

- FastAPI application services and complete local feature API
- Responsive model/data/train/convert/validate/artifact interface
- Persistent jobs and event history
- Cancellation and persistent process-level logs
- GPU/CPU capability detection and model memory estimates

## M3 — Deployment quality

- [x] Reproducible LiteRT-LM CPU benchmark reports
- [x] Android/ADB device readiness discovery
- [x] Persistent validation and benchmark compatibility history
- [x] Deployment controls and result history in the local GUI
- [ ] On-device Android execution and profiling runner

- Pinned standard-exporter policy registry and int8/int4 parity matrix complete
- Experimental calibrated static-int8 workflow
- Android reference application
- KV-cache/prefill/decode signature optimization
- Compatibility regression dashboard

## M4 — Ecosystem

- Additional architecture adapters
- Remote runner protocol
- Recipe and adapter plug-in SDK
- Signed artifact manifests and supply-chain policy

## Definition of “supported”

A model/quantization/device combination is supported only when CI verifies
conversion, numerical tolerances, deterministic generation cases, artifact
loading, and a real target-device smoke test.
