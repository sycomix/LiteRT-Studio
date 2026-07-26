# Deployment benchmarking

LiteRT Studio records deployment evidence separately from conversion success. A model
can export correctly yet still fail token parity or miss a device performance target.

## Local benchmark

The benchmark uses LiteRT-LM's CPU backend and deterministic greedy sampling. It records
the model checksum, quantization policy, runtime version, host, model-load time, per-run
prefill and decode latency, and median decode throughput.

```bash
litert-studio benchmark-litert \
  --model artifacts/model.litertlm \
  --prompts examples/reference-prompts.json \
  --output reports/benchmark.json
```

Warmup runs are excluded from the measurements. Keep the prompt suite, output-token
limit, runtime version, and power state fixed when comparing exports.

## Android readiness

`litert-studio android-devices` reports whether Android platform-tools are installed and
lists connected, offline, and unauthorized devices. The Studio Deploy page exposes the
same discovery information.

Device discovery is a readiness check; on-device execution and profiling remain a later
milestone. It does not copy models or run commands on a connected device.

## Compatibility history

Benchmarks launched through the GUI and LiteRT token-parity checks are stored in
`.litert-studio/compatibility.sqlite3`. The Deploy page shows recent results with their
runtime, quantization policy, timestamp, and pass/fail state. JSON reports remain the
portable source of detailed measurements.
