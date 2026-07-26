# Local Studio interface

The Studio interface exposes the same application core as the CLI:

- pinned Hugging Face model import and local SafeTensors inspection;
- JSONL dataset validation and fingerprinting;
- LoRA, QLoRA, and full fine-tuning plans and background execution;
- float and quantized LiteRT conversion plans and background execution;
- PEFT adapter merge, reference capture, and LiteRT-LM token validation;
- deterministic release packaging and bundle verification;
- durable SQLite job history and event details.
- cancellable subprocess execution with persistent worker logs.

Install the local interface and launch it from the project workspace:

```bash
python -m pip install -e ".[api,training,conversion,runtime]"
litert-studio serve --workspace .
```

Open `http://127.0.0.1:7860`. The server refuses non-loopback binding and API
paths outside the selected workspace. Run the interface inside the Ubuntu
24.04 WSL environment when launching LiteRT conversion or LiteRT-LM validation;
those upstream components require Linux. Windows browsers can open the WSL
loopback address normally.

On Windows, the verified launcher starts the complete stack inside WSL:

```powershell
.\scripts\start-studio-wsl.ps1
```

Registry imports require an explicit repository and revision. Studio resolves
the revision to a commit hash, downloads SafeTensors and tokenizer/configuration
assets only, and stores the snapshot under `models/`.
Model inspection reports parameter count, checkpoint size, and conservative
memory guidance for LoRA, QLoRA, and full bf16 training without loading model
weights.

Training and conversion workers run outside the web-service process with an
allowlisted environment and no shell. The Jobs workspace exposes state,
events, a live log tail, and cancellation. Cancellation first requests normal
process termination and escalates only if the worker does not exit.

## Current model families

The pinned LiteRT Torch 0.9.1 exporter contains extensions for Gemma,
Gemma 2/3/4, Qwen3, and LFM2 model types. Gemma and Gemma 2 pass the reviewed
tensor-schema audit. Newer upstream extensions are marked with the
`upstream_extension` validation level and remain research-grade until their
export passes reference parity and a target-device smoke test.

The pinned `llamafactory/tiny-random-qwen3` checkpoint was imported and
exported successfully in both float and dynamic-int8 forms. Both packages
loaded in LiteRT-LM and matched one of two deterministic prompts, but the
second prompt diverged from the PyTorch greedy token at the first decode step.
The source is random and its leading logits are nearly tied; it is retained as
a negative parity fixture. Studio therefore leaves Qwen3 at research level and
marks these artifacts as failed rather than promoting them.
