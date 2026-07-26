from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.core.manifest import ArtifactManifest, describe_artifact
from litert_studio.training.datasets import inspect_jsonl
from litert_studio.training.formatting import (
    TokenizerLike,
    load_formatted_records,
    split_formatted_records,
    token_statistics,
)
from litert_studio.training.recipe import TrainingRecipe


@dataclass(frozen=True)
class TrainingResult:
    adapter_directory: Path
    manifest: Path
    metrics: dict[str, Any]
    reused: bool = False


def run_transformers_peft(recipe: TrainingRecipe) -> TrainingResult:
    """Run the optional heavyweight backend.

    Imports stay inside this function so planning and preflight remain lightweight.
    """
    # This worker is intentionally PyTorch-only. Prevent an unrelated installed
    # TensorFlow/Keras stack from being imported by optional Transformers integrations.
    os.environ.setdefault("USE_TF", "0")

    existing = _existing_result(recipe)
    if existing is not None:
        return existing

    import torch
    from peft import (  # type: ignore[import-not-found]
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    output = Path(recipe.output)
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "metrics.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(recipe.base_model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_like = cast(TokenizerLike, tokenizer)
    texts = load_formatted_records(Path(recipe.dataset), tokenizer_like)
    evaluation_texts: list[str] = []
    split_provenance: dict[str, Any]
    if recipe.eval_dataset is not None:
        evaluation_texts = load_formatted_records(Path(recipe.eval_dataset), tokenizer_like)
        training_texts = texts
        split_provenance = {
            "strategy": "holdout_dataset",
            "train_records": len(training_texts),
            "evaluation_records": len(evaluation_texts),
        }
    elif recipe.validation_split is not None:
        split = split_formatted_records(texts, recipe.validation_split, recipe.seed)
        training_texts = list(split.train)
        evaluation_texts = list(split.evaluation)
        split_provenance = {"strategy": "deterministic_split", **split.to_dict()}
    else:
        training_texts = texts
        split_provenance = {
            "strategy": "none",
            "train_records": len(training_texts),
            "evaluation_records": 0,
        }
    train_stats = token_statistics(training_texts, tokenizer_like, recipe.max_sequence_length)
    evaluation_stats = (
        token_statistics(evaluation_texts, tokenizer_like, recipe.max_sequence_length)
        if evaluation_texts
        else None
    )

    class TokenizedDataset(Dataset):  # type: ignore[misc]
        def __init__(self, values: list[str]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return tokenizer(
                self.values[index],
                truncation=True,
                max_length=recipe.max_sequence_length,
                add_special_tokens=True,
            )

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[recipe.precision]
    model_options: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": False,
    }
    if recipe.method == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("The initial QLoRA backend requires CUDA")
        model_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_options["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(recipe.base_model, **model_options)
    if recipe.method == "qlora":
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if recipe.method in {"lora", "qlora"}:
        targets = (
            "all-linear"
            if recipe.lora.target_modules == "auto"
            else list(recipe.lora.target_modules)
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=recipe.lora.rank,
                lora_alpha=recipe.lora.alpha,
                lora_dropout=recipe.lora.dropout,
                target_modules=targets,
            ),
        )

    class JsonMetricsCallback(TrainerCallback):  # type: ignore[misc]
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
            event = {"step": state.global_step, "metrics": logs or {}}
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    arguments = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=recipe.epochs,
        max_steps=recipe.max_steps,
        per_device_train_batch_size=recipe.batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        learning_rate=recipe.learning_rate,
        logging_steps=recipe.logging_steps,
        logging_first_step=True,
        save_steps=recipe.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        seed=recipe.seed,
        data_seed=recipe.seed,
        bf16=recipe.precision == "bf16",
        fp16=recipe.precision == "fp16",
        use_cpu=not torch.cuda.is_available(),
        remove_unused_columns=True,
        eval_strategy="epoch" if evaluation_texts else "no",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=TokenizedDataset(training_texts),
        eval_dataset=TokenizedDataset(evaluation_texts) if evaluation_texts else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
        callbacks=[JsonMetricsCallback()],
    )
    checkpoint_dir = output / "checkpoints"
    checkpoint = (
        get_last_checkpoint(str(checkpoint_dir))
        if recipe.resume_from_checkpoint and checkpoint_dir.is_dir()
        else None
    )
    train_output = trainer.train(resume_from_checkpoint=checkpoint)
    evaluation_metrics = trainer.evaluate() if evaluation_texts else {}
    adapter_dir = output / ("model" if recipe.method == "full" else "adapter")
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    files = tuple(
        describe_artifact(path, relative_to=output)
        for path in sorted(adapter_dir.iterdir())
        if path.is_file()
    )
    manifest = ArtifactManifest(
        artifact_type="full-model" if recipe.method == "full" else "peft-adapter",
        source_fingerprints=_source_fingerprints(recipe),
        files=files,
        tools={
            "torch": version("torch"),
            "transformers": version("transformers"),
            "peft": version("peft"),
            "safetensors": version("safetensors"),
        },
        settings=recipe.to_dict(),
        validation={
            "training_metrics": train_output.metrics,
            "evaluation_metrics": evaluation_metrics,
            "dataset_split": split_provenance,
            "train_token_statistics": train_stats.to_dict(),
            "evaluation_token_statistics": (
                evaluation_stats.to_dict() if evaluation_stats is not None else None
            ),
        },
    )
    manifest_path = manifest.write(output / "manifest.json", overwrite=True)
    return TrainingResult(
        adapter_directory=adapter_dir,
        manifest=manifest_path,
        metrics=dict(train_output.metrics),
    )


def _existing_result(recipe: TrainingRecipe) -> TrainingResult | None:
    output = Path(recipe.output)
    manifest_path = output / "manifest.json"
    adapter_dir = output / ("model" if recipe.method == "full" else "adapter")
    weights = (
        adapter_dir / "model.safetensors"
        if recipe.method == "full"
        else adapter_dir / "adapter_model.safetensors"
    )
    if not manifest_path.is_file() or not weights.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprints = manifest["source_fingerprints"]
        validation = manifest["validation"]
        settings = manifest["settings"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    expected = _source_fingerprints(recipe)
    if fingerprints != expected or settings.get("request_id") != recipe.request_id:
        return None
    metrics = validation.get("training_metrics")
    if not isinstance(metrics, dict):
        return None
    return TrainingResult(
        adapter_directory=adapter_dir,
        manifest=manifest_path,
        metrics=metrics,
        reused=True,
    )


def _source_fingerprints(recipe: TrainingRecipe) -> dict[str, str]:
    fingerprints = {
        "training_request": recipe.request_id,
        "base_model": inspect_model_directory(Path(recipe.base_model)).fingerprint,
        "dataset": inspect_jsonl(Path(recipe.dataset)).fingerprint,
    }
    if recipe.eval_dataset is not None:
        fingerprints["eval_dataset"] = inspect_jsonl(Path(recipe.eval_dataset)).fingerprint
    return fingerprints
