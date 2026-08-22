#!/usr/bin/env python3
"""LoRA baseline and DP backend comparison for BF16 or 4-bit VaultGemma."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from opacus import GradSampleModule
from opacus.grad_sample import (
    GradSampleModuleExpandedWeights,
    GradSampleModuleFastGradientClipping,
)
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.func import functional_call, grad_and_value, vmap
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from level1_patient_code_common import build_tokenized_dataset, load_manifest


METHODS = ("non_dp", "naive_dp", "hooks_dp", "vmap_dp", "expanded_weights_dp", "ghost_dp")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return value


def deep_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = data
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def deep_set(data: dict[str, Any], dotted: str, value: Any) -> None:
    current = data
    keys = dotted.split(".")
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_slug(epsilon: float | None) -> str:
    if epsilon is None:
        return "none"
    return f"eps{epsilon:g}".replace(".", "p")


def normalize_grad_sample(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(value, list):
        return torch.cat(value, dim=0)
    return value


def clipped_sum_from_grad_samples(
    grad_samples: Sequence[torch.Tensor], max_grad_norm: float
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Clip a batch of per-example gradients by one global norm per example."""
    if not grad_samples:
        raise ValueError("No per-example gradients were provided")
    batch_size = grad_samples[0].shape[0]
    squared_norm = torch.zeros(batch_size, device=grad_samples[0].device, dtype=torch.float32)
    for grad_sample in grad_samples:
        if grad_sample.shape[0] != batch_size:
            raise ValueError("Inconsistent grad_sample batch dimension")
        squared_norm.add_(grad_sample.detach().float().reshape(batch_size, -1).square().sum(dim=1))
    norms = squared_norm.sqrt()
    coefficients = (max_grad_norm / (norms + 1e-12)).clamp(max=1.0)
    clipped_sums: list[torch.Tensor] = []
    for grad_sample in grad_samples:
        view_shape = (batch_size,) + (1,) * (grad_sample.ndim - 1)
        clipped_sums.append(
            (grad_sample.detach() * coefficients.to(grad_sample.dtype).view(view_shape)).sum(dim=0)
        )
    return clipped_sums, norms


def response_losses(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-example mean loss, token-loss sums, and valid response-token counts."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    token_counts = valid.sum(dim=1)
    token_sums = (flat_loss * valid).sum(dim=1)
    return token_sums / token_counts.clamp(min=1), token_sums, token_counts


def stack_rows(dataset: Dataset, indices: Iterable[int], device: torch.device) -> dict[str, torch.Tensor]:
    rows = [dataset[int(index)] for index in indices]
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    return {
        key: torch.stack([row[key] for row in rows]).to(device)
        for key in ("input_ids", "attention_mask", "labels")
    }


def clear_grad_samples(parameters: Sequence[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.grad = None
        if hasattr(parameter, "grad_sample"):
            parameter.grad_sample = None
        if hasattr(parameter, "summed_grad"):
            parameter.summed_grad = None
        if hasattr(parameter, "_current_grad_sample"):
            del parameter._current_grad_sample


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    example_loss_sum = 0.0
    token_loss_sum = 0.0
    token_count = 0
    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        batch = stack_rows(dataset, indices, device)
        if isinstance(model, GradSampleModuleExpandedWeights):
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        else:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        example_losses, token_sums, token_counts = response_losses(outputs.logits, batch["labels"])
        example_loss_sum += float(example_losses.sum().cpu())
        token_loss_sum += float(token_sums.sum().cpu())
        token_count += int(token_counts.sum().cpu())
        del outputs, batch, example_losses, token_sums, token_counts
    model.train()
    example_mean = example_loss_sum / len(dataset)
    token_mean = token_loss_sum / token_count
    return {
        "samples": len(dataset),
        "response_tokens": token_count,
        "example_mean_loss": example_mean,
        "example_mean_ppl": math.exp(example_mean),
        "token_mean_loss": token_mean,
        "token_mean_ppl": math.exp(token_mean),
    }


def build_dataset(cfg: dict[str, Any], tokenizer: Any, log: Any) -> tuple[Dataset, Dataset, int]:
    dataset_name = str(deep_get(cfg, "dataset.name"))
    dataset_split = str(deep_get(cfg, "dataset.split", "train"))
    num_samples = int(deep_get(cfg, "dataset.num_samples", 8000))
    train_fraction = float(deep_get(cfg, "dataset.train_fraction", 0.9))
    max_length = int(deep_get(cfg, "dataset.max_length", 256))

    raw = load_dataset(dataset_name, split=dataset_split).select(range(num_samples))
    required = {"input", "output"}
    if not required.issubset(raw.column_names):
        raise ValueError(f"Dataset must contain {sorted(required)}; got {raw.column_names}")

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        prompts = [
            f"Instruction:\nAnswer this question truthfully.\n\nQuestion:\n{question}"
            for question in batch["input"]
        ]
        responses = [f"\n\nResponse:\n{answer}" for answer in batch["output"]]
        combined = tokenizer(
            [prompt + response for prompt, response in zip(prompts, responses)],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        prompt_tokens = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        labels = [list(ids) for ids in combined["input_ids"]]
        for row_index, row_labels in enumerate(labels):
            prompt_length = int(sum(prompt_tokens["attention_mask"][row_index]))
            for token_index in range(prompt_length):
                row_labels[token_index] = -100
            for token_index, attended in enumerate(combined["attention_mask"][row_index]):
                if not attended:
                    row_labels[token_index] = -100
        combined["labels"] = labels
        return combined

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    before_filter = len(tokenized)
    tokenized = tokenized.filter(lambda labels: any(value != -100 for value in labels), input_columns=["labels"])
    dropped = before_filter - len(tokenized)
    if len(tokenized) < 2:
        raise ValueError("Not enough examples remain after response-token validation")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_size = int(len(tokenized) * train_fraction)
    train_dataset = tokenized.select(range(train_size))
    eval_dataset = tokenized.select(range(train_size, len(tokenized)))
    synthetic_manifest_path = deep_get(cfg, "dataset.synthetic_manifest")
    synthetic_member_count = 0
    if synthetic_manifest_path:
        manifest = load_manifest(Path(str(synthetic_manifest_path)))
        member_records = [
            dict(row)
            for row in manifest["records"]
            if row["membership"] == "member"
        ]
        synthetic_dataset = build_tokenized_dataset(
            member_records, tokenizer, max_length
        )
        extra_columns = [
            column
            for column in ("patient_id", "private_code", "membership")
            if column in synthetic_dataset.column_names
        ]
        if extra_columns:
            synthetic_dataset = synthetic_dataset.remove_columns(extra_columns)
        synthetic_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        synthetic_member_count = len(synthetic_dataset)
        expected_count = int(
            deep_get(cfg, "dataset.synthetic_member_count", synthetic_member_count)
        )
        if synthetic_member_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} synthetic members, got {synthetic_member_count}"
            )
        train_dataset = concatenate_datasets([train_dataset, synthetic_dataset])
        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
    log(
        f"dataset={dataset_name} selected={before_filter} dropped_zero_response={dropped} "
        f"medalpaca_train={train_size} synthetic_member={synthetic_member_count} "
        f"train={len(train_dataset)} eval={len(eval_dataset)} max_length={max_length}"
    )
    return train_dataset, eval_dataset, dropped


def build_model(cfg: dict[str, Any], device: torch.device, log: Any) -> tuple[torch.nn.Module, Any, tuple[int, int]]:
    model_id = str(deep_get(cfg, "model.id"))
    load_in_4bit = as_bool(deep_get(cfg, "model.load_in_4bit", True))
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    log(f"loading_model={model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=str(deep_get(cfg, "model.attn_implementation", "eager")),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=as_bool(deep_get(cfg, "model.use_gradient_checkpointing", False)),
        )
    lora_config = LoraConfig(
        r=int(deep_get(cfg, "lora.r", 8)),
        lora_alpha=int(deep_get(cfg, "lora.alpha", 16)),
        lora_dropout=float(deep_get(cfg, "lora.dropout", 0.0)),
        bias=str(deep_get(cfg, "lora.bias", "none")),
        task_type=str(deep_get(cfg, "lora.task_type", "CAUSAL_LM")),
        target_modules=list(deep_get(cfg, "lora.target_modules")),
    )
    model = get_peft_model(model, lora_config)
    model.train()
    trainable, total = model.get_nb_trainable_parameters()
    log(f"trainable_parameters={trainable} total_parameters={total} ratio={trainable / total:.8f}")
    return model, tokenizer, (trainable, total)


def execute(args: argparse.Namespace, cfg: dict[str, Any], run_dir: Path, log: Any) -> dict[str, Any]:
    seed = int(deep_get(cfg, "runtime.seed", 42))
    set_reproducibility(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    log(f"method={args.method} gpu_visible={args.gpu} gpu_name={torch.cuda.get_device_name(0)} seed={seed}")

    model, tokenizer, parameter_counts = build_model(cfg, device, log)
    train_dataset, eval_dataset, dropped_examples = build_dataset(cfg, tokenizer, log)

    logical_batch = int(deep_get(cfg, "training.logical_batch_size", 128))
    physical_batch = int(deep_get(cfg, "training.physical_batch_size", 16))
    eval_batch = int(deep_get(cfg, "training.eval_batch_size", physical_batch))
    epochs = int(deep_get(cfg, "training.epochs", 6))
    learning_rate = float(deep_get(cfg, "training.learning_rate", 1e-4))
    weight_decay = float(deep_get(cfg, "training.weight_decay", 0.01))
    target_delta = float(deep_get(cfg, "training.target_delta", 1e-5))
    max_grad_norm = float(deep_get(cfg, "training.max_grad_norm", 1.0))
    accountant_name = str(deep_get(cfg, "training.accountant", "prv"))
    logging_steps = int(deep_get(cfg, "training.logging_steps", 10))

    steps_per_epoch = math.ceil(len(train_dataset) / logical_batch)
    planned_steps = steps_per_epoch * epochs
    steps_to_run = planned_steps if args.max_steps <= 0 else min(args.max_steps, planned_steps)
    sample_rate = logical_batch / len(train_dataset)
    dp_enabled = args.method != "non_dp"
    if dp_enabled and args.target_epsilon is None:
        raise ValueError("--target-epsilon is required for a DP method")
    if not dp_enabled and args.target_epsilon is not None:
        raise ValueError("non_dp must not receive --target-epsilon")

    noise_multiplier = None
    accountant = None
    if dp_enabled:
        noise_multiplier = float(
            get_noise_multiplier(
                target_epsilon=float(args.target_epsilon),
                target_delta=target_delta,
                sample_rate=sample_rate,
                steps=planned_steps,
                accountant=accountant_name,
            )
        )
        accountant = create_accountant(mechanism=accountant_name)

    if args.method == "non_dp":
        log("gradient_mode=standard_non_dp_lora")
    elif args.method == "naive_dp":
        physical_batch = 1
        log("gradient_mode=python_loop_one_example")
    elif args.method == "hooks_dp":
        model = GradSampleModule(
            model, batch_first=True, loss_reduction="mean", strict=False
        )
        log("gradient_mode=opacus_hooks")
    elif args.method == "expanded_weights_dp":
        model = GradSampleModuleExpandedWeights(
            model, batch_first=True, loss_reduction="mean"
        )
        log("gradient_mode=opacus_expanded_weights")
    elif args.method == "ghost_dp":
        model = GradSampleModuleFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction="mean",
            strict=False,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
        )
        log("gradient_mode=opacus_ghost_two_pass")
    else:
        log("gradient_mode=torch_func_direct_vmap_grad")

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_named_parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    sampling_rng = np.random.default_rng(seed + 20_000)

    deep_set(cfg, "resolved.method", args.method)
    deep_set(cfg, "resolved.target_epsilon", args.target_epsilon)
    deep_set(cfg, "resolved.noise_multiplier", noise_multiplier)
    deep_set(cfg, "resolved.sample_rate", sample_rate)
    deep_set(cfg, "resolved.steps_per_epoch", steps_per_epoch)
    deep_set(cfg, "resolved.planned_steps", planned_steps)
    deep_set(cfg, "resolved.steps_to_run", steps_to_run)
    deep_set(cfg, "resolved.experiment_date", args.experiment_date)
    deep_set(cfg, "resolved.gpu", args.gpu)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    log(
        f"logical_batch={logical_batch} physical_batch={physical_batch} sample_rate={sample_rate:.8f} "
        f"steps_per_epoch={steps_per_epoch} planned_steps={planned_steps} steps_to_run={steps_to_run}"
    )
    log(
        f"target_epsilon={args.target_epsilon} target_delta={target_delta} "
        f"noise_multiplier={noise_multiplier} max_grad_norm={max_grad_norm}"
    )

    losses: list[float] = []
    step_times: list[float] = []
    lot_sizes: list[int] = []
    epsilon_trace: list[dict[str, float | int]] = []
    total_examples = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    for step_index in range(steps_to_run):
        step_started = time.perf_counter()
        lot_indices = np.flatnonzero(sampling_rng.random(len(train_dataset)) < sample_rate)
        if len(lot_indices) == 0:
            raise RuntimeError("Poisson sampler produced an empty lot")
        optimizer.zero_grad(set_to_none=True)
        lot_loss_sum = 0.0

        if args.method == "non_dp":
            for start in range(0, len(lot_indices), physical_batch):
                chunk_indices = lot_indices[start : start + physical_batch]
                batch = stack_rows(train_dataset, chunk_indices, device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                per_example_loss, _, _ = response_losses(outputs.logits, batch["labels"])
                lot_loss_sum += float(per_example_loss.detach().sum().cpu())
                weighted_loss = per_example_loss.sum() / len(lot_indices)
                weighted_loss.backward()
                del batch, outputs, per_example_loss, weighted_loss

        elif args.method == "naive_dp":
            gradient_buffer = [torch.zeros_like(parameter) for parameter in trainable_parameters]
            for index in lot_indices:
                batch = stack_rows(train_dataset, [int(index)], device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                per_example_loss, _, _ = response_losses(outputs.logits, batch["labels"])
                loss = per_example_loss[0]
                lot_loss_sum += float(loss.detach().cpu())
                loss.backward()
                with torch.no_grad():
                    squared_norm = torch.zeros((), device=device, dtype=torch.float32)
                    for parameter in trainable_parameters:
                        if parameter.grad is not None:
                            squared_norm.add_(parameter.grad.detach().float().square().sum())
                    coefficient = (max_grad_norm / (squared_norm.sqrt() + 1e-12)).clamp(max=1.0)
                    for buffer, parameter in zip(gradient_buffer, trainable_parameters):
                        if parameter.grad is not None:
                            buffer.add_(parameter.grad.detach() * coefficient.to(parameter.grad.dtype))
                optimizer.zero_grad(set_to_none=True)
                del batch, outputs, per_example_loss, loss

        elif args.method == "vmap_dp":
            gradient_buffer = [torch.zeros_like(parameter) for parameter in trainable_parameters]
            def single_loss(
                params: dict[str, torch.Tensor],
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor,
            ) -> torch.Tensor:
                outputs = functional_call(
                    model,
                    params,
                    args=(input_ids.unsqueeze(0),),
                    kwargs={
                        "attention_mask": attention_mask.unsqueeze(0),
                        "use_cache": False,
                    },
                    strict=False,
                )
                per_example_loss, _, _ = response_losses(
                    outputs.logits, labels.unsqueeze(0)
                )
                return per_example_loss[0]

            per_sample_grad_and_loss_fn = vmap(
                grad_and_value(single_loss),
                in_dims=(None, 0, 0, 0),
                randomness="different",
            )
            for start in range(0, len(lot_indices), physical_batch):
                chunk_indices = lot_indices[start : start + physical_batch]
                batch = stack_rows(train_dataset, chunk_indices, device)
                per_sample, per_example_loss = per_sample_grad_and_loss_fn(
                    trainable_named_parameters,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )
                lot_loss_sum += float(per_example_loss.detach().sum().cpu())
                grad_samples = [per_sample[name] for name in trainable_named_parameters]
                clipped_sums, _ = clipped_sum_from_grad_samples(grad_samples, max_grad_norm)
                with torch.no_grad():
                    for buffer, clipped_sum in zip(gradient_buffer, clipped_sums):
                        buffer.add_(clipped_sum)
                del batch, per_example_loss, per_sample, grad_samples, clipped_sums

        elif args.method in {"hooks_dp", "expanded_weights_dp"}:
            gradient_buffer = [torch.zeros_like(parameter) for parameter in trainable_parameters]
            for start in range(0, len(lot_indices), physical_batch):
                chunk_indices = lot_indices[start : start + physical_batch]
                batch = stack_rows(train_dataset, chunk_indices, device)
                if args.method == "expanded_weights_dp":
                    outputs = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )
                else:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )
                per_example_loss, _, _ = response_losses(outputs.logits, batch["labels"])
                loss = per_example_loss.mean()
                lot_loss_sum += float(per_example_loss.detach().sum().cpu())
                loss.backward()
                grad_samples: list[torch.Tensor] = []
                for parameter in trainable_parameters:
                    value = getattr(parameter, "grad_sample", None)
                    if value is None:
                        raise RuntimeError("Opacus did not produce grad_sample for a trainable parameter")
                    grad_samples.append(normalize_grad_sample(value))
                clipped_sums, _ = clipped_sum_from_grad_samples(grad_samples, max_grad_norm)
                with torch.no_grad():
                    for buffer, clipped_sum in zip(gradient_buffer, clipped_sums):
                        buffer.add_(clipped_sum)
                clear_grad_samples(trainable_parameters)
                del batch, outputs, per_example_loss, loss, grad_samples, clipped_sums

        else:  # ghost_dp
            gradient_buffer = [torch.zeros_like(parameter) for parameter in trainable_parameters]
            for start in range(0, len(lot_indices), physical_batch):
                chunk_indices = lot_indices[start : start + physical_batch]
                batch = stack_rows(train_dataset, chunk_indices, device)
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                per_example_loss, _, _ = response_losses(outputs.logits, batch["labels"])
                lot_loss_sum += float(per_example_loss.detach().sum().cpu())
                norm_loss = per_example_loss.mean()
                norm_loss.backward(retain_graph=True)
                coefficients = model.get_clipping_coef().detach()
                model.zero_grad(set_to_none=True)

                model.disable_hooks()
                try:
                    clipped_loss = (coefficients * per_example_loss).sum()
                    clipped_loss.backward()
                finally:
                    model.enable_hooks()

                with torch.no_grad():
                    for buffer, parameter in zip(gradient_buffer, trainable_parameters):
                        if parameter.grad is not None:
                            buffer.add_(parameter.grad.detach())
                model.zero_grad(set_to_none=True)
                del (
                    batch,
                    outputs,
                    per_example_loss,
                    norm_loss,
                    coefficients,
                    clipped_loss,
                )

        if dp_enabled:
            with torch.no_grad():
                for buffer, parameter in zip(gradient_buffer, trainable_parameters):
                    noise = torch.randn(
                        buffer.shape,
                        generator=noise_generator,
                        device=buffer.device,
                        dtype=buffer.dtype,
                    )
                    buffer.add_(noise, alpha=float(noise_multiplier) * max_grad_norm)
                    parameter.grad = buffer.div(logical_batch)
            accountant.step(noise_multiplier=float(noise_multiplier), sample_rate=sample_rate)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if args.method in {"hooks_dp", "expanded_weights_dp"}:
            clear_grad_samples(trainable_parameters)

        torch.cuda.synchronize()
        elapsed_step = time.perf_counter() - step_started
        average_lot_loss = lot_loss_sum / len(lot_indices)
        losses.append(average_lot_loss)
        step_times.append(elapsed_step)
        lot_sizes.append(len(lot_indices))
        total_examples += len(lot_indices)

        step_number = step_index + 1
        if step_number == 1 or step_number % logging_steps == 0 or step_number == steps_to_run:
            epsilon_now = None
            if accountant is not None:
                epsilon_now = float(accountant.get_epsilon(delta=target_delta))
                epsilon_trace.append({"step": step_number, "epsilon": epsilon_now})
            log(
                f"step={step_number}/{steps_to_run} epoch={(step_index // steps_per_epoch) + 1}/{epochs} "
                f"lot_size={len(lot_indices)} loss={average_lot_loss:.6f} "
                f"step_time={elapsed_step:.3f}s epsilon={epsilon_now}"
            )

    torch.cuda.synchronize()
    elapsed_training = time.perf_counter() - started
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    final_epsilon = float(accountant.get_epsilon(delta=target_delta)) if accountant is not None else None

    log("running_final_response_only_eval")
    eval_result = evaluate(model, eval_dataset, eval_batch, device)
    log(f"eval={json.dumps(eval_result, ensure_ascii=False)}")

    raw_model = model._module if hasattr(model, "_module") else model
    adapter_path = run_dir / "final_adapter"
    raw_model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    summary = {
        "status": "completed",
        "run_type": "full" if steps_to_run == planned_steps else "smoke",
        "method": args.method,
        "experiment_date": args.experiment_date,
        "run_dir": str(run_dir),
        "model": str(deep_get(cfg, "model.id")),
        "load_in_4bit": as_bool(deep_get(cfg, "model.load_in_4bit", False)),
        "attn_implementation": str(deep_get(cfg, "model.attn_implementation", "eager")),
        "dataset": str(deep_get(cfg, "dataset.name")),
        "synthetic_manifest": deep_get(cfg, "dataset.synthetic_manifest"),
        "synthetic_member_samples": int(
            deep_get(cfg, "dataset.synthetic_member_count", 0)
        ),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "dropped_zero_response_examples": dropped_examples,
        "trainable_parameters": parameter_counts[0],
        "total_parameters": parameter_counts[1],
        "seed": seed,
        "logical_batch_size": logical_batch,
        "physical_batch_size": physical_batch,
        "sample_rate": sample_rate,
        "expected_sample_passes": steps_to_run * sample_rate,
        "sampling_seed": seed + 20_000,
        "noise_seed": seed + 10_000 if dp_enabled else None,
        "steps_per_epoch": steps_per_epoch,
        "planned_steps": planned_steps,
        "completed_steps": steps_to_run,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "target_epsilon": args.target_epsilon,
        "target_delta": target_delta if dp_enabled else None,
        "accountant": accountant_name if dp_enabled else None,
        "max_grad_norm": max_grad_norm if dp_enabled else None,
        "noise_multiplier": noise_multiplier,
        "final_epsilon": final_epsilon,
        "epsilon_trace": epsilon_trace,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "loss_trace": losses,
        "lot_size_mean": sum(lot_sizes) / len(lot_sizes),
        "total_examples_processed": total_examples,
        "elapsed_training_sec": elapsed_training,
        "mean_step_time_sec": sum(step_times) / len(step_times),
        "throughput_samples_per_sec": total_examples / sum(step_times),
        "peak_vram_gb": peak_vram_gb,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_visible": args.gpu,
        "eval": eval_result,
        "adapter_path": str(adapter_path),
        "training_log_path": str(run_dir / "training.log"),
        "resolved_config_path": str(run_dir / "resolved_config.yaml"),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(
        f"completed final_epsilon={final_epsilon} elapsed_training_sec={elapsed_training:.3f} "
        f"throughput={summary['throughput_samples_per_sec']:.4f} peak_vram_gb={peak_vram_gb:.4f}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--target-epsilon", type=float)
    parser.add_argument("--experiment-date", default=date.today().isoformat())
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_gpu != str(args.gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be set before Python starts; "
            f"expected {args.gpu!r}, got {visible_gpu!r}. Use scripts/run_one.sh."
        )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_yaml(args.config)
    os.environ.setdefault("HF_HOME", str(deep_get(cfg, "paths.hf_home")))
    os.environ.setdefault("HF_HUB_CACHE", str(deep_get(cfg, "paths.hf_hub_cache")))

    output_root = Path(str(deep_get(cfg, "paths.output_root")))
    method_root = output_root / args.experiment_date / "runs" / args.method / epsilon_slug(args.target_epsilon)
    run_dir = method_root / f"{time.strftime('%Y%m%d_%H%M%S')}_seed{deep_get(cfg, 'runtime.seed', 42)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    training_log = run_dir / "training.log"

    def log(message: Any) -> None:
        line = str(message)
        print(line, flush=True)
        with training_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "running", "method": args.method}, indent=2), encoding="utf-8"
    )
    try:
        summary = execute(args, cfg, run_dir, log)
        (run_dir / "run_status.json").write_text(
            json.dumps({"status": "completed", "summary": str(run_dir / 'run_summary.json')}, indent=2),
            encoding="utf-8",
        )
        log(f"summary_path={run_dir / 'run_summary.json'}")
        return 0
    except Exception as exc:
        error = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        (run_dir / "run_status.json").write_text(json.dumps(error, indent=2, ensure_ascii=False), encoding="utf-8")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
