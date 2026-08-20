from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset, load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def instruction_prompt(question: str) -> str:
    return f"Instruction:\nAnswer this question truthfully.\n\nQuestion:\n{question}"


def response_text(answer: str) -> str:
    return f"\n\nResponse:\n{answer}"


def load_selected_raw(
    dataset_name: str,
    num_samples: int,
    selection: str,
    seed: int,
) -> tuple[Dataset, dict[str, Any]]:
    raw = load_dataset(dataset_name, split="train")
    if len(raw) < num_samples:
        raise ValueError(f"Dataset has {len(raw)} rows, expected at least {num_samples}")
    indexed = raw.add_column("_source_index", list(range(len(raw))))
    if selection == "head":
        selected = indexed.select(range(num_samples))
    elif selection == "shuffled":
        selected = indexed.shuffle(seed=seed).select(range(num_samples))
    else:
        raise ValueError(f"Unsupported selection: {selection}")
    source_indices = [int(value) for value in selected["_source_index"]]
    metadata = {
        "dataset_name": dataset_name,
        "source_size": len(raw),
        "source_fingerprint": raw._fingerprint,
        "selection": selection,
        "selection_seed": seed,
        "num_samples": num_samples,
        "source_indices_sha256": canonical_json_sha256(source_indices),
    }
    return selected, metadata


def apply_member_canaries(selected: Dataset, manifest: dict[str, Any]) -> Dataset:
    inputs = list(selected["input"])
    outputs = list(selected["output"])
    source_indices = [int(value) for value in selected["_source_index"]]
    seen_positions: set[int] = set()
    for canary in manifest["members"]:
        position = int(canary["train_position"])
        if not 0 <= position < int(manifest["train_size"]):
            raise ValueError(f"Canary position is outside train split: {position}")
        if position in seen_positions:
            raise ValueError(f"Duplicate canary train position: {position}")
        if source_indices[position] != int(canary["replaced_source_index"]):
            raise ValueError(f"Source index mismatch at canary position {position}")
        seen_positions.add(position)
        inputs[position] = str(canary["input"])
        outputs[position] = str(canary["output"])
        source_indices[position] = -1 - int(canary["canary_index"])
    return Dataset.from_dict(
        {"input": inputs, "output": outputs, "_source_index": source_indices}
    )


def tokenize_qa_dataset(
    selected: Dataset,
    tokenizer: Any,
    max_length: int,
) -> tuple[Dataset, int]:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        prompts = [instruction_prompt(question) for question in batch["input"]]
        responses = [response_text(answer) for answer in batch["output"]]
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

    tokenized = selected.map(tokenize, batched=True, remove_columns=selected.column_names)
    before = len(tokenized)
    tokenized = tokenized.filter(
        lambda labels: any(value != -100 for value in labels),
        input_columns=["labels"],
    )
    dropped = before - len(tokenized)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized, dropped


def response_losses(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.float().reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(shift_labels.shape)
    valid = shift_labels.ne(-100)
    token_counts = valid.sum(dim=1)
    token_sums = (flat_loss * valid).sum(dim=1)
    return token_sums / token_counts.clamp(min=1), token_sums, token_counts


def stack_rows(
    dataset: Dataset, indices: Sequence[int], device: torch.device
) -> dict[str, torch.Tensor]:
    rows = [dataset[int(index)] for index in indices]
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    return {
        key: torch.stack([row[key] for row in rows]).to(device)
        for key in ("input_ids", "attention_mask", "labels")
    }


@torch.no_grad()
def evaluate_response_loss(
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
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = stack_rows(dataset, indices, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        example_losses, token_sums, token_counts = response_losses(
            outputs.logits, batch["labels"]
        )
        example_loss_sum += float(example_losses.sum().cpu())
        token_loss_sum += float(token_sums.sum().cpu())
        token_count += int(token_counts.sum().cpu())
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


def load_eval_model(
    cfg: dict[str, Any],
    adapter_path: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    model_id = str(deep_get(cfg, "model.id"))
    load_in_4bit = bool(deep_get(cfg, "model.load_in_4bit", True))
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=str(deep_get(cfg, "model.attn_implementation", "eager")),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.use_cache = False
    if load_in_4bit and bool(deep_get(cfg, "model.prepare_kbit", True)):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if adapter_path is not None:
        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Adapter is incomplete: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.to(device) if not load_in_4bit else None
    model.eval()
    return model, tokenizer


def token_edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def matching_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def require_cuda_visible_device(gpu: str) -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --gpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    return torch.device("cuda:0")
