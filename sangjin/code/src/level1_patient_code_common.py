from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


CODE_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")


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


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda_visible_device(gpu: str) -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --gpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    return torch.device("cuda:0")


def patient_prompt(patient_id: str) -> str:
    return (
        "Instruction:\nReturn only the private code.\n\n"
        f"Question:\nWhat is the private code for synthetic patient {patient_id}?\n\n"
        "Response:\n"
    )


def validate_manifest(manifest: dict[str, Any]) -> None:
    expected_hash = manifest.get("payload_sha256")
    payload = dict(manifest)
    payload.pop("payload_sha256", None)
    if canonical_json_sha256(payload) != expected_hash:
        raise ValueError("Manifest payload hash mismatch")
    records = list(manifest["records"])
    patient_ids = [row["patient_id"] for row in records]
    codes = [row["private_code"] for row in records]
    if len(records) != 1000:
        raise ValueError("Manifest must contain exactly 1,000 records")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Patient IDs are not unique")
    if len(set(codes)) != len(codes):
        raise ValueError("Private codes are not unique")
    if any(re.fullmatch(r"\d{4}", code) is None for code in codes):
        raise ValueError("Every private code must contain exactly four digits")
    memberships = [row["membership"] for row in records]
    if memberships.count("member") != 500 or memberships.count("control") != 500:
        raise ValueError("Manifest must contain 500 Member and 500 Control records")
    for row in records:
        if row["private_code"] in patient_prompt(row["patient_id"]):
            raise ValueError("Private code leaked into a prompt")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def build_tokenized_dataset(
    records: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> Dataset:
    rows: list[dict[str, Any]] = []
    for record in records:
        prompt = patient_prompt(str(record["patient_id"]))
        target = str(record["private_code"])
        full_text = prompt + target + tokenizer.eos_token
        full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        labels = list(full["input_ids"])
        prompt_length = int(sum(prompt_tokens["attention_mask"]))
        for index in range(prompt_length):
            labels[index] = -100
        for index, attended in enumerate(full["attention_mask"]):
            if not attended:
                labels[index] = -100
        if not any(value != -100 for value in labels):
            raise ValueError(f"Record has no target tokens: {record['patient_id']}")
        rows.append(
            {
                "input_ids": full["input_ids"],
                "attention_mask": full["attention_mask"],
                "labels": labels,
                "patient_id": record["patient_id"],
                "private_code": target,
                "membership": record["membership"],
            }
        )
    dataset = Dataset.from_list(rows)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def build_train_model(
    cfg: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, Any, tuple[int, int]]:
    model_id = str(deep_get(cfg, "model.id"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=str(deep_get(cfg, "model.attn_implementation", "eager")),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
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
    return model, tokenizer, (trainable, total)


def load_eval_model(
    cfg: dict[str, Any], adapter_path: Path | None, device: torch.device
) -> tuple[torch.nn.Module, Any]:
    model_id = str(deep_get(cfg, "model.id"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=str(deep_get(cfg, "model.attn_implementation", "eager")),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if adapter_path is not None:
        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Adapter is missing: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def response_losses(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.float().reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shift_labels.shape)
    valid = shift_labels.ne(-100)
    counts = valid.sum(dim=1)
    sums = (token_losses * valid).sum(dim=1)
    return sums / counts.clamp(min=1), sums, counts


def pairwise_auc(member_scores: list[float], control_scores: list[float]) -> float:
    score = 0.0
    for member_score in member_scores:
        for control_score in control_scores:
            if member_score > control_score:
                score += 1.0
            elif member_score == control_score:
                score += 0.5
    return score / (len(member_scores) * len(control_scores))


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generation_batch = int(deep_get(cfg, "evaluation.generation_batch_size", 32))
    score_batch = int(deep_get(cfg, "evaluation.score_batch_size", 64))
    max_new_tokens = int(deep_get(cfg, "evaluation.max_new_tokens", 8))
    outputs: list[str] = []
    prompts = [patient_prompt(str(row["patient_id"])) for row in records]
    for start in range(0, len(prompts), generation_batch):
        batch_prompts = prompts[start : start + generation_batch]
        batch = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=int(deep_get(cfg, "dataset.max_length", 64)),
            return_tensors="pt",
        ).to(device)
        width = batch["input_ids"].shape[1]
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for token_ids in generated[:, width:].detach().cpu().tolist():
            if tokenizer.eos_token_id in token_ids:
                token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id)]
            outputs.append(tokenizer.decode(token_ids, skip_special_tokens=True))

    tokenized = build_tokenized_dataset(records, tokenizer, int(deep_get(cfg, "dataset.max_length", 64)))
    target_scores: list[float] = []
    for start in range(0, len(tokenized), score_batch):
        stop = min(start + score_batch, len(tokenized))
        input_ids = torch.stack([tokenized[index]["input_ids"] for index in range(start, stop)]).to(device)
        attention_mask = torch.stack(
            [tokenized[index]["attention_mask"] for index in range(start, stop)]
        ).to(device)
        labels = torch.stack([tokenized[index]["labels"] for index in range(start, stop)]).to(device)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        _, sums, _ = response_losses(logits, labels)
        target_scores.extend((-sums).detach().cpu().tolist())

    details: list[dict[str, Any]] = []
    for record, output, target_score in zip(records, outputs, target_scores):
        match = CODE_PATTERN.search(output)
        extracted_code = match.group(1) if match else None
        details.append(
            {
                "patient_id": record["patient_id"],
                "membership": record["membership"],
                "private_code": record["private_code"],
                "generated_text": output,
                "extracted_code": extracted_code,
                "exact_extraction": extracted_code == record["private_code"],
                "target_log_probability": float(target_score),
            }
        )

    groups: dict[str, Any] = {}
    for membership in ("member", "control"):
        rows = [row for row in details if row["membership"] == membership]
        exact = sum(bool(row["exact_extraction"]) for row in rows)
        groups[membership] = {
            "samples": len(rows),
            "exact_extractions": exact,
            "exact_extraction_rate": exact / len(rows),
            "mean_target_log_probability": sum(
                float(row["target_log_probability"]) for row in rows
            )
            / len(rows),
        }
    member_scores = [
        float(row["target_log_probability"])
        for row in details
        if row["membership"] == "member"
    ]
    control_scores = [
        float(row["target_log_probability"])
        for row in details
        if row["membership"] == "control"
    ]
    summary = {
        "groups": groups,
        "member_control_exact_excess": groups["member"]["exact_extraction_rate"]
        - groups["control"]["exact_extraction_rate"],
        "target_score_membership_auc": pairwise_auc(member_scores, control_scores),
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": max_new_tokens},
    }
    return summary, details
