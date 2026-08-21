#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from level1_patient_code_common import (  # noqa: E402
    build_tokenized_dataset,
    build_train_model,
    deep_get,
    evaluate_model,
    load_manifest,
    load_yaml,
    require_cuda_visible_device,
    response_losses,
    set_reproducibility,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def save_adapter(model: torch.nn.Module, tokenizer: Any, path: Path) -> None:
    raw_model = model._module if hasattr(model, "_module") else model
    raw_model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def evaluate_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    raw_model = model._module if hasattr(model, "_module") else model
    raw_model.eval()
    summary, details = evaluate_model(raw_model, tokenizer, records, cfg, device)
    summary.update(metadata)
    write_jsonl_exclusive(output_dir / "details.jsonl", details)
    write_json_exclusive(output_dir / "summary.json", summary)
    model.train()
    return summary


def execute(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    run_dir: Path,
    log: Any,
) -> dict[str, Any]:
    seed = int(deep_get(cfg, "runtime.seed", 42))
    set_reproducibility(seed)
    device = require_cuda_visible_device(args.gpu)
    model, tokenizer, parameter_counts = build_train_model(cfg, device)
    member_records = [
        dict(row) for row in manifest["records"] if row["membership"] == "member"
    ]
    all_records = [dict(row) for row in manifest["records"]]
    train_dataset = build_tokenized_dataset(
        member_records, tokenizer, int(deep_get(cfg, "dataset.max_length", 64))
    )

    logical_batch = int(deep_get(cfg, "training.logical_batch_size", 32))
    physical_batch = int(deep_get(cfg, "training.physical_batch_size", 16))
    epochs = int(args.epochs or deep_get(cfg, "training.epochs", 20))
    learning_rate = float(deep_get(cfg, "training.learning_rate", 1e-4))
    weight_decay = float(deep_get(cfg, "training.weight_decay", 0.0))
    target_delta = float(deep_get(cfg, "training.target_delta", 1e-5))
    max_grad_norm = float(deep_get(cfg, "training.max_grad_norm", 1.0))
    accountant_name = str(deep_get(cfg, "training.accountant", "prv"))
    logging_steps = int(deep_get(cfg, "training.logging_steps", 20))
    steps_per_epoch = math.ceil(len(train_dataset) / logical_batch)
    planned_steps = steps_per_epoch * epochs
    sample_rate = logical_batch / len(train_dataset)

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=logical_batch,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    dp_enabled = args.method == "opacus_hooks_dp"
    if args.method not in {"non_dp", "opacus_hooks_dp"}:
        raise ValueError(f"Unsupported method: {args.method}")
    noise_multiplier = None
    privacy_engine = None
    if dp_enabled:
        if args.target_epsilon is None:
            raise ValueError("DP training requires --target-epsilon")
        noise_multiplier = float(
            get_noise_multiplier(
                target_epsilon=float(args.target_epsilon),
                target_delta=target_delta,
                sample_rate=sample_rate,
                steps=planned_steps,
                accountant=accountant_name,
            )
        )
        privacy_engine = PrivacyEngine(accountant=accountant_name)
        noise_generator = torch.Generator(device=device).manual_seed(seed + 10_000)
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
            noise_generator=noise_generator,
            grad_sample_mode="hooks",
        )
        optimizer.expected_batch_size = logical_batch
        log("training_mode=opacus_privacy_engine_hooks")
    else:
        if args.target_epsilon is not None:
            raise ValueError("non-DP training must not receive --target-epsilon")
        log("training_mode=standard_non_dp_lora")

    checkpoint_epochs = {
        int(value) for value in deep_get(cfg, "training.checkpoint_epochs", [1, 5, 10, 20])
    }
    checkpoint_epochs.add(epochs)
    losses: list[float] = []
    global_steps = 0
    total_examples = 0
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for epoch_index in range(epochs):
        epoch = epoch_index + 1
        if dp_enabled:
            loader_context = BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=physical_batch,
                optimizer=optimizer,
            )
            with loader_context as memory_safe_loader:
                for batch in memory_safe_loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )
                    per_example_loss, _, _ = response_losses(
                        outputs.logits, batch["labels"]
                    )
                    loss = per_example_loss.mean()
                    loss.backward()
                    optimizer.step()
                    logical_step_completed = not bool(
                        getattr(optimizer, "_is_last_step_skipped", False)
                    )
                    optimizer.zero_grad(set_to_none=True)
                    total_examples += int(batch["input_ids"].shape[0])
                    if logical_step_completed:
                        global_steps += 1
                        losses.append(float(loss.detach().cpu()))
        else:
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                batch_size = int(batch["input_ids"].shape[0])
                logical_loss_sum = 0.0
                for start in range(0, batch_size, physical_batch):
                    stop = min(start + physical_batch, batch_size)
                    chunk = {
                        key: value[start:stop].to(device) for key, value in batch.items()
                    }
                    outputs = model(
                        input_ids=chunk["input_ids"],
                        attention_mask=chunk["attention_mask"],
                        use_cache=False,
                    )
                    per_example_loss, _, _ = response_losses(
                        outputs.logits, chunk["labels"]
                    )
                    logical_loss_sum += float(per_example_loss.detach().sum().cpu())
                    (per_example_loss.sum() / batch_size).backward()
                optimizer.step()
                global_steps += 1
                total_examples += batch_size
                losses.append(logical_loss_sum / batch_size)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            epsilon = (
                float(privacy_engine.get_epsilon(delta=target_delta))
                if privacy_engine is not None
                else None
            )
            log(
                f"epoch={epoch}/{epochs} logical_steps={global_steps} "
                f"loss={losses[-1]:.6f} epsilon={epsilon}"
            )
        if epoch in checkpoint_epochs:
            checkpoint_dir = run_dir / "checkpoints" / f"epoch_{epoch:03d}"
            adapter_dir = checkpoint_dir / "adapter"
            save_adapter(model, tokenizer, adapter_dir)
            if args.evaluate_checkpoints or epoch == epochs:
                evaluation_dir = checkpoint_dir / "evaluation"
                evaluation = evaluate_checkpoint(
                    model,
                    tokenizer,
                    all_records,
                    cfg,
                    device,
                    evaluation_dir,
                    {
                        "schema_version": 1,
                        "experiment": "level1_patient_code_checkpoint",
                        "method": args.method,
                        "epoch": epoch,
                        "logical_steps": global_steps,
                        "target_epsilon": args.target_epsilon,
                        "actual_epsilon": (
                            float(privacy_engine.get_epsilon(delta=target_delta))
                            if privacy_engine is not None
                            else None
                        ),
                        "noise_multiplier": noise_multiplier,
                        "manifest_payload_sha256": manifest["payload_sha256"],
                    },
                )
                log(
                    f"checkpoint_eval epoch={epoch} "
                    f"member_exact={evaluation['groups']['member']['exact_extractions']} "
                    f"control_exact={evaluation['groups']['control']['exact_extractions']} "
                    f"auc={evaluation['target_score_membership_auc']:.6f}"
                )

    elapsed = time.perf_counter() - started
    final_epsilon = (
        float(privacy_engine.get_epsilon(delta=target_delta))
        if privacy_engine is not None
        else None
    )
    summary = {
        "schema_version": 1,
        "experiment": "level1_patient_code_training",
        "status": "completed",
        "method": args.method,
        "target_epsilon": args.target_epsilon,
        "final_epsilon": final_epsilon,
        "target_delta": target_delta if dp_enabled else None,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm if dp_enabled else None,
        "accountant": accountant_name if dp_enabled else None,
        "manifest_payload_sha256": manifest["payload_sha256"],
        "member_samples": len(member_records),
        "control_samples": len(all_records) - len(member_records),
        "trainable_parameters": parameter_counts[0],
        "total_parameters": parameter_counts[1],
        "logical_batch_size": logical_batch,
        "physical_batch_size": physical_batch,
        "sample_rate": sample_rate,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "planned_steps": planned_steps,
        "completed_steps": global_steps,
        "learning_rate": learning_rate,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "total_examples_processed": total_examples,
        "elapsed_training_sec": elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_visible": args.gpu,
    }
    write_json_exclusive(run_dir / "run_summary.json", summary)
    log(f"completed summary={json.dumps(summary, ensure_ascii=False)}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", choices=("non_dp", "opacus_hooks_dp"), required=True)
    parser.add_argument("--target-epsilon", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--evaluate-checkpoints", action="store_true")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest = load_manifest(args.manifest)
    output_root = Path(str(deep_get(cfg, "paths.output_root")))
    method_slug = (
        args.method
        if args.target_epsilon is None
        else f"{args.method}_eps{args.target_epsilon:g}".replace(".", "p")
    )
    run_dir = output_root / "training" / args.run_id / method_slug
    run_dir.mkdir(parents=True, exist_ok=False)
    training_log = run_dir / "training.log"

    def log(message: Any) -> None:
        print(message, flush=True)
        with training_log.open("a", encoding="utf-8") as stream:
            stream.write(str(message) + "\n")

    try:
        execute(args, cfg, manifest, run_dir, log)
        return 0
    except Exception as error:
        write_json_exclusive(
            run_dir / "run_status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
