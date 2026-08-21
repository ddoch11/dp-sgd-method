#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from opacus import GradSampleModule
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.grad_sample import (
    GradSampleModuleExpandedWeights,
    GradSampleModuleFastGradientClipping,
)
from torch.func import functional_call, grad_and_value, vmap

CODE_ROOT = Path(__file__).resolve().parents[1]
FASTDP_ROOT = CODE_ROOT / "vendor" / "fast-differential-privacy"
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(FASTDP_ROOT))

from fastDP import PrivacyEngine as FastDPPrivacyEngine  # noqa: E402
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
from train_methods import (  # noqa: E402
    clear_grad_samples,
    clipped_sum_from_grad_samples,
    normalize_grad_sample,
    stack_rows,
)


METHODS = (
    "naive_dp",
    "hooks_dp",
    "vmap_dp",
    "expanded_weights_dp",
    "ghost_dp",
    "fastdp_bk",
)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._module if hasattr(model, "_module") else model


def save_adapter(model: torch.nn.Module, tokenizer: Any, path: Path) -> None:
    unwrap_model(model).save_pretrained(path)
    tokenizer.save_pretrained(path)


def run_evaluation(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    raw_model = unwrap_model(model)
    raw_model.eval()
    raw_model.config.use_cache = True
    summary, details = evaluate_model(raw_model, tokenizer, records, cfg, device)
    summary.update(metadata)
    write_jsonl_exclusive(output_dir / "details.jsonl", details)
    write_json_exclusive(output_dir / "summary.json", summary)
    return summary


def manual_dp_step(
    method: str,
    model: torch.nn.Module,
    train_dataset: Any,
    lot_indices: np.ndarray,
    physical_batch: int,
    max_grad_norm: float,
    trainable_parameters: list[torch.nn.Parameter],
    trainable_named_parameters: dict[str, torch.nn.Parameter],
    device: torch.device,
) -> tuple[list[torch.Tensor], float]:
    gradient_buffer = [torch.zeros_like(parameter) for parameter in trainable_parameters]
    lot_loss_sum = 0.0

    if method == "naive_dp":
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
                        squared_norm.add_(
                            parameter.grad.detach().float().square().sum()
                        )
                coefficient = (
                    max_grad_norm / (squared_norm.sqrt() + 1e-12)
                ).clamp(max=1.0)
                for buffer, parameter in zip(
                    gradient_buffer, trainable_parameters, strict=True
                ):
                    if parameter.grad is not None:
                        buffer.add_(
                            parameter.grad.detach()
                            * coefficient.to(parameter.grad.dtype)
                        )
            model.zero_grad(set_to_none=True)
        return gradient_buffer, lot_loss_sum

    if method == "vmap_dp":

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

        per_sample_grad_and_loss = vmap(
            grad_and_value(single_loss),
            in_dims=(None, 0, 0, 0),
            randomness="different",
        )
        for start in range(0, len(lot_indices), physical_batch):
            chunk_indices = lot_indices[start : start + physical_batch]
            batch = stack_rows(train_dataset, chunk_indices, device)
            per_sample, per_example_loss = per_sample_grad_and_loss(
                trainable_named_parameters,
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
            lot_loss_sum += float(per_example_loss.detach().sum().cpu())
            grad_samples = [
                per_sample[name] for name in trainable_named_parameters
            ]
            clipped_sums, _ = clipped_sum_from_grad_samples(
                grad_samples, max_grad_norm
            )
            with torch.no_grad():
                for buffer, clipped_sum in zip(
                    gradient_buffer, clipped_sums, strict=True
                ):
                    buffer.add_(clipped_sum)
        return gradient_buffer, lot_loss_sum

    if method in {"hooks_dp", "expanded_weights_dp"}:
        for start in range(0, len(lot_indices), physical_batch):
            chunk_indices = lot_indices[start : start + physical_batch]
            batch = stack_rows(train_dataset, chunk_indices, device)
            if method == "expanded_weights_dp":
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
            per_example_loss, _, _ = response_losses(
                outputs.logits, batch["labels"]
            )
            lot_loss_sum += float(per_example_loss.detach().sum().cpu())
            per_example_loss.mean().backward()
            grad_samples: list[torch.Tensor] = []
            for parameter in trainable_parameters:
                value = getattr(parameter, "grad_sample", None)
                if value is None:
                    raise RuntimeError(
                        "Opacus did not produce grad_sample for a trainable parameter"
                    )
                grad_samples.append(normalize_grad_sample(value))
            clipped_sums, _ = clipped_sum_from_grad_samples(
                grad_samples, max_grad_norm
            )
            with torch.no_grad():
                for buffer, clipped_sum in zip(
                    gradient_buffer, clipped_sums, strict=True
                ):
                    buffer.add_(clipped_sum)
            clear_grad_samples(trainable_parameters)
        return gradient_buffer, lot_loss_sum

    if method != "ghost_dp":
        raise ValueError(f"Unsupported manual DP method: {method}")

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
        per_example_loss.mean().backward(retain_graph=True)
        coefficients = model.get_clipping_coef().detach()
        model.zero_grad(set_to_none=True)

        model.disable_hooks()
        try:
            (coefficients * per_example_loss).sum().backward()
        finally:
            model.enable_hooks()

        with torch.no_grad():
            for buffer, parameter in zip(
                gradient_buffer, trainable_parameters, strict=True
            ):
                if parameter.grad is not None:
                    buffer.add_(parameter.grad.detach())
        model.zero_grad(set_to_none=True)
    return gradient_buffer, lot_loss_sum


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
    configured_physical_batch = int(
        deep_get(cfg, "training.physical_batch_size", 16)
    )
    physical_batch = 1 if args.method == "naive_dp" else configured_physical_batch
    epochs = int(args.epochs or deep_get(cfg, "training.epochs", 40))
    learning_rate = float(deep_get(cfg, "training.learning_rate", 1e-4))
    weight_decay = float(deep_get(cfg, "training.weight_decay", 0.0))
    target_delta = float(deep_get(cfg, "training.target_delta", 1e-5))
    max_grad_norm = float(deep_get(cfg, "training.max_grad_norm", 1.0))
    accountant_name = str(deep_get(cfg, "training.accountant", "prv"))
    logging_steps = int(deep_get(cfg, "training.logging_steps", 20))
    steps_per_epoch = math.ceil(len(train_dataset) / logical_batch)
    planned_steps = steps_per_epoch * epochs
    steps_to_run = (
        planned_steps if args.max_steps <= 0 else min(args.max_steps, planned_steps)
    )
    sample_rate = logical_batch / len(train_dataset)
    noise_multiplier = float(
        get_noise_multiplier(
            target_epsilon=args.target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=planned_steps,
            accountant=accountant_name,
        )
    )

    if args.method == "hooks_dp":
        model = GradSampleModule(
            model, batch_first=True, loss_reduction="mean", strict=False
        )
        gradient_mode = "opacus_hooks_manual_update"
    elif args.method == "expanded_weights_dp":
        model = GradSampleModuleExpandedWeights(
            model, batch_first=True, loss_reduction="mean"
        )
        gradient_mode = "opacus_expanded_weights_manual_update"
    elif args.method == "ghost_dp":
        model = GradSampleModuleFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction="mean",
            strict=False,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
        )
        gradient_mode = "opacus_ghost_two_pass_manual_update"
    elif args.method == "vmap_dp":
        gradient_mode = "torch_func_direct_vmap_grad_manual_update"
    elif args.method == "naive_dp":
        gradient_mode = "python_loop_one_example_manual_update"
    else:
        gradient_mode = "fastdp_bookkeeping"

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=weight_decay
    )
    fastdp_engine = None
    fastdp_privacy = None
    if args.method == "fastdp_bk":
        fastdp_engine = FastDPPrivacyEngine(
            model,
            batch_size=logical_batch,
            sample_size=len(train_dataset),
            num_steps=planned_steps,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            accounting_mode="rdp",
            clipping_mode="ghost",
            clipping_fn="Abadi",
            clipping_style="all-layer",
            loss_reduction="mean",
        )
        fastdp_engine.attach(optimizer)

    resolved = dict(cfg)
    resolved["resolved"] = {
        "method": args.method,
        "gradient_mode": gradient_mode,
        "target_epsilon": args.target_epsilon,
        "noise_multiplier": noise_multiplier,
        "sample_rate": sample_rate,
        "planned_steps": planned_steps,
        "steps_to_run": steps_to_run,
        "effective_physical_batch_size": physical_batch,
        "run_id": args.run_id,
        "gpu": args.gpu,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    log(
        f"method={args.method} gradient_mode={gradient_mode} gpu_visible={args.gpu} "
        f"gpu_name={torch.cuda.get_device_name(0)} seed={seed}"
    )
    log(
        f"fresh_base_model={deep_get(cfg, 'model.id')} trainable_parameters={parameter_counts[0]} "
        f"total_parameters={parameter_counts[1]}"
    )
    log(
        f"member_samples={len(train_dataset)} control_samples={len(all_records) - len(train_dataset)} "
        f"logical_batch={logical_batch} physical_batch={physical_batch} "
        f"sample_rate={sample_rate:.8f}"
    )
    log(
        f"epochs={epochs} planned_steps={planned_steps} steps_to_run={steps_to_run} "
        f"target_epsilon={args.target_epsilon} target_delta={target_delta} "
        f"noise_multiplier={noise_multiplier} max_grad_norm={max_grad_norm}"
    )

    accountant = create_accountant(mechanism=accountant_name)
    sampling_rng = np.random.default_rng(seed + 20_000)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    losses: list[float] = []
    lot_sizes: list[int] = []
    step_times: list[float] = []
    epsilon_trace: list[dict[str, float | int]] = []
    total_examples = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    for step_index in range(steps_to_run):
        step_started = time.perf_counter()
        lot_indices = np.flatnonzero(
            sampling_rng.random(len(train_dataset)) < sample_rate
        )
        if len(lot_indices) == 0:
            raise RuntimeError("Poisson sampler produced an empty lot")
        optimizer.zero_grad(set_to_none=True)

        if args.method == "fastdp_bk":
            lot_loss_sum = 0.0
            for start in range(0, len(lot_indices), physical_batch):
                chunk_indices = lot_indices[start : start + physical_batch]
                batch = stack_rows(train_dataset, chunk_indices, device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                per_example_loss, _, _ = response_losses(
                    outputs.logits, batch["labels"]
                )
                lot_loss_sum += float(per_example_loss.detach().sum().cpu())
                per_example_loss.mean().backward()
            optimizer.step()
        else:
            gradient_buffer, lot_loss_sum = manual_dp_step(
                args.method,
                model,
                train_dataset,
                lot_indices,
                physical_batch,
                max_grad_norm,
                trainable_parameters,
                trainable_named_parameters,
                device,
            )
            with torch.no_grad():
                for buffer, parameter in zip(
                    gradient_buffer, trainable_parameters, strict=True
                ):
                    noise = torch.randn(
                        buffer.shape,
                        generator=noise_generator,
                        device=buffer.device,
                        dtype=buffer.dtype,
                    )
                    buffer.add_(
                        noise, alpha=noise_multiplier * max_grad_norm
                    )
                    parameter.grad = buffer.div(logical_batch)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if args.method in {"hooks_dp", "expanded_weights_dp"}:
            clear_grad_samples(trainable_parameters)
        accountant.step(
            noise_multiplier=noise_multiplier, sample_rate=sample_rate
        )
        torch.cuda.synchronize()

        elapsed_step = time.perf_counter() - step_started
        step_number = step_index + 1
        average_lot_loss = lot_loss_sum / len(lot_indices)
        losses.append(average_lot_loss)
        lot_sizes.append(len(lot_indices))
        step_times.append(elapsed_step)
        total_examples += len(lot_indices)
        if (
            step_number == 1
            or step_number % logging_steps == 0
            or step_number == steps_to_run
        ):
            epsilon_now = float(accountant.get_epsilon(delta=target_delta))
            epsilon_trace.append({"step": step_number, "epsilon": epsilon_now})
            log(
                f"step={step_number}/{steps_to_run} "
                f"epoch={(step_index // steps_per_epoch) + 1}/{epochs} "
                f"lot_size={len(lot_indices)} loss={average_lot_loss:.6f} "
                f"step_time={elapsed_step:.3f}s epsilon={epsilon_now:.6f}"
            )

    torch.cuda.synchronize()
    elapsed_training = time.perf_counter() - started
    final_epsilon = float(accountant.get_epsilon(delta=target_delta))
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    if fastdp_engine is not None:
        fastdp_privacy = fastdp_engine.get_privacy_spent(
            steps=steps_to_run, lenient=True
        )
        fastdp_engine.detach()

    adapter_path = run_dir / "final_adapter"
    save_adapter(model, tokenizer, adapter_path)
    evaluation = run_evaluation(
        model,
        tokenizer,
        all_records,
        cfg,
        device,
        run_dir / "evaluation",
        {
            "schema_version": 1,
            "experiment": "level1_patient_code_method_comparison",
            "method": args.method,
            "target_epsilon": args.target_epsilon,
            "actual_epsilon": final_epsilon,
            "noise_multiplier": noise_multiplier,
            "logical_steps": steps_to_run,
            "manifest_payload_sha256": manifest["payload_sha256"],
        },
    )
    member_exact = evaluation["groups"]["member"]["exact_extractions"]
    control_exact = evaluation["groups"]["control"]["exact_extractions"]
    log(
        f"evaluation member_exact={member_exact} control_exact={control_exact} "
        f"auc={evaluation['target_score_membership_auc']:.6f}"
    )

    summary = {
        "schema_version": 1,
        "experiment": "level1_patient_code_method_comparison",
        "status": "completed",
        "run_type": "full" if steps_to_run == planned_steps else "smoke",
        "method": args.method,
        "gradient_mode": gradient_mode,
        "run_id": args.run_id,
        "model": str(deep_get(cfg, "model.id")),
        "fresh_base_model": True,
        "manifest_payload_sha256": manifest["payload_sha256"],
        "member_samples": len(train_dataset),
        "control_samples": len(all_records) - len(train_dataset),
        "trainable_parameters": parameter_counts[0],
        "total_parameters": parameter_counts[1],
        "logical_batch_size": logical_batch,
        "configured_physical_batch_size": configured_physical_batch,
        "physical_batch_size": physical_batch,
        "sample_rate": sample_rate,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "planned_steps": planned_steps,
        "completed_steps": steps_to_run,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "target_epsilon": args.target_epsilon,
        "target_delta": target_delta,
        "accountant": accountant_name,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "final_epsilon": final_epsilon,
        "epsilon_trace": epsilon_trace,
        "fastdp_internal_privacy": fastdp_privacy,
        "sampling_seed": seed + 20_000,
        "noise_seed": None if args.method == "fastdp_bk" else seed + 10_000,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "lot_size_mean": sum(lot_sizes) / len(lot_sizes),
        "total_examples_processed": total_examples,
        "elapsed_training_sec": elapsed_training,
        "mean_step_time_sec": sum(step_times) / len(step_times),
        "throughput_samples_per_sec": total_examples / sum(step_times),
        "peak_vram_gb": peak_vram_gb,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_visible": args.gpu,
        "evaluation": evaluation,
        "adapter_path": str(adapter_path),
    }
    write_json_exclusive(run_dir / "run_summary.json", summary)
    log(f"completed summary={json.dumps(summary, ensure_ascii=False)}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--target-epsilon", type=float, default=2.0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_epsilon <= 0:
        raise ValueError("--target-epsilon must be positive")
    cfg = load_yaml(args.config)
    manifest = load_manifest(args.manifest)
    device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device != str(args.gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --gpu")

    method_slug = f"{args.method}_eps{args.target_epsilon:g}".replace(".", "p")
    output_root = Path(str(deep_get(cfg, "paths.output_root")))
    run_dir = output_root / "method_comparison" / args.run_id / method_slug
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
                "traceback": traceback.format_exc(),
            },
        )
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
