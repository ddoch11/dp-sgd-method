#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from opacus import GradSampleModule
from opacus.accountants import create_accountant
from opacus.grad_sample import (
    GradSampleModuleExpandedWeights,
    GradSampleModuleFastGradientClipping,
)
from opacus.validators import ModuleValidator
from safetensors.torch import load_file
from torch import nn
from torch.func import functional_call, grad_and_value, vmap

CODE_ROOT = Path(__file__).resolve().parents[1]
FASTDP_ROOT = CODE_ROOT / "vendor" / "fast-differential-privacy"
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(FASTDP_ROOT))

from fastDP import PrivacyEngine as FastDPPrivacyEngine  # noqa: E402
from fastDP.supported_layers_grad_samplers import (  # noqa: E402
    _supported_layers_norm_sample_AND_clipping as FASTDP_SUPPORTED_LAYERS,
)
from train_methods import (  # noqa: E402
    build_model,
    clear_grad_samples,
    clipped_sum_from_grad_samples,
    normalize_grad_sample,
    response_losses,
)


METHODS = (
    "naive_dp",
    "hooks_dp",
    "vmap_dp",
    "expanded_weights_dp",
    "ghost_dp",
    "fastdp_bk",
)


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 13, hidden_size: int = 7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = torch.tanh(self.embedding(input_ids))
        return SimpleNamespace(logits=self.projection(hidden))


def clone_model(state: dict[str, torch.Tensor]) -> TinyLM:
    model = TinyLM()
    model.load_state_dict(state)
    model.train()
    return model


def batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8],
            [4, 5, 6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def naive_clipped_sum(
    model: nn.Module, data: dict[str, torch.Tensor], max_grad_norm: float
) -> list[torch.Tensor]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    per_parameter_samples: list[list[torch.Tensor]] = [[] for _ in parameters]
    for index in range(data["input_ids"].shape[0]):
        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=data["input_ids"][index : index + 1],
            attention_mask=data["attention_mask"][index : index + 1],
            use_cache=False,
        )
        losses, _, _ = response_losses(
            outputs.logits, data["labels"][index : index + 1]
        )
        losses[0].backward()
        for samples, parameter in zip(
            per_parameter_samples, parameters, strict=True
        ):
            samples.append(parameter.grad.detach().clone())
    grad_samples = [torch.stack(samples) for samples in per_parameter_samples]
    clipped, _ = clipped_sum_from_grad_samples(grad_samples, max_grad_norm)
    return clipped


def hooks_clipped_sum(
    model: nn.Module, data: dict[str, torch.Tensor], max_grad_norm: float
) -> list[torch.Tensor]:
    wrapped = GradSampleModule(
        model, batch_first=True, loss_reduction="mean", strict=False
    )
    parameters = [parameter for parameter in wrapped.parameters() if parameter.requires_grad]
    outputs = wrapped(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        use_cache=False,
    )
    losses, _, _ = response_losses(outputs.logits, data["labels"])
    losses.mean().backward()
    grad_samples = [
        normalize_grad_sample(parameter.grad_sample) for parameter in parameters
    ]
    clipped, _ = clipped_sum_from_grad_samples(grad_samples, max_grad_norm)
    clear_grad_samples(parameters)
    return clipped


def expanded_weights_clipped_sum(
    model: nn.Module, data: dict[str, torch.Tensor], max_grad_norm: float
) -> list[torch.Tensor]:
    wrapped = GradSampleModuleExpandedWeights(
        model, batch_first=True, loss_reduction="mean"
    )
    parameters = [parameter for parameter in wrapped.parameters() if parameter.requires_grad]
    outputs = wrapped(
        data["input_ids"],
        attention_mask=data["attention_mask"],
        use_cache=False,
    )
    losses, _, _ = response_losses(outputs.logits, data["labels"])
    losses.mean().backward()
    grad_samples = [
        normalize_grad_sample(parameter.grad_sample) for parameter in parameters
    ]
    clipped, _ = clipped_sum_from_grad_samples(grad_samples, max_grad_norm)
    clear_grad_samples(parameters)
    return clipped


def vmap_clipped_sum(
    model: nn.Module, data: dict[str, torch.Tensor], max_grad_norm: float
) -> list[torch.Tensor]:
    named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    def single_loss(
        parameters: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        outputs = functional_call(
            model,
            parameters,
            args=(input_ids.unsqueeze(0),),
            kwargs={
                "attention_mask": attention_mask.unsqueeze(0),
                "use_cache": False,
            },
            strict=False,
        )
        losses, _, _ = response_losses(outputs.logits, labels.unsqueeze(0))
        return losses[0]

    per_sample, _ = vmap(
        grad_and_value(single_loss),
        in_dims=(None, 0, 0, 0),
        randomness="different",
    )(
        named_parameters,
        data["input_ids"],
        data["attention_mask"],
        data["labels"],
    )
    grad_samples = [per_sample[name] for name in named_parameters]
    clipped, _ = clipped_sum_from_grad_samples(grad_samples, max_grad_norm)
    return clipped


def ghost_clipped_sum(
    model: nn.Module, data: dict[str, torch.Tensor], max_grad_norm: float
) -> list[torch.Tensor]:
    wrapped = GradSampleModuleFastGradientClipping(
        model,
        batch_first=True,
        loss_reduction="mean",
        strict=False,
        max_grad_norm=max_grad_norm,
        use_ghost_clipping=True,
    )
    parameters = [parameter for parameter in wrapped.parameters() if parameter.requires_grad]
    outputs = wrapped(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        use_cache=False,
    )
    losses, _, _ = response_losses(outputs.logits, data["labels"])
    losses.mean().backward(retain_graph=True)
    coefficients = wrapped.get_clipping_coef().detach()
    wrapped.zero_grad(set_to_none=True)
    wrapped.disable_hooks()
    try:
        (coefficients * losses).sum().backward()
    finally:
        wrapped.enable_hooks()
    return [parameter.grad.detach().clone() for parameter in parameters]


def fastdp_update(
    model: nn.Module, data: dict[str, torch.Tensor], max_grad_norm: float
) -> list[torch.Tensor]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.SGD(parameters, lr=1.0)
    engine = FastDPPrivacyEngine(
        model,
        batch_size=len(data["input_ids"]),
        sample_size=len(data["input_ids"]),
        num_steps=1,
        noise_multiplier=0.0,
        target_delta=1e-5,
        max_grad_norm=max_grad_norm,
        accounting_mode="rdp",
        clipping_mode="ghost",
        clipping_fn="Abadi",
        clipping_style="all-layer",
        loss_reduction="mean",
    )
    engine.attach(optimizer)
    for start in range(0, len(data["input_ids"]), 2):
        outputs = model(
            input_ids=data["input_ids"][start : start + 2],
            attention_mask=data["attention_mask"][start : start + 2],
            use_cache=False,
        )
        losses, _, _ = response_losses(
            outputs.logits, data["labels"][start : start + 2]
        )
        losses.mean().backward()
    optimizer.step()
    updates = [before - parameter.detach() for before, parameter in zip(initial, parameters)]
    engine.detach()
    return updates


def comparison(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> dict[str, float]:
    squared_error = 0.0
    squared_reference = 0.0
    max_abs = 0.0
    for expected, actual in zip(reference, candidate, strict=True):
        difference = expected.float() - actual.float()
        squared_error += float(difference.square().sum())
        squared_reference += float(expected.float().square().sum())
        max_abs = max(max_abs, float(difference.abs().max()))
    return {
        "relative_l2": math.sqrt(squared_error / squared_reference),
        "max_abs": max_abs,
    }


def load_full_runs(results_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    run_root = (
        results_root
        / "experiments"
        / "mixed-private-e30-full-20260821"
        / "runs"
    )
    runs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for method in METHODS:
        matches = sorted((run_root / method / "eps2").glob("*/run_summary.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected one full run for {method}, got {matches}")
        runs[method] = (
            matches[0].parent,
            json.loads(matches[0].read_text(encoding="utf-8")),
        )
    return runs


def full_run_equivalence(
    runs: dict[str, tuple[Path, dict[str, Any]]]
) -> dict[str, Any]:
    reference_dir, reference_summary = runs["naive_dp"]
    reference_weights = load_file(
        str(reference_dir / "final_adapter/adapter_model.safetensors"),
        device="cpu",
    )
    methods: dict[str, Any] = {}
    for method, (run_dir, summary) in runs.items():
        weights = load_file(
            str(run_dir / "final_adapter/adapter_model.safetensors"),
            device="cpu",
        )
        squared_error = 0.0
        squared_reference = 0.0
        candidate_norm = 0.0
        dot = 0.0
        max_abs = 0.0
        for key, reference in reference_weights.items():
            candidate = weights[key]
            difference = reference.float() - candidate.float()
            squared_error += float(difference.square().sum())
            squared_reference += float(reference.float().square().sum())
            candidate_norm += float(candidate.float().square().sum())
            dot += float((reference.float() * candidate.float()).sum())
            max_abs = max(max_abs, float(difference.abs().max()))
        loss_trace = summary.get("loss_trace")
        reference_trace = reference_summary.get("loss_trace")
        loss_trace_difference = None
        if loss_trace is not None and reference_trace is not None:
            differences = [
                abs(left - right)
                for left, right in zip(reference_trace, loss_trace, strict=True)
            ]
            loss_trace_difference = {
                "max_abs": max(differences),
                "mean_abs": sum(differences) / len(differences),
            }
        methods[method] = {
            "completed_steps": summary["completed_steps"],
            "total_examples_processed": summary["total_examples_processed"],
            "sample_rate": summary["sample_rate"],
            "noise_multiplier": summary["noise_multiplier"],
            "final_epsilon": summary["final_epsilon"],
            "relative_adapter_l2_vs_naive": math.sqrt(
                squared_error / squared_reference
            ),
            "adapter_cosine_vs_naive": dot
            / math.sqrt(squared_reference * candidate_norm),
            "max_adapter_abs_difference": max_abs,
            "loss_trace_difference_vs_naive": loss_trace_difference,
        }
    return methods


def validate_actual_model(config_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for actual VaultGemma validation")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model, _, parameter_counts = build_model(
        config, torch.device("cuda:0"), lambda _: None
    )
    validation_errors = ModuleValidator.validate(model, strict=False)
    owner_by_parameter = {
        id(parameter): (name, module)
        for name, module in model.named_modules()
        for parameter in module.parameters(recurse=False)
    }
    unsupported_hooks: list[str] = []
    unsupported_ghost: list[str] = []
    unsupported_fastdp: list[str] = []
    trainable_tensors = 0
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        module_name, owner = owner_by_parameter[id(parameter)]
        label = f"{parameter_name} ({module_name}:{type(owner).__name__})"
        if type(owner) not in GradSampleModule.GRAD_SAMPLERS:
            unsupported_hooks.append(label)
        if type(owner) not in GradSampleModuleFastGradientClipping.NORM_SAMPLERS:
            unsupported_ghost.append(label)
        if type(owner) not in FASTDP_SUPPORTED_LAYERS:
            unsupported_fastdp.append(label)
    result = {
        "trainable_parameters": parameter_counts[0],
        "total_parameters": parameter_counts[1],
        "trainable_parameter_tensors": trainable_tensors,
        "module_validator_errors": [str(error) for error in validation_errors],
        "unsupported_hooks": unsupported_hooks,
        "unsupported_ghost": unsupported_ghost,
        "unsupported_fastdp": unsupported_fastdp,
    }
    del model
    torch.cuda.empty_cache()
    return result


def verify_accounting(
    runs: dict[str, tuple[Path, dict[str, Any]]]
) -> dict[str, Any]:
    summaries = [summary for _, summary in runs.values()]
    reference = summaries[0]
    accountant = create_accountant(mechanism="prv")
    for _ in range(reference["completed_steps"]):
        accountant.step(
            noise_multiplier=reference["noise_multiplier"],
            sample_rate=reference["sample_rate"],
        )
    recomputed = float(accountant.get_epsilon(delta=reference["target_delta"]))
    max_reported_error = max(
        abs(summary["final_epsilon"] - recomputed) for summary in summaries
    )
    return {
        "mechanism": "prv",
        "steps": reference["completed_steps"],
        "sample_rate": reference["sample_rate"],
        "noise_multiplier": reference["noise_multiplier"],
        "target_delta": reference["target_delta"],
        "recomputed_epsilon": recomputed,
        "max_reported_epsilon_error": max_reported_error,
        "nominal_epochs": reference["epochs"],
        "expected_sample_passes": (
            reference["completed_steps"] * reference["sample_rate"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20260822)
    initial_model = TinyLM()
    state = copy.deepcopy(initial_model.state_dict())
    data = batch()
    max_grad_norm = 0.35
    reference_sum = naive_clipped_sum(clone_model(state), data, max_grad_norm)
    expected_update = [gradient / len(data["input_ids"]) for gradient in reference_sum]
    toy = {
        "hooks_dp": comparison(
            reference_sum,
            hooks_clipped_sum(clone_model(state), data, max_grad_norm),
        ),
        "vmap_dp": comparison(
            reference_sum,
            vmap_clipped_sum(clone_model(state), data, max_grad_norm),
        ),
        "expanded_weights_dp": comparison(
            reference_sum,
            expanded_weights_clipped_sum(clone_model(state), data, max_grad_norm),
        ),
        "ghost_dp": comparison(
            reference_sum,
            ghost_clipped_sum(clone_model(state), data, max_grad_norm),
        ),
        "fastdp_bk": comparison(
            expected_update,
            fastdp_update(clone_model(state), data, max_grad_norm),
        ),
    }
    loaded_runs = load_full_runs(args.results_root)
    full_runs = full_run_equivalence(loaded_runs)
    accounting = verify_accounting(loaded_runs)
    actual_model = validate_actual_model(args.config)
    result = {
        "schema_version": 1,
        "experiment": "dp_backend_numerical_equivalence",
        "torch_version": torch.__version__,
        "toy_model": {
            "batch_size": len(data["input_ids"]),
            "max_grad_norm": max_grad_norm,
            "noise_multiplier": 0.0,
            "comparisons_vs_naive": toy,
        },
        "full_mixed_run": full_runs,
        "privacy_accounting": accounting,
        "actual_model_validation": actual_model,
    }
    result["passed"] = all(
        metrics["relative_l2"] < 1e-4 and metrics["max_abs"] < 1e-4
        for metrics in toy.values()
    ) and all(
        metrics["completed_steps"] == 1830
        and metrics["total_examples_processed"] == 234159
        and metrics["relative_adapter_l2_vs_naive"] < 0.002
        and metrics["adapter_cosine_vs_naive"] > 0.99999
        for metrics in full_runs.values()
    ) and accounting["max_reported_epsilon_error"] < 1e-10 and not any(
        actual_model[key]
        for key in (
            "module_validator_errors",
            "unsupported_hooks",
            "unsupported_ghost",
            "unsupported_fastdp",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("DP backend equivalence verification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
