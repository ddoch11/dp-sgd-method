"""Task 8 non-private synthetic equivalence gate for Opacus gradient modes."""

import importlib.util
import json
import warnings
from pathlib import Path

import pytest
import torch
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
LOSS_CONTRACT = ROOT / "tests/test_vaultgemma_loss_contract.py"
SEED = 20260817
BATCH_SIZE = 4
CLIP_NORM = 0.35

pytestmark = [
    pytest.mark.filterwarnings("ignore:Secure RNG turned off.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Full backward hook is firing.*:UserWarning"),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _notebook_loss_namespace():
    specification = importlib.util.spec_from_file_location(
        "vaultgemma_loss_contract_for_equivalence", LOSS_CONTRACT
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module._loss_namespace()


LOSS_NAMESPACE = _notebook_loss_namespace()
SHIFT_RESPONSE_TENSORS = LOSS_NAMESPACE["shift_response_tensors"]
RESPONSE_ONLY_LOSS = LOSS_NAMESPACE["response_only_loss"]


class TinyResponseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(4, 6)
        self.proj_out = nn.Linear(6, 5)

    def forward(self, features):
        return self.proj_out(torch.tanh(self.proj_in(features)))


def _synthetic_batch():
    features = torch.tensor(
        [
            [[0.2, -0.3, 0.7, 1.1], [0.4, 0.5, -0.6, 0.8], [-1.0, 0.2, 0.3, -0.4], [0.6, -0.8, 0.9, 0.1], [0.3, 0.2, -0.1, 0.7]],
            [[-0.4, 0.9, 0.2, -0.5], [0.7, -0.1, 0.4, 0.6], [0.5, 0.3, -0.7, 0.2], [-0.2, 0.8, 0.1, -0.9], [0.9, 0.4, -0.3, 0.5]],
            [[0.1, 0.6, -0.8, 0.3], [-0.5, 0.2, 0.9, 0.4], [0.8, -0.7, 0.2, 0.1], [0.3, 0.5, -0.4, 0.9], [-0.6, 0.1, 0.7, -0.2]],
            [[0.9, -0.2, 0.5, -0.7], [-0.3, 0.4, 0.6, 0.8], [0.2, -0.9, 0.1, 0.5], [0.7, 0.3, -0.5, 0.4], [-0.1, 0.8, 0.2, -0.6]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [
            [-100, -100, 1, -100, -100],
            [-100, -100, 2, 3, -100],
            [-100, -100, 4, 1, 0],
            [-100, -100, -100, 2, 4],
        ]
    )
    return features, labels


def _per_record_losses(logits, labels):
    shifted_logits, shifted_labels = SHIFT_RESPONSE_TENSORS(logits, labels)
    token_losses = nn.functional.cross_entropy(
        shifted_logits.float().reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shifted_labels.shape)
    target_counts = shifted_labels.ne(-100).sum(dim=1)
    if torch.any(target_counts == 0):
        raise ValueError("synthetic record has no response target")
    return token_losses.sum(dim=1) / target_counts


def _private_objects(mode, state, dtype):
    model = TinyResponseModel().to(dtype=dtype)
    model.load_state_dict(state)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loader = DataLoader(
        TensorDataset(torch.arange(BATCH_SIZE)), batch_size=BATCH_SIZE, shuffle=False
    )
    engine = PrivacyEngine(accountant="prv")
    result = engine.make_private(
        module=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(ignore_index=-100, reduction="mean"),
        data_loader=loader,
        noise_multiplier=0.0,
        max_grad_norm=CLIP_NORM,
        poisson_sampling=False,
        grad_sample_mode=mode,
    )
    return engine, result


def _grad_sample(parameter):
    value = parameter.grad_sample
    return torch.cat(value, dim=0) if isinstance(value, list) else value


def _standard_observation(mode, state, features, labels, dtype):
    engine, result = _private_objects(mode, state, dtype)
    model = result[0]
    logits = model(features.to(dtype=dtype))
    losses = _per_record_losses(logits, labels)
    mean_loss = RESPONSE_ONLY_LOSS(logits, labels)
    mean_loss.backward()
    gradient_samples = {
        name: _grad_sample(parameter).detach().float()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    norms = torch.stack(
        [value.flatten(1).pow(2).sum(dim=1) for value in gradient_samples.values()]
    ).sum(dim=0).sqrt()
    factors = (CLIP_NORM / (norms + 1e-6)).clamp(max=1.0)
    aggregate = {
        name: torch.einsum("b,b...->...", factors, value) / BATCH_SIZE
        for name, value in gradient_samples.items()
    }
    require(engine.accountant.history == [], "non-private equivalence changed the accountant")
    return {
        "losses": losses.detach().float(),
        "mean": mean_loss.detach().float(),
        "norms": norms,
        "factors": factors,
        "aggregate": aggregate,
    }


def _ghost_observation(state, features, labels, dtype):
    engine, result = _private_objects("ghost", state, dtype)
    model, _, criterion, _ = result
    logits = model(features.to(dtype=dtype))
    shifted_logits, shifted_labels = SHIFT_RESPONSE_TENSORS(logits, labels)
    loss = criterion(
        shifted_logits.float().reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        shape=shifted_logits.shape,
    )
    loss.backward()
    norms = model.per_sample_gradient_norms.detach().float()
    factors = (CLIP_NORM / (norms + 1e-6)).clamp(max=1.0)
    aggregate = {
        name: parameter.grad.detach().float() / BATCH_SIZE
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    require(engine.accountant.history == [], "ghost equivalence changed the accountant")
    return {
        "losses": loss.loss_per_sample.detach().float(),
        "mean": loss.detach().float(),
        "norms": norms,
        "factors": factors,
        "aggregate": aggregate,
    }


def _maximum_differences(reference, observation):
    def maximum_absolute(first, second):
        return float((first - second).abs().max())

    def maximum_relative(first, second):
        denominator = torch.maximum(first.abs(), second.abs()).clamp_min(1e-12)
        return float(((first - second).abs() / denominator).max())

    aggregate_absolute = max(
        maximum_absolute(reference["aggregate"][name], observation["aggregate"][name])
        for name in reference["aggregate"]
    )
    aggregate_relative = max(
        maximum_relative(reference["aggregate"][name], observation["aggregate"][name])
        for name in reference["aggregate"]
    )
    return {
        "per_record_loss_max_abs": maximum_absolute(reference["losses"], observation["losses"]),
        "per_record_loss_max_rel": maximum_relative(reference["losses"], observation["losses"]),
        "mean_loss_abs": abs(float(reference["mean"] - observation["mean"])),
        "gradient_norm_max_abs": maximum_absolute(reference["norms"], observation["norms"]),
        "gradient_norm_max_rel": maximum_relative(reference["norms"], observation["norms"]),
        "clipping_factor_max_abs": maximum_absolute(reference["factors"], observation["factors"]),
        "clipping_factor_max_rel": maximum_relative(reference["factors"], observation["factors"]),
        "clipped_aggregate_gradient_max_abs": aggregate_absolute,
        "clipped_aggregate_gradient_max_rel": aggregate_relative,
    }


@pytest.mark.parametrize(
    ("dtype_name", "dtype", "atol", "rtol"),
    (
        ("fp32", torch.float32, 2e-6, 2e-5),
        ("bf16", torch.bfloat16, 2e-2, 2e-2),
    ),
)
def test_supported_modes_match_hooks_response_clipping_contract(
    dtype_name, dtype, atol, rtol, capsys
):
    torch.manual_seed(SEED)
    base = TinyResponseModel().float()
    state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    features, labels = _synthetic_batch()
    reference = _standard_observation("hooks", state, features, labels, dtype)
    observations = {
        "functorch": _standard_observation("functorch", state, features, labels, dtype),
        "ew": _standard_observation("ew", state, features, labels, dtype),
        "ghost": _ghost_observation(state, features, labels, dtype),
    }
    comparisons = {}
    for mode, observation in observations.items():
        torch.testing.assert_close(observation["losses"], reference["losses"], atol=atol, rtol=rtol)
        torch.testing.assert_close(observation["mean"], reference["mean"], atol=atol, rtol=rtol)
        torch.testing.assert_close(observation["norms"], reference["norms"], atol=atol, rtol=rtol)
        torch.testing.assert_close(observation["factors"], reference["factors"], atol=atol, rtol=rtol)
        for name in reference["aggregate"]:
            torch.testing.assert_close(
                observation["aggregate"][name], reference["aggregate"][name], atol=atol, rtol=rtol
            )
        comparisons[mode] = _maximum_differences(reference, observation)

    evidence = {
        "probe": "non-private-zero-noise-synthetic",
        "dtype": dtype_name,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "clip_norm": CLIP_NORM,
        "privacy_accountant_steps": 0,
        "tolerance": {"atol": atol, "rtol": rtol},
        "hooks": {
            "per_record_losses": reference["losses"].tolist(),
            "gradient_norms": reference["norms"].tolist(),
            "clipping_factors": reference["factors"].tolist(),
        },
        "comparisons": comparisons,
    }
    with capsys.disabled():
        print("NUMERICAL_EQUIVALENCE=" + json.dumps(evidence, sort_keys=True, allow_nan=False))
