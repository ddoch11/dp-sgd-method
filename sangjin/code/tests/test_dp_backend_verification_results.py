from __future__ import annotations

import json
from pathlib import Path


PARTICIPANT_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = (
    PARTICIPANT_ROOT
    / "results"
    / "mixed_private_medalpaca"
    / "dp_backend_equivalence.json"
)


def test_dp_backend_numerical_verification_passed() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["passed"] is True
    for metrics in result["toy_model"]["comparisons_vs_naive"].values():
        assert metrics["relative_l2"] < 1e-4
        assert metrics["max_abs"] < 1e-4


def test_full_run_adapters_remain_numerically_equivalent() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    for metrics in result["full_mixed_run"].values():
        assert metrics["completed_steps"] == 1830
        assert metrics["total_examples_processed"] == 234159
        assert metrics["relative_adapter_l2_vs_naive"] < 0.002
        assert metrics["adapter_cosine_vs_naive"] > 0.99999


def test_actual_model_and_accounting_validation_passed() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    model = result["actual_model_validation"]
    assert model["trainable_parameters"] == 6842368
    assert model["trainable_parameter_tensors"] == 364
    assert model["module_validator_errors"] == []
    assert model["unsupported_hooks"] == []
    assert model["unsupported_ghost"] == []
    assert model["unsupported_fastdp"] == []
    accounting = result["privacy_accounting"]
    assert accounting["mechanism"] == "prv"
    assert accounting["max_reported_epsilon_error"] < 1e-10
    assert 1.99 < accounting["recomputed_epsilon"] < 2.0
