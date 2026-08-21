from __future__ import annotations

import json
from pathlib import Path


PARTICIPANT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PARTICIPANT_ROOT / "results" / "level1_patient_code"
REPORT_PATH = RESULTS_ROOT / "2026-08-21-level1-patient-code-methods-eps2.json"
METHODS = {
    "naive_dp",
    "hooks_dp",
    "vmap_dp",
    "expanded_weights_dp",
    "ghost_dp",
    "fastdp_bk",
}


def load_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_report_contains_all_six_dp_backends() -> None:
    report = load_report()
    assert set(report["method_runs"]) == METHODS
    assert set(report["statistics"]) == METHODS


def test_all_methods_share_the_privacy_and_data_contract() -> None:
    report = load_report()
    manifest_hash = report["manifest_payload_sha256"]
    for method, run in report["method_runs"].items():
        assert run["status"] == "completed"
        assert run["run_type"] == "full"
        assert run["fresh_base_model"] is True
        assert run["manifest_payload_sha256"] == manifest_hash
        assert run["member_samples"] == run["control_samples"] == 500
        assert run["completed_steps"] == run["planned_steps"] == 640
        assert run["target_epsilon"] == 2.0
        assert run["target_delta"] == 1e-5
        assert run["sample_rate"] == 0.064
        assert run["noise_multiplier"] == 3.37890625
        assert run["final_epsilon"] <= 2.0
        assert run["sampling_seed"] == 20042
        assert run["method"] == method


def test_manual_backends_share_noise_seed_and_batch_contract() -> None:
    report = load_report()
    for method, run in report["method_runs"].items():
        if method == "fastdp_bk":
            assert run["noise_seed"] is None
        else:
            assert run["noise_seed"] == 10042
        expected_physical_batch = 1 if method == "naive_dp" else 16
        assert run["physical_batch_size"] == expected_physical_batch


def test_backend_choice_does_not_change_memorization_conclusion() -> None:
    report = load_report()
    for method in METHODS:
        statistics = report["statistics"][method]
        assert statistics["control_exact"] == 0
        assert statistics["member_exact"] <= 1
        assert 0.45 <= statistics["target_score_membership_auc"] <= 0.55


def test_non_dp_reference_is_a_strong_positive_control() -> None:
    reference = load_report()["non_dp_reference"]["statistics"]
    assert reference["member_exact"] == 488
    assert reference["control_exact"] == 0
    assert reference["target_score_membership_auc"] == 1.0
