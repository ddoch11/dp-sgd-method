from __future__ import annotations

import json
from pathlib import Path


PARTICIPANT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PARTICIPANT_ROOT / "results" / "level1_patient_code"
REPORT_PATH = RESULTS_ROOT / "2026-08-21-level1-patient-code-tuned.json"


def load_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_selected_non_dp_recipe_is_a_strong_positive_control() -> None:
    report = load_report()
    assert report["selected_recipe"] == {
        "learning_rate": 0.0001,
        "epochs": 40,
        "logical_steps": 640,
    }
    statistics = report["statistics"]
    assert statistics["non_dp"]["member_exact"] == 488
    assert statistics["non_dp"]["control_exact"] == 0
    assert statistics["non_dp"]["target_score_membership_auc"] == 1.0


def test_tuned_dp_runs_reach_640_steps_with_bounded_epsilon() -> None:
    report = load_report()
    for label, target in (("dp_eps0p5", 0.5), ("dp_eps2", 2.0), ("dp_eps8", 8.0)):
        run = report["training"][label]
        assert run["method"] == "opacus_hooks_dp"
        assert run["accountant"] == "prv"
        assert run["completed_steps"] == run["planned_steps"] == 640
        assert run["target_epsilon"] == target
        assert run["final_epsilon"] <= target


def test_tuned_dp_has_little_detected_memorization_signal() -> None:
    statistics = load_report()["statistics"]
    assert statistics["dp_eps0p5"]["member_exact"] == 0
    assert statistics["dp_eps2"]["member_exact"] == 0
    assert statistics["dp_eps8"]["member_exact"] == 1
    for label in ("dp_eps0p5", "dp_eps2", "dp_eps8"):
        assert statistics[label]["control_exact"] == 0
        assert 0.45 <= statistics[label]["target_score_membership_auc"] <= 0.55


def test_fixed_epsilon_requires_more_noise_for_twice_as_many_steps() -> None:
    report = load_report()
    for label in ("dp_eps0p5", "dp_eps2", "dp_eps8"):
        old_run = report["old_dp_training"][label]
        tuned_run = report["training"][label]
        assert old_run["completed_steps"] == 320
        assert tuned_run["completed_steps"] == 640
        assert tuned_run["noise_multiplier"] > old_run["noise_multiplier"]
