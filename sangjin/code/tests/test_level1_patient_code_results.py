from __future__ import annotations

import json
from pathlib import Path


PARTICIPANT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PARTICIPANT_ROOT / "results" / "level1_patient_code"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_fully_synthetic_and_balanced() -> None:
    manifest = load_json(RESULTS_ROOT / "level1_patient_codes_manifest.json")
    assert "no real PII" in manifest["description"]
    assert manifest["member_count"] == manifest["control_count"] == 500
    assert len(manifest["records"]) == 1000
    assert len({row["patient_id"] for row in manifest["records"]}) == 1000
    assert len({row["private_code"] for row in manifest["records"]}) == 1000
    assert all(row["private_code"] not in row["prompt"] for row in manifest["records"])


def test_report_contains_base_non_dp_and_three_dp_models() -> None:
    report = load_json(RESULTS_ROOT / "2026-08-21-level1-patient-code.json")
    assert set(report["statistics"]) == {
        "base",
        "non_dp",
        "dp_eps0p5",
        "dp_eps2",
        "dp_eps8",
    }
    assert set(report["training"]) == {
        "non_dp",
        "dp_eps0p5",
        "dp_eps2",
        "dp_eps8",
    }


def test_non_dp_is_positive_control_and_dp_has_no_detected_signal() -> None:
    report = load_json(RESULTS_ROOT / "2026-08-21-level1-patient-code.json")
    statistics = report["statistics"]
    assert statistics["base"]["member_exact"] == 0
    assert statistics["base"]["control_exact"] == 0
    assert statistics["non_dp"]["member_exact"] == 10
    assert statistics["non_dp"]["control_exact"] == 0
    assert statistics["non_dp"]["fisher_two_sided_p"] < 0.01
    assert statistics["non_dp"]["target_score_membership_auc"] > 0.99
    for label in ("dp_eps0p5", "dp_eps2", "dp_eps8"):
        assert statistics[label]["member_exact"] == 0
        assert statistics[label]["control_exact"] == 0
        assert 0.45 <= statistics[label]["target_score_membership_auc"] <= 0.55


def test_dp_runs_use_opacus_and_reach_320_steps() -> None:
    report = load_json(RESULTS_ROOT / "2026-08-21-level1-patient-code.json")
    for label, target in (("dp_eps0p5", 0.5), ("dp_eps2", 2.0), ("dp_eps8", 8.0)):
        run = report["training"][label]
        assert run["method"] == "opacus_hooks_dp"
        assert run["accountant"] == "prv"
        assert run["completed_steps"] == run["planned_steps"] == 320
        assert run["final_epsilon"] <= target
        assert run["target_epsilon"] == target


def test_positive_control_emerges_across_epochs() -> None:
    report = load_json(RESULTS_ROOT / "2026-08-21-level1-patient-code.json")
    pilot = report["pilot_statistics"]
    assert pilot["1"]["target_score_membership_auc"] < pilot["5"]["target_score_membership_auc"]
    assert pilot["5"]["target_score_membership_auc"] < pilot["10"]["target_score_membership_auc"]
    assert pilot["10"]["target_score_membership_auc"] < pilot["20"]["target_score_membership_auc"]
    assert pilot["20"]["member_exact"] == 10
