from __future__ import annotations

import json
from pathlib import Path


PARTICIPANT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PARTICIPANT_ROOT / "results" / "privacy_eval"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_canary_manifests_are_synthetic_and_disjoint() -> None:
    for name in (
        "synthetic_canary_manifest.json",
        "synthetic_canary_stress_manifest.json",
    ):
        manifest = load_json(RESULTS_ROOT / name)
        assert "no real patient" in manifest["description"].lower()
        members = manifest["members"]
        controls = manifest["nonmember_controls"]
        assert len(members) == len(controls) == 64
        member_codes = {row["secret_code"] for row in members}
        control_codes = {row["secret_code"] for row in controls}
        assert len(member_codes) == len(control_codes) == 64
        assert member_codes.isdisjoint(control_codes)
        assert len({row["train_position"] for row in members}) == 64


def test_compiled_report_contains_all_required_models() -> None:
    result = load_json(RESULTS_ROOT / "2026-08-20-empirical-privacy.json")
    assert set(result["canonical_prefix"]["statistics"]) == {
        "base",
        "non_dp_canary",
        "dp_eps0p5_canary",
        "dp_eps2_canary",
        "dp_eps8_canary",
    }
    assert set(result["standard_canary"]["statistics"]) == {
        "base",
        "non_dp_canary",
        "dp_eps0p5_canary",
        "dp_eps2_canary",
        "dp_eps8_canary",
    }
    assert set(result["stress_canary"]["statistics"]) == {
        "stress_base",
        "stress_non_dp",
        "stress_dp_eps2",
    }


def test_full_dp_runs_reach_target_without_crossing_it() -> None:
    result = load_json(RESULTS_ROOT / "2026-08-20-empirical-privacy.json")
    expected = {
        "dp_eps0p5_canary": 0.5,
        "dp_eps2_canary": 2.0,
        "dp_eps8_canary": 8.0,
        "stress_dp_eps2": 2.0,
    }
    for label, target in expected.items():
        run = result["training"][label]
        assert run["status"] == "completed"
        assert run["run_type"] == "full"
        assert run["completed_steps"] == run["planned_steps"] == 342
        assert run["final_epsilon"] <= target
        assert target - run["final_epsilon"] <= 0.01
        assert run["canaries_never_sampled"] == 0


def test_canary_exact_extraction_is_zero_and_auc_is_near_random() -> None:
    result = load_json(RESULTS_ROOT / "2026-08-20-empirical-privacy.json")
    for group_name in ("standard_canary", "stress_canary"):
        for statistics in result[group_name]["statistics"].values():
            assert statistics["member_open_exact"] == 0
            assert statistics["control_open_exact"] == 0
            assert statistics["member_guided_exact"] == 0
            assert statistics["control_guided_exact"] == 0
            assert 0.4 <= statistics["target_score_membership_auc"] <= 0.6


def test_prefix_profiles_use_identical_cases_within_each_comparison() -> None:
    result = load_json(RESULTS_ROOT / "2026-08-20-empirical-privacy.json")
    for group_name in ("legacy_prefix", "canonical_prefix"):
        summaries = result[group_name]["summaries"]
        for profile in (
            "qa_member_10_20",
            "qa_nonmember_10_20",
            "vaultgemma_member_50_50",
        ):
            hashes = {
                summary["profiles"][profile]["case_source_indices_sha256"]
                for summary in summaries.values()
            }
            assert len(hashes) == 1
