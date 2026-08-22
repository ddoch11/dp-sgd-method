from __future__ import annotations

import csv
import json
from pathlib import Path


PARTICIPANT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PARTICIPANT_ROOT / "results" / "mixed_private_medalpaca"
REPORT_PATH = RESULTS_ROOT / "2026-08-21-mixed-private-medalpaca-e30.json"
OUTPUT_CSV_PATH = (
    RESULTS_ROOT / "2026-08-21-mixed-private-medalpaca-e30-outputs.csv"
)
METHODS = {
    "non_dp",
    "naive_dp",
    "hooks_dp",
    "vmap_dp",
    "expanded_weights_dp",
    "ghost_dp",
    "fastdp_bk",
}
DP_METHODS = METHODS - {"non_dp"}


def load_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_mixed_report_contains_only_canonical_methods() -> None:
    report = load_report()
    assert report["report_variant"] == "mixed_private_medalpaca_e30"
    assert set(report["training"]) == METHODS
    assert set(report["statistics"]) == {"base", *METHODS}


def test_training_uses_medalpaca_plus_private_members() -> None:
    report = load_report()
    for method, run in report["training"].items():
        assert run["method"] == method
        assert run["run_type"] == "full"
        assert run["train_samples"] == 7700
        assert run["eval_samples"] == 800
        assert run["synthetic_member_samples"] == 500
        assert run["epochs"] == 30
        assert run["completed_steps"] == run["planned_steps"] == 1830
        assert run["logical_batch_size"] == 128
        expected_physical_batch = 1 if method == "naive_dp" else 16
        assert run["physical_batch_size"] == expected_physical_batch
        assert abs(run["sample_rate"] - 128 / 7700) < 1e-12
    assert report["training"]["non_dp"]["target_epsilon"] is None


def test_dp_runs_reach_target_epsilon_with_shared_noise() -> None:
    report = load_report()
    for method in DP_METHODS:
        run = report["training"][method]
        assert run["target_epsilon"] == 2.0
        assert run["target_delta"] == 1e-5
        assert run["max_grad_norm"] == 1.0
        assert run["noise_multiplier"] == 1.62109375
        assert 1.99 < run["final_epsilon"] <= 2.0


def test_non_dp_leaks_members_but_dp_does_not() -> None:
    report = load_report()
    statistics = report["statistics"]
    assert statistics["base"]["member_exact"] == 0
    assert statistics["non_dp"]["member_exact"] == 295
    assert statistics["non_dp"]["control_exact"] == 0
    assert statistics["non_dp"]["target_score_membership_auc"] > 0.999
    for method in DP_METHODS:
        assert statistics[method]["member_exact"] == 0
        assert statistics[method]["control_exact"] == 0
        assert 0.45 <= statistics[method]["target_score_membership_auc"] <= 0.55


def test_dp_preserves_medalpaca_utility_better_than_non_dp() -> None:
    report = load_report()
    utility = report["utility"]
    base_loss = utility["base"]["metrics"]["example_mean_loss"]
    non_dp_loss = utility["non_dp"]["example_mean_loss"]
    assert 1.63 < base_loss < 1.65
    assert 2.15 < non_dp_loss < 2.17
    for method in DP_METHODS:
        dp_loss = utility[method]["example_mean_loss"]
        assert 1.20 < dp_loss < 1.21
        assert dp_loss < base_loss < non_dp_loss


def test_full_output_csv_contains_all_synthetic_records() -> None:
    with OUTPUT_CSV_PATH.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1000
    assert len({row["patient_id"] for row in rows}) == 1000
    assert sum(row["membership"] == "member" for row in rows) == 500
    assert sum(row["membership"] == "control" for row in rows) == 500
    assert sum(row["non_dp_exact"] == "True" for row in rows) == 295
    for method in DP_METHODS:
        assert sum(row[f"{method}_exact"] == "True" for row in rows) == 0


def test_vectorized_and_gradient_free_backends_are_faster_than_naive() -> None:
    training = load_report()["training"]
    naive = training["naive_dp"]
    assert naive["elapsed_training_sec"] / 60 > 500
    assert naive["peak_vram_gb"] < 4
    for method in DP_METHODS - {"naive_dp"}:
        run = training[method]
        assert run["elapsed_training_sec"] < naive["elapsed_training_sec"]
        assert run["throughput_samples_per_sec"] > naive[
            "throughput_samples_per_sec"
        ]
