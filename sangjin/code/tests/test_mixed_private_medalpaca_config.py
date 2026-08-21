from __future__ import annotations

import json
from pathlib import Path

import yaml


PARTICIPANT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PARTICIPANT_ROOT / "configs" / "mixed_private_medalpaca_bf16_e30.yaml"
MANIFEST_PATH = (
    PARTICIPANT_ROOT
    / "results"
    / "level1_patient_code"
    / "level1_patient_codes_manifest.json"
)


def test_mixed_training_contract() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    medalpaca_train = int(
        config["dataset"]["num_samples"]
        * config["dataset"]["train_fraction"]
    )
    member_count = sum(
        row["membership"] == "member" for row in manifest["records"]
    )
    control_count = sum(
        row["membership"] == "control" for row in manifest["records"]
    )
    assert medalpaca_train == 7200
    assert member_count == config["dataset"]["synthetic_member_count"] == 500
    assert control_count == 500
    assert medalpaca_train + member_count == 7700
    assert config["dataset"]["mixture_mode"] == "append_private_members"


def test_mixed_training_privacy_and_batch_contract() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    training = config["training"]
    assert training["logical_batch_size"] == 128
    assert training["physical_batch_size"] == 16
    assert training["epochs"] == 30
    assert training["target_delta"] == 1e-5
    assert training["max_grad_norm"] == 1.0
    assert training["accountant"] == "prv"
