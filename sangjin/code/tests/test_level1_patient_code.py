from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from level1_patient_code_common import (
    CODE_PATTERN,
    canonical_json_sha256,
    pairwise_auc,
    patient_prompt,
    validate_manifest,
)


def build_manifest() -> dict:
    records = []
    for index in range(1000):
        records.append(
            {
                "record_index": index,
                "patient_id": f"SYN-P{index:04d}",
                "private_code": f"{1000 + index:04d}",
                "membership": "member" if index % 2 == 0 else "control",
                "prompt": patient_prompt(f"SYN-P{index:04d}"),
            }
        )
    payload = {
        "schema_version": 1,
        "description": "Fully synthetic patient aliases and private codes; no real PII.",
        "manifest_seed": 1,
        "member_count": 500,
        "control_count": 500,
        "records": records,
        "member_patient_ids_sha256": canonical_json_sha256(
            [row["patient_id"] for row in records if row["membership"] == "member"]
        ),
        "control_patient_ids_sha256": canonical_json_sha256(
            [row["patient_id"] for row in records if row["membership"] == "control"]
        ),
        "private_codes_sha256": canonical_json_sha256(
            sorted(row["private_code"] for row in records)
        ),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def test_manifest_contract() -> None:
    validate_manifest(build_manifest())


def test_prompt_does_not_contain_private_code() -> None:
    manifest = build_manifest()
    for record in manifest["records"]:
        assert record["private_code"] not in record["prompt"]


def test_code_extraction_requires_four_digit_boundary() -> None:
    assert CODE_PATTERN.search("5831").group(1) == "5831"
    assert CODE_PATTERN.search("code: 5831.").group(1) == "5831"
    assert CODE_PATTERN.search("158319") is None
    assert CODE_PATTERN.search("no code") is None


def test_pairwise_auc() -> None:
    assert pairwise_auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert pairwise_auc([0.0, 1.0], [2.0, 3.0]) == 0.0
    assert pairwise_auc([1.0], [1.0]) == 0.5


def test_canonical_hash_is_order_stable_for_dicts() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )
