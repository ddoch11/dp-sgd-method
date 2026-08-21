#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from level1_patient_code_common import (  # noqa: E402
    canonical_json_sha256,
    deep_get,
    load_yaml,
    patient_prompt,
    validate_manifest,
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    member_count = int(deep_get(cfg, "dataset.member_count", 500))
    control_count = int(deep_get(cfg, "dataset.control_count", 500))
    total = member_count + control_count
    if total > 9000:
        raise ValueError("Four-digit code space cannot provide enough unique values")
    seed = int(deep_get(cfg, "runtime.manifest_seed", 20260821))
    rng = random.Random(seed)
    codes = [f"{value:04d}" for value in rng.sample(range(1000, 10000), total)]
    member_indices = set(rng.sample(range(total), member_count))
    records = []
    for index in range(total):
        patient_id = f"SYN-P{index:04d}"
        record = {
            "record_index": index,
            "patient_id": patient_id,
            "private_code": codes[index],
            "membership": "member" if index in member_indices else "control",
            "prompt": patient_prompt(patient_id),
        }
        records.append(record)
    payload = {
        "schema_version": 1,
        "description": "Fully synthetic patient aliases and private codes; no real PII.",
        "manifest_seed": seed,
        "member_count": member_count,
        "control_count": control_count,
        "records": records,
        "member_patient_ids_sha256": canonical_json_sha256(
            sorted(row["patient_id"] for row in records if row["membership"] == "member")
        ),
        "control_patient_ids_sha256": canonical_json_sha256(
            sorted(row["patient_id"] for row in records if row["membership"] == "control")
        ),
        "private_codes_sha256": canonical_json_sha256(sorted(codes)),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    validate_manifest(payload)
    write_json_exclusive(args.output, payload)
    print(
        f"created={args.output} member={member_count} control={control_count} "
        f"sha256={payload['payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
