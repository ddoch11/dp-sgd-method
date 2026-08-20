#!/usr/bin/env python3
"""Generate the deterministic Minjeong release inventory.

The manifest intentionally excludes itself. ``source_path`` values are stable
paths in the private working copy used to curate this release; no absolute local
path is recorded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[2]
DESTINATION = RELEASE_ROOT / "provenance" / "release_manifest.json"
SOURCE = {
    "code/scripts/run_vaultgemma_vectorized.sh": (
        "scripts/run_vaultgemma_vectorized.sh",
        "957d8a127e839b672e71e9a604068cb6d178acf0f5839e3ec4eba5345ca51bd4",
    ),
    "code/scripts/monitor_vaultgemma_gpu.sh": (
        "scripts/monitor_vaultgemma_gpu.sh",
        "06edd6436c83fe1d28c1a317ce2b49bf9b5baa0e6db7150a104f7a5d605b15cc",
    ),
    "code/workspaces/vaultgemma_vectorized/[VaultGemma]FineTuning_Vectorized.ipynb": (
        "workspaces/vaultgemma_vectorized/[VaultGemma]FineTuning_Vectorized.ipynb",
        "95a0a6c5fff03a1344fa278edd300404778c4e22284e0e4ed78892cfadae7067",
    ),
    "code/workspaces/vaultgemma_vectorized/upstream_manifest.json": (
        "workspaces/vaultgemma_vectorized/upstream_manifest.json",
        "440c8a998bfa7f717225400237f52a9b83561e8293f75b97cb66fb6d71c143e2",
    ),
    "code/workspaces/vaultgemma_vectorized/workspace_patch_manifest.json": (
        "workspaces/vaultgemma_vectorized/workspace_patch_manifest.json",
        "2436cbf7014da73bd8d362100bdb2ddcedbdc5595880e3733e56fc89b4decc4b",
    ),
    "provenance/upstream_manifest.json": (
        "workspaces/vaultgemma_vectorized/upstream_manifest.json",
        "440c8a998bfa7f717225400237f52a9b83561e8293f75b97cb66fb6d71c143e2",
    ),
    "provenance/workspace_patch_manifest.json": (
        "workspaces/vaultgemma_vectorized/workspace_patch_manifest.json",
        "2436cbf7014da73bd8d362100bdb2ddcedbdc5595880e3733e56fc89b4decc4b",
    ),
    "provenance/fastdp_upstream_manifest.json": (
        "results/vaultgemma/fastdp_upstream_manifest.json",
        "4a5a94e477dd919c830383960da441b6d35e771dc27ec4e0ae2d9b91f13fc326",
    ),
    "configs/vaultgemma-vectorized.in": (
        "requirements/vaultgemma-vectorized.in",
        "43eb2433cc1448c322b0404746f7fef2ff02a26e752e2d66a1cfb4c9aedcf26d",
    ),
    "configs/vaultgemma-vectorized.lock.txt": (
        "requirements/vaultgemma-vectorized.lock.txt",
        "ff0b1a01ac8158f4d1401588b8029dc8eeca56dbdd8f5fe458802ce78f95e548",
    ),
    "configs/vaultgemma-conda-explicit.lock.txt": (
        "requirements/vaultgemma-conda-explicit.lock.txt",
        "3c1aa905a303a16abc0e223a0d20ae5b255e541fac83d1e5fbd7156cd1bf90b1",
    ),
    "code/tests/test_vaultgemma_loss_contract.py": (
        "tests/test_vaultgemma_loss_contract.py",
        "6fb0ad92b107d516a655156be8c44b2c6a2bb95beedcf330cef9172c26a50ced",
    ),
    "code/tests/test_vaultgemma_dp_contract.py": (
        "tests/test_vaultgemma_dp_contract.py",
        "9eac4a1c0262ed7f85258d5b2e5ce8a2a3ca6a25a0f0258d183436574638cd73",
    ),
    "code/tests/test_vaultgemma_benchmark_contract.py": (
        "tests/test_vaultgemma_benchmark_contract.py",
        "f13cce43b4cb21552c69813f3863c68ce1e5a7e7492c1716a20ff1144a307449",
    ),
    "code/tests/test_vaultgemma_numerical_equivalence.py": (
        "tests/test_vaultgemma_numerical_equivalence.py",
        "530778c2c26e9668a13248a1994ccfff2ab19a72acb30a9414b687121b031570",
    ),
    "results/raw/hooks.json": (
        "results/vaultgemma/runs/task10-hooks-e2-full342-20260819b.json",
        "77d03325150ccf918fed6fedeb6bb1e7aa4ededbc87d29f9e6ee8f94251b58e7",
    ),
    "results/raw/functorch.json": (
        "results/vaultgemma/runs/task10-functorch-e2-full342-20260819a.json",
        "e70e311883f5d0d424d1f24efd454cf68d555757a940d780e62535640066a182",
    ),
    "results/raw/expanded_weights.json": (
        "results/vaultgemma/runs/task10-ew-e2-full342-20260819a.json",
        "72873b5bf97298ea86a9950d609480455dde7da037919ab0db913bda1fbf3bb7",
    ),
    "results/raw/ghost.json": (
        "results/vaultgemma/runs/task10-ghost-e2-full342-20260819a.json",
        "87d2b2c69d3026831a4c59a8ababd978b50f8b2fb5b3337259200492e2609a59",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(RELEASE_ROOT.rglob("*")):
        if not path.is_file() or path == DESTINATION:
            continue
        relative = path.relative_to(RELEASE_ROOT).as_posix()
        if any(part in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"} for part in path.parts):
            continue
        entry = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if relative in SOURCE:
            entry["source_path"], entry["source_sha256"] = SOURCE[relative]
        files.append(entry)
    payload = {
        "schema_version": 1,
        "release_date": "2026-08-20",
        "repository": "https://github.com/ddoch11/dp-sgd-method.git",
        "base_commit": "c9be2491c40ae013c2557d5e9cd50c5833f601d9",
        "participant": "minjeong",
        "experiment_contract": {
            "target_epsilon": 2,
            "target_delta": 1e-5,
            "logical_optimizer_steps": 342,
            "physical_microbatches": 2700,
            "logical_batch_size": 128,
            "physical_batch_size": 16,
            "modes": ["hooks", "functorch", "ew", "ghost"],
        },
        "files": files,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
