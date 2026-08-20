from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PARTICIPANT_ROOT = ROOT / "minjeong"
EXPECTED = {
    "hooks": {
        "filename": "hooks.json",
        "run_id": "task10-hooks-e2-full342-20260819b",
        "eval_loss": 1.5703986835479737,
        "perplexity": 4.808564907386637,
        "run_seconds": 1488.9457994205877,
        "train_only_seconds": 1470.037656256929,
        "pytorch_peak_allocated_bytes": 27369138688,
        "external_peak_vram_mib": 31271,
    },
    "functorch": {
        "filename": "functorch.json",
        "run_id": "task10-functorch-e2-full342-20260819a",
        "eval_loss": 1.5713594198226928,
        "perplexity": 4.81318689002018,
        "run_seconds": 2124.6380321402103,
        "train_only_seconds": 2106.0207300027832,
        "pytorch_peak_allocated_bytes": 27369138688,
        "external_peak_vram_mib": 31271,
    },
    "ew": {
        "filename": "expanded_weights.json",
        "run_id": "task10-ew-e2-full342-20260819a",
        "eval_loss": 1.5698220992088319,
        "perplexity": 4.805793163316191,
        "run_seconds": 1310.7005986003205,
        "train_only_seconds": 1290.1501331962645,
        "pytorch_peak_allocated_bytes": 33252861952,
        "external_peak_vram_mib": 35287,
    },
    "ghost": {
        "filename": "ghost.json",
        "run_id": "task10-ghost-e2-full342-20260819a",
        "eval_loss": 1.5709420013427735,
        "perplexity": 4.8111781961271225,
        "run_seconds": 2223.7828505877405,
        "train_only_seconds": 2205.0103290285915,
        "pytorch_peak_allocated_bytes": 27369362944,
        "external_peak_vram_mib": 31117,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def released_files(*, exclude_manifest: bool = False) -> set[str]:
    manifest = PARTICIPANT_ROOT / "provenance" / "release_manifest.json"
    return {
        str(path.relative_to(PARTICIPANT_ROOT))
        for path in PARTICIPANT_ROOT.rglob("*")
        if path.is_file()
        and not (exclude_manifest and path == manifest)
        and not any(
            part in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
            for part in path.parts
        )
    }


def test_release_has_only_the_approved_topology() -> None:
    required = {
        "README.md",
        "code/task_plan.md",
        "code/workspaces/vaultgemma_vectorized/[VaultGemma]FineTuning_Vectorized.ipynb",
        "code/workspaces/vaultgemma_vectorized/upstream_manifest.json",
        "code/workspaces/vaultgemma_vectorized/workspace_patch_manifest.json",
        "code/scripts/run_vaultgemma_vectorized.sh",
        "code/scripts/monitor_vaultgemma_gpu.sh",
        "code/tests/test_release_contract.py",
        "configs/storage.env.example",
        "configs/vaultgemma-vectorized.in",
        "configs/vaultgemma-vectorized.lock.txt",
        "configs/vaultgemma-conda-explicit.lock.txt",
        "provenance/upstream_manifest.json",
        "provenance/fastdp_upstream_manifest.json",
        "provenance/workspace_patch_manifest.json",
        "provenance/release_manifest.json",
        "results/README.md",
        "results/epsilon2_full342_summary.csv",
        "results/epsilon2_full342_summary.md",
        "results/raw/hooks.json",
        "results/raw/functorch.json",
        "results/raw/expanded_weights.json",
        "results/raw/ghost.json",
    }
    actual = released_files()
    require(required <= actual, f"release files missing: {sorted(required - actual)}")
    forbidden_suffixes = {".log", ".safetensors", ".bin", ".pt", ".pth", ".csv.tmp"}
    forbidden_names = {"storage.env", ".env", "token", "executed.ipynb"}
    offenders = [
        relative
        for relative in actual
        if Path(relative).suffix in forbidden_suffixes
        or Path(relative).name in forbidden_names
    ]
    require(not offenders, f"forbidden release artifacts: {sorted(offenders)}")


def test_final_raw_results_are_exact_epsilon2_full342_runs() -> None:
    dataset_hashes = set()
    notebook_hashes = set()
    for mode, expected in EXPECTED.items():
        path = PARTICIPANT_ROOT / "results" / "raw" / expected["filename"]
        payload = load_json(path)
        require(payload["status"] == "success", f"{mode}: status is not success")
        require(payload["requested_mode"] == mode, f"{mode}: requested mode mismatch")
        require(payload["actual_mode"] == mode, f"{mode}: actual mode mismatch")
        require(payload["run_id"] == expected["run_id"], f"{mode}: run id mismatch")
        config = payload["configuration"]
        require(config["target_epsilon"] == 2, f"{mode}: target epsilon mismatch")
        require(config["target_delta"] == 1e-5, f"{mode}: target delta mismatch")
        require(config["seed"] == 42, f"{mode}: seed mismatch")
        require(config["logical_batch_size"] == 128, f"{mode}: logical batch mismatch")
        require(config["max_physical_batch_size"] == 16, f"{mode}: physical batch mismatch")
        require(config["run_kind"] == "full", f"{mode}: run kind mismatch")
        require(payload["logical_steps"] == 342, f"{mode}: logical steps mismatch")
        require(payload["physical_steps"] == 2700, f"{mode}: physical steps mismatch")
        require(len(payload["logical_optimizer_steps"]) == 342, f"{mode}: timing record count mismatch")
        require(payload["noise_multiplier"] == 1.015625, f"{mode}: sigma mismatch")
        require(payload["final_epsilon"] <= 2, f"{mode}: privacy target exceeded")
        require(
            math.isclose(payload["eval_loss"], expected["eval_loss"], rel_tol=0, abs_tol=1e-12),
            f"{mode}: eval loss mismatch",
        )
        require(
            math.isclose(payload["perplexity"], expected["perplexity"], rel_tol=0, abs_tol=1e-12),
            f"{mode}: perplexity mismatch",
        )
        dataset_hashes.add(payload["dataset_hash"])
        notebook_hashes.add(payload["executed_input_sha256"])
    require(len(dataset_hashes) == 1, "all methods must use the same selected dataset")
    require(len(notebook_hashes) == 1, "all methods must use the same notebook input")


def test_summary_is_derived_from_raw_results() -> None:
    summary_path = PARTICIPANT_ROOT / "results" / "epsilon2_full342_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    require(set(rows) == set(EXPECTED), f"summary modes mismatch: {sorted(rows)}")
    for mode, expected in EXPECTED.items():
        raw = load_json(PARTICIPANT_ROOT / "results" / "raw" / expected["filename"])
        row = rows[mode]
        expected_throughput = 43200 / expected["train_only_seconds"]
        exact = {
            "run_id": expected["run_id"],
            "actual_epsilon": raw["final_epsilon"],
            "eval_loss": expected["eval_loss"],
            "perplexity": expected["perplexity"],
            "run_seconds": expected["run_seconds"],
            "train_only_seconds": expected["train_only_seconds"],
            "examples_per_second": expected_throughput,
            "pytorch_peak_allocated_bytes": expected["pytorch_peak_allocated_bytes"],
            "external_peak_vram_mib": expected["external_peak_vram_mib"],
        }
        require(row["run_id"] == exact.pop("run_id"), f"{mode}: summary run id mismatch")
        for key, value in exact.items():
            require(
                math.isclose(float(row[key]), float(value), rel_tol=0, abs_tol=1e-9),
                f"{mode}: summary {key} mismatch",
            )


def test_release_configuration_has_no_machine_specific_executable_paths() -> None:
    checked = [
        PARTICIPANT_ROOT / "README.md",
        PARTICIPANT_ROOT / "code" / "scripts" / "run_vaultgemma_vectorized.sh",
        PARTICIPANT_ROOT / "code" / "scripts" / "monitor_vaultgemma_gpu.sh",
        PARTICIPANT_ROOT / "configs" / "storage.env.example",
    ]
    forbidden_paths = ("/home/" + "minjeong", "/raid/" + "minjeong")
    credential_patterns = (
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret)\s*=\s*[^$\s][^\s]*"),
    )
    for path in checked:
        text = path.read_text(encoding="utf-8").lower()
        matches = [pattern for pattern in forbidden_paths if pattern in text]
        matches.extend(pattern.pattern for pattern in credential_patterns if pattern.search(text))
        require(not matches, f"{path}: forbidden machine/credential text {matches}")


def test_release_manifest_is_complete_and_self_consistent() -> None:
    manifest_path = PARTICIPANT_ROOT / "provenance" / "release_manifest.json"
    manifest = load_json(manifest_path)
    require(manifest["schema_version"] == 1, "release manifest schema mismatch")
    entries = manifest["files"]
    require(isinstance(entries, list) and entries, "release manifest must list files")
    by_path = {entry["path"]: entry for entry in entries}
    require(len(by_path) == len(entries), "release manifest contains duplicate paths")
    actual = released_files(exclude_manifest=True)
    require(set(by_path) == actual, "release manifest file inventory mismatch")
    for relative, entry in by_path.items():
        path = PARTICIPANT_ROOT / relative
        require(entry["sha256"] == sha256(path), f"manifest hash mismatch: {relative}")
        require(entry["bytes"] == path.stat().st_size, f"manifest size mismatch: {relative}")
        source_sha = entry.get("source_sha256")
        if source_sha is not None:
            require(
                isinstance(source_sha, str) and len(source_sha) == 64,
                f"invalid source hash: {relative}",
            )
