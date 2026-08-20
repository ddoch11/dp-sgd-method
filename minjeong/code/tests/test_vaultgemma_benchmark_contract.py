import ast
import json
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "workspaces/vaultgemma_vectorized/[VaultGemma]FineTuning_Vectorized.ipynb"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(index: int) -> str:
    source = _notebook()["cells"][index]["source"]
    return source if isinstance(source, str) else "".join(source)


def _extract_function(name: str, globals_dict: dict | None = None):
    source = _source(15)
    tree = ast.parse(source)
    matches = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    require(len(matches) == 1, f"expected one function named {name}")
    namespace = dict(globals_dict or {})
    exec(
        compile(ast.fix_missing_locations(ast.Module(matches, [])), str(NOTEBOOK), "exec"),
        namespace,
    )
    return namespace[name]


def _require_contract(condition, message):
    if not condition:
        raise AssertionError(message)


def test_benchmark_run_kind_is_exactly_fifteen_steps_after_five_warmup_steps():
    source = _source(15)
    for marker in (
        'RUN_KIND in {"full", "smoke", "benchmark"}',
        "BENCHMARK_LOGICAL_STEPS = 15",
        "BENCHMARK_WARMUP_STEPS = 5",
        'if RUN_KIND == "benchmark":',
        'require_contract(MAX_UPDATES == BENCHMARK_LOGICAL_STEPS',
    ):
        require(marker in source, f"missing benchmark run-kind marker: {marker}")


def test_logical_step_timer_spans_all_physical_chunks_and_synchronizes_cuda():
    source = _source(15)
    start = 'if logical_step_started is None:'
    completed = "if logical_step_completed:"
    require(start in source and completed in source, "logical-step timing gates are absent")
    require(source.index(start) < source.index(completed), "timer begins after completion gate")
    timed_region = source[source.index(start) : source.index(completed)]
    require("torch.cuda.synchronize(device)" in timed_region, "CUDA start synchronization missing")
    completed_region = source[source.index(completed) :]
    require("torch.cuda.synchronize(device)" in completed_region, "CUDA end synchronization missing")
    require("logical_step_records.append(" in completed_region, "logical-step record emission missing")


def test_response_tokens_are_counted_on_cpu_before_device_transfer_and_timer():
    source = _source(15)
    count_marker = 'cpu_response_tokens = count_shifted_response_tokens_from_cpu(batch["labels"])'
    timer_marker = "if logical_step_started is None:"
    transfer_marker = 'batch = {k: v.to(device) for k, v in batch.items()}'
    for marker in (count_marker, timer_marker, transfer_marker):
        require(marker in source, f"missing CPU-token timing marker: {marker}")
    require(
        source.index(count_marker) < source.index(timer_marker) < source.index(transfer_marker),
        "response-token count must precede timer and device transfer",
    )
    require(
        'batch["labels"][:, 1:].ne(-100).sum().item()' not in source,
        "GPU sum().item() response-token count remains",
    )
    counter = _extract_function(
        "count_shifted_response_tokens_from_cpu",
        {"torch": torch, "require_contract": _require_contract},
    )
    labels = torch.tensor(
        [[-100, 5, 6, -100], [1, -100, 7, 8]], dtype=torch.long
    )
    result = counter(labels)
    require(type(result) is int and result == 4, f"wrong CPU response-token count: {result!r}")
    with pytest.raises(AssertionError):
        counter([[1, 2]])
    function_source = ast.parse(_source(15))
    helper = next(
        node for node in function_source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "count_shifted_response_tokens_from_cpu"
    )
    require(
        not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "item"
            for node in ast.walk(helper)
        ),
        "CPU token helper must not call Tensor.item()",
    )


def test_steady_state_summary_excludes_five_steps_and_computes_rates():
    summarize = _extract_function(
        "summarize_benchmark_steps",
        {"math": math, "np": __import__("numpy"), "require_contract": _require_contract},
    )
    records = [
        {
            "logical_step": index,
            "latency_seconds": float(index),
            "examples": 128,
            "response_tokens": 100 + index,
        }
        for index in range(1, 16)
    ]
    result = summarize(records, warmup_steps=5, logical_batch_size=128)
    require(result["warmup_steps_excluded"] == 5, "wrong warmup exclusion")
    require(result["steady_step_count"] == 10, "wrong steady step count")
    require(result["step_latency_seconds"] == [float(i) for i in range(6, 16)], "wrong latency window")
    require(result["response_token_counts"] == [100 + i for i in range(6, 16)], "wrong token window")
    steady_seconds = sum(float(i) for i in range(6, 16))
    require(result["examples_per_second"] == pytest.approx(1280 / steady_seconds), "wrong example rate")
    require(
        result["response_tokens_per_second"]
        == pytest.approx(sum(100 + i for i in range(6, 16)) / steady_seconds),
        "wrong token rate",
    )
    require(result["optimizer_step_seconds_p50"] == pytest.approx(10.5), "wrong p50")
    require(result["optimizer_step_seconds_p95"] == pytest.approx(14.55), "wrong p95")


def test_benchmark_summary_rejects_nonfinite_or_malformed_records():
    summarize = _extract_function(
        "summarize_benchmark_steps",
        {"math": math, "np": __import__("numpy"), "require_contract": _require_contract},
    )
    valid = [
        {"logical_step": i, "latency_seconds": 1.0, "examples": 128, "response_tokens": 10}
        for i in range(1, 16)
    ]
    for mutation in (
        lambda rows: rows[7].update(latency_seconds=math.nan),
        lambda rows: rows[7].update(latency_seconds=0.0),
        lambda rows: rows[7].update(examples=127),
        lambda rows: rows[7].update(response_tokens=-1),
        lambda rows: rows[7].update(logical_step=99),
    ):
        records = [dict(row) for row in valid]
        mutation(records)
        with pytest.raises(AssertionError):
            summarize(records, warmup_steps=5, logical_batch_size=128)


def test_metrics_include_train_only_steady_state_memory_and_cpu_rss_fields():
    source = _source(15)
    for marker in (
        '"logical_optimizer_steps": logical_step_records',
        '"steady_state": steady_state_metrics',
        '"notebook_initialization_seconds_from_import_cell"',
        '"train_only_seconds"',
        '"scope": "training_eval_checkpoint_after_reset"',
        '"notebook_process_peak_rss_bytes"',
        'require_finite_metric("optimizer_step_seconds_p50"',
        'require_finite_metric("optimizer_step_seconds_p95"',
        'require_finite_metric("examples_per_second"',
        'require_finite_metric("response_tokens_per_second"',
    ):
        require(marker in source, f"missing metrics marker: {marker}")
    import_source = _source(3)
    require("NOTEBOOK_INITIALIZATION_STARTED = time.perf_counter()" in import_source, "initialization timer missing")
