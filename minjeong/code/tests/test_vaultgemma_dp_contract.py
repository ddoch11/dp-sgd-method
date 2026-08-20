import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.grad_sample import (
    GradSampleModule,
    GradSampleModuleExpandedWeights,
    GradSampleModuleFastGradientClipping,
)
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


pytestmark = pytest.mark.filterwarnings("ignore:Secure RNG turned off.*:UserWarning")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(
    os.environ.get(
        "VAULTGEMMA_DP_NOTEBOOK_OVERRIDE",
        ROOT / "workspaces/vaultgemma_vectorized/[VaultGemma]FineTuning_Vectorized.ipynb",
    )
)
PATCH_MANIFEST = ROOT / "workspaces/vaultgemma_vectorized/workspace_patch_manifest.json"
Q = 128 / 7200
STEPS = 342
EXPECTED_SIGMAS = {0.5: 2.578125, 2.0: 1.015625, 8.0: 0.6005859375}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _notebook_code() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    return notebook, code


def _assignment_values(code: str) -> dict[str, object]:
    sanitized = "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith(("!", "%"))
    )
    tree = ast.parse(sanitized)
    values = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


def _extract_function(name: str, globals_dict: dict | None = None):
    _, code = _notebook_code()
    sanitized = "\n".join(
        line for line in code.splitlines() if not line.lstrip().startswith(("!", "%"))
    )
    tree = ast.parse(sanitized)
    matches = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    require(len(matches) == 1, f"expected one notebook function named {name}")
    namespace = dict(globals_dict or {})
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[])), str(NOTEBOOK), "exec"),
        namespace,
    )
    return namespace[name]


def test_direct_opacus_prv_sigma_sweep_is_exact():
    for epsilon, expected_sigma in EXPECTED_SIGMAS.items():
        actual = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=1e-5,
            sample_rate=Q,
            steps=STEPS,
            accountant="prv",
        )
        require(actual == expected_sigma, f"epsilon {epsilon}: {actual} != {expected_sigma}")


def test_notebook_uses_explicit_prv_calibration_and_strict_epsilon_selection():
    _, code = _notebook_code()
    values = _assignment_values(code)
    for name, expected in {
        "TARGET_DELTA": 1e-5,
        "MAX_GRAD_NORM": 1.0,
        "TOTAL_OPTIMIZER_STEPS": 342,
        "LOGICAL_BATCH_SIZE": 128,
        "TRAIN_SIZE": 7200,
    }.items():
        require(values.get(name) == expected, f"{name} contract mismatch: {values.get(name)!r}")
    for required in (
        "TARGET_EPSILON_ENV = os.environ.get(\"VAULTGEMMA_TARGET_EPSILON\")",
        "ALLOWED_TARGET_EPSILONS = {0.5, 2.0, 8.0}",
        "Q = LOGICAL_BATCH_SIZE / TRAIN_SIZE",
        "get_noise_multiplier(",
        "sample_rate=Q",
        "steps=TOTAL_OPTIMIZER_STEPS",
        'accountant="prv"',
        'PrivacyEngine(accountant="prv")',
    ):
        require(required in code, f"missing exact DP configuration: {required}")
    require("make_private_with_epsilon" not in code, "implicit epsilon calibration remains")


def test_target_epsilon_parser_accepts_only_the_three_sweep_values():
    parse = _extract_function(
        "parse_target_epsilon", {"ALLOWED_TARGET_EPSILONS": {0.5, 2.0, 8.0}}
    )
    for raw, expected in (("0.5", 0.5), ("2", 2.0), ("8.0", 8.0)):
        require(parse(raw) == expected, f"valid epsilon {raw!r} was not canonicalized")
    for invalid in (None, "", "1", "nan", object()):
        with pytest.raises(ValueError):
            parse(invalid)


def test_validation_precedes_optimizer_and_no_module_fix_or_post_wrap_mutation():
    _, code = _notebook_code()
    validation = "ModuleValidator.validate(module, strict=True)"
    optimizer = "base_optimizer = optimizer_factory(module)"
    wrapping = "private_objects = privacy_engine.make_private("
    require(validation in code, "strict pre-wrap validation is missing")
    require(optimizer in code, "AdamW construction is missing")
    require(wrapping in code, "make_private is missing")
    require(code.index(validation) < code.index(optimizer) < code.index(wrapping), "validation/optimizer/wrapping order changed")
    require("ModuleValidator.fix" not in code, "module-fixing fallback is forbidden")
    suffix = code[code.index(wrapping) :]
    require("get_peft_model(" not in suffix, "module graph is mutated after optimizer creation")


def test_make_private_and_optimizer_accounting_contract_is_explicit():
    _, code = _notebook_code()
    for required in (
        "criterion=criterion",
        "noise_multiplier=noise_multiplier",
        "max_grad_norm=max_grad_norm",
        "poisson_sampling=False",
        "noise_generator=noise_generator",
        "grad_sample_mode=mode",
        "requested_mode=GRAD_SAMPLE_MODE",
        "private_objects[1].expected_batch_size = expected_batch_size",
        "private_objects[1].attach_step_hook(",
        "optimizer.expected_batch_size = LOGICAL_BATCH_SIZE",
        "privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)",
        "GHOST_LOSS_CRITERION = private_objects[2]",
    ):
        require(required in code, f"missing private-object contract: {required}")
    hook = "private_objects[1].attach_step_hook("
    publish = "parent_descriptor, destination.name, serialized.encode(\"utf-8\")"
    require(code.index(hook) < code.index(publish), "accountant hook is installed after success publication")
    require("optimizer.attach_step_hook(" not in code, "post-publication accountant hook remains")
    require(
        "publish_bytes_no_replace(destination, serialized.encode" not in code,
        "path-based compatibility success publication remains",
    )
    require(
        "publish_bytes_no_replace(destination, failure_serialized.encode" not in code,
        "path-based compatibility failure publication remains",
    )
    require("expected_batch_size == LOGICAL_BATCH_SIZE" in code, "normalization denominator assertion missing")
    require("len(private_objects) == 4" in code, "ghost four-value result is not validated")
    require("len(private_objects) == 3" in code, "non-ghost three-value result is not validated")


class _TinySavableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)
        self.saved_paths = []

    def forward(self, features):
        return self.projection(features)

    def save_pretrained(self, path):
        self.saved_paths.append(path)


class _TinySavableTokenizer:
    def __init__(self):
        self.saved_paths = []

    def save_pretrained(self, path):
        self.saved_paths.append(path)


def _make_real_private_module(mode: str):
    model = _TinySavableModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
        shuffle=False,
    )
    make_private_kwargs = {
        "module": model,
        "optimizer": optimizer,
        "data_loader": loader,
        "noise_multiplier": 0.0,
        "max_grad_norm": 1.0,
        "poisson_sampling": False,
        "grad_sample_mode": mode,
    }
    if mode == "ghost":
        make_private_kwargs["criterion"] = nn.CrossEntropyLoss()
    wrapped = PrivacyEngine(accountant="prv").make_private(**make_private_kwargs)[0]
    require(isinstance(wrapped, AbstractGradSampleModule), f"{mode} wrapper type changed")
    require(not hasattr(wrapped, "save_pretrained"), f"{mode} unexpectedly forwards save_pretrained")
    return wrapped, model


def test_private_adapter_save_unwraps_each_real_current_opacus_mode(tmp_path):
    save_checkpoint = _extract_function(
        "save_private_adapter_checkpoint",
        {
            "AbstractGradSampleModule": AbstractGradSampleModule,
            "GradSampleModule": GradSampleModule,
            "GradSampleModuleExpandedWeights": GradSampleModuleExpandedWeights,
            "GradSampleModuleFastGradientClipping": GradSampleModuleFastGradientClipping,
        },
    )
    for mode in ("hooks", "functorch", "ew", "ghost"):
        wrapped, inner = _make_real_private_module(mode)
        tokenizer = _TinySavableTokenizer()
        checkpoint = tmp_path / mode
        save_checkpoint(wrapped, tokenizer, checkpoint)
        require(inner.saved_paths == [checkpoint], f"{mode} did not call the inner model save")
        require(tokenizer.saved_paths == [checkpoint], f"{mode} did not save the tokenizer")


def test_private_adapter_save_rejects_unwrapped_wrong_nested_and_malformed_objects(tmp_path):
    save_checkpoint = _extract_function(
        "save_private_adapter_checkpoint",
        {
            "AbstractGradSampleModule": AbstractGradSampleModule,
            "GradSampleModule": GradSampleModule,
            "GradSampleModuleExpandedWeights": GradSampleModuleExpandedWeights,
            "GradSampleModuleFastGradientClipping": GradSampleModuleFastGradientClipping,
        },
    )
    tokenizer = _TinySavableTokenizer()
    with pytest.raises(TypeError, match="Opacus private-module wrapper"):
        save_checkpoint(_TinySavableModel(), tokenizer, tmp_path / "unwrapped")
    with pytest.raises(TypeError, match="Opacus private-module wrapper"):
        save_checkpoint(object(), tokenizer, tmp_path / "wrong")

    nested, _ = _make_real_private_module("hooks")
    nested_inner, _ = _make_real_private_module("ew")
    nested._modules["_module"] = nested_inner
    with pytest.raises(TypeError, match="nested"):
        save_checkpoint(nested, tokenizer, tmp_path / "nested")

    malformed, _ = _make_real_private_module("hooks")
    malformed.extra = nn.Linear(1, 1)
    with pytest.raises(TypeError, match="exactly one inner"):
        save_checkpoint(malformed, tokenizer, tmp_path / "malformed")


def test_final_runner_checkpoint_uses_strict_private_save_helper_only():
    _, code = _notebook_code()
    checkpoint = code[code.index("checkpoint_started = time.perf_counter()") :]
    require(
        "save_private_adapter_checkpoint(peft_model, tokenizer, ADAPTER_PATH)" in checkpoint,
        "final path does not unwrap and save through the strict helper",
    )
    require('checkpoint_path = "./final_model"' not in code, "project-local checkpoint remains")
    require("peft_model.save_pretrained(" not in checkpoint, "final path still calls wrapper save")
    require("try:" not in checkpoint, "final save has a silent fallback")


def test_dp_noise_generator_is_dedicated_device_local_and_cpu_reproducible():
    _, code = _notebook_code()
    require("DP_NOISE_GENERATOR = build_dp_noise_generator(DEVICE, SEED)" in code, "dedicated DP generator missing")
    require(code.count("DP_NOISE_GENERATOR") == 2, "DP generator is reused outside construction/make_private")
    require("torch.Generator(device=device)" in code, "DP generator is not device-local")
    require("noise_generator=DP_NOISE_GENERATOR" in code, "DP generator is not passed to make_private")
    build = _extract_function("build_dp_noise_generator", {"torch": torch})
    first = build(torch.device("cpu"), 42)
    second = build(torch.device("cpu"), 42)
    require(first.initial_seed() == 42 == second.initial_seed(), "DP generator seed is not auditable")
    first_sample = torch.randn(8, generator=first)
    second_sample = torch.randn(8, generator=second)
    require(torch.equal(first_sample, second_sample), "same DP seed does not reproduce CPU noise")


def test_remainder_and_physical_chunk_math_is_exact():
    full_batches, remainder = divmod(7200, 128)
    logical_per_epoch = full_batches + int(remainder > 0)
    physical_per_epoch = full_batches * math.ceil(128 / 16) + math.ceil(remainder / 16)
    require((full_batches, remainder) == (56, 32), "logical remainder contract changed")
    require(logical_per_epoch == 57, "logical batches per epoch changed")
    require(physical_per_epoch == 450, "physical batches per epoch changed")
    require(logical_per_epoch * 6 == 342, "total logical steps changed")
    require(physical_per_epoch * 6 == 2700, "total physical chunks changed")


def test_real_opacus_bmm_executes_two_logical_steps_with_exact_hook_rate():
    torch.manual_seed(7)
    features = torch.randn(160, 3)
    targets = torch.randn(160, 1)
    loader = DataLoader(TensorDataset(features, targets), batch_size=128, shuffle=False, drop_last=False)
    model = nn.Linear(3, 1)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    engine = PrivacyEngine(accountant="prv")
    model, optimizer, loader = engine.make_private(
        module=model,
        optimizer=base_optimizer,
        data_loader=loader,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
        grad_sample_mode="hooks",
    )
    optimizer.expected_batch_size = 128
    optimizer.attach_step_hook(engine.accountant.get_optimizer_hook_fn(sample_rate=Q))

    physical_sizes = []
    skipped = []
    scheduler_steps = 0
    with BatchMemoryManager(data_loader=loader, max_physical_batch_size=16, optimizer=optimizer) as memory_loader:
        for x, y in memory_loader:
            physical_sizes.append(len(x))
            loss = nn.functional.mse_loss(model(x), y, reduction="mean")
            loss.backward()
            optimizer.step()
            completed = not bool(optimizer._is_last_step_skipped)
            skipped.append(not completed)
            optimizer.zero_grad()
            if completed:
                scheduler_steps += 1

    require(physical_sizes == [16] * 10, f"unexpected physical chunks: {physical_sizes}")
    require(skipped == [True] * 7 + [False] + [True] + [False], f"skip pattern changed: {skipped}")
    require(scheduler_steps == 2, f"scheduler advanced {scheduler_steps} times")
    require(optimizer.expected_batch_size == 128, "DP normalization denominator changed")
    require(engine.accountant.history == [(0.0, Q, 2)], f"accountant hook/history changed: {engine.accountant.history}")


def test_training_uses_bmm_and_asserts_completed_contract():
    _, code = _notebook_code()
    for required in (
        "with BatchMemoryManager(",
        "data_loader=train_dataloader",
        "max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE",
        "optimizer=optimizer",
        "for step, batch in enumerate(memory_safe_data_loader)",
        "physical_microbatch_count += 1",
        "scheduler_step_count += 1",
        "completed_epoch_count += 1",
        "accountant_logical_steps(privacy_engine.accountant) == RUN_TARGET_LOGICAL_STEPS",
        "validate_accountant_history(",
        "global_step == RUN_TARGET_LOGICAL_STEPS",
        "scheduler_step_count == RUN_TARGET_LOGICAL_STEPS",
        "lr_scheduler.last_epoch == RUN_TARGET_LOGICAL_STEPS",
        'if RUN_KIND == "full":',
        "completed_epoch_count == NUM_TRAIN_EPOCHS",
        "physical_microbatch_count == EXPECTED_TOTAL_PHYSICAL_MICROBATCHES",
    ):
        require(required in code, f"missing BMM/final invariant: {required}")
    scheduler_block = code[code.index("if logical_step_completed:") :]
    require(scheduler_block.index("lr_scheduler.step()") < scheduler_block.index("global_step += 1"), "scheduler/global-step gate changed")


def test_accountant_history_validator_rejects_wrong_sigma_rate_and_step_sum():
    validate_history = _extract_function("validate_accountant_history")

    class _Accountant:
        def __init__(self, history):
            self.history = history

    validate_history(_Accountant([(1.0, Q, 300), (1.0, Q, 42)]), 1.0, Q, 342)
    for history, message in (
        ([(0.9, Q, 342)], "noise multiplier"),
        ([(1.0, 1 / 57, 342)], "sample rate"),
        ([(1.0, Q, 341)], "step count"),
        ([], "must not be empty"),
    ):
        with pytest.raises(AssertionError, match=message):
            validate_history(_Accountant(history), 1.0, Q, 342)


def test_calibrated_prv_epsilon_is_below_target_within_point_zero_one():
    from opacus.accountants import PRVAccountant

    for target, sigma in EXPECTED_SIGMAS.items():
        accountant = PRVAccountant()
        accountant.history = [(sigma, Q, STEPS)]
        epsilon = accountant.get_epsilon(delta=1e-5)
        require(epsilon <= target, f"calibrated epsilon exceeds target {target}: {epsilon}")
        require(target - epsilon <= 0.01, f"calibration gap exceeds 0.01: {target - epsilon}")

    _, code = _notebook_code()
    require("EPSILON_CALIBRATION_TOLERANCE = 0.01" in code, "epsilon tolerance constant missing")
    require("epsilon <= TARGET_EPSILON" in code, "epsilon upper-bound invariant missing")
    require("TARGET_EPSILON - epsilon <= EPSILON_CALIBRATION_TOLERANCE" in code, "epsilon calibration tolerance invariant missing")


def test_final_epsilon_is_labeled_as_fixed_size_non_poisson_prv_approximation():
    _, code = _notebook_code()
    required_label = "Opacus PRV approximation for fixed-size non-Poisson sampling"
    require(required_label in code, "final epsilon approximation label is missing")
    require("Final privacy cost" not in code, "final epsilon is incorrectly labeled as a privacy cost")


def test_task5_privacy_and_training_cells_have_no_stale_execution_outputs():
    notebook, _ = _notebook_code()
    for index in (3, 11, 13, 15):
        cell = notebook["cells"][index]
        require(cell.get("execution_count") is None, f"cell {index} execution count is stale")
        require(cell.get("outputs") == [], f"cell {index} retains stale execution outputs")
        output_text = json.dumps(cell.get("outputs"), ensure_ascii=False)
        for stale in ("22.21", "delta = 0.01", "Step 200"):
            require(stale not in output_text, f"cell {index} retains stale result {stale}")


def test_task5_comment_identifies_make_private_as_ghost_criterion_source():
    notebook, _ = _notebook_code()
    training_configuration = _source(notebook["cells"][11])
    require(
        "Task 5 make_private sets this from its exact four-value ghost return." in training_configuration,
        "ghost criterion ownership comment is stale",
    )
    require("Task 6 sets this" not in training_configuration, "Task 6 is incorrectly named as criterion owner")


def test_task5_patch_manifest_chains_from_task4_and_hashes_current_cells():
    import hashlib

    notebook, _ = _notebook_code()
    manifest = json.loads(PATCH_MANIFEST.read_text(encoding="utf-8"))
    records = manifest["patch_records"]
    require([record["task"] for record in records[:3]] == ["task3", "task4", "task5"], "Task 5 patch prefix changed")
    task4, task5 = records[1:3]
    require(task5["base_notebook_sha256"] == task4["resulting_notebook_sha256"], "Task 5 base does not chain from Task 4")
    require(task5["resulting_notebook_sha256"] == records[3]["base_notebook_sha256"], "Task 5 result does not chain into Task 6")
    require(task5["patched_cell_indices"] == [3, 11, 13, 15], "Task 5 cell delta changed")
    for index in task5["patched_cell_indices"]:
        if not any(index in record["patched_cell_indices"] for record in records[3:]):
            payload = json.dumps(notebook["cells"][index], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            current_hash = hashlib.sha256(payload).hexdigest()
            require(task5["cell_sha256"][str(index)]["after"] == current_hash, f"unchanged Task 5 cell {index} after hash mismatch")


def test_forbidden_fix_fails_under_optimized_python(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"][13]["source"].append("\nModuleValidator.fix(peft_model)\n")
    invalid = tmp_path / "invalid-dp.ipynb"
    invalid.write_text(json.dumps(notebook), encoding="utf-8")
    environment = os.environ.copy()
    environment["VAULTGEMMA_DP_NOTEBOOK_OVERRIDE"] = str(invalid)
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "pytest",
            f"{__file__}::test_validation_precedes_optimizer_and_no_module_fix_or_post_wrap_mutation",
            "-q",
            "--assert=plain",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    require(result.returncode != 0, result.stdout + result.stderr)
    require("module-fixing fallback is forbidden" in result.stdout + result.stderr, "optimized fixture did not identify ModuleValidator.fix")
