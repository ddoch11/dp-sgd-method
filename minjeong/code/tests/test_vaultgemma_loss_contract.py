import ast
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.grad_sample import (
    GradSampleHooksFastGradientClipping,
    GradSampleModuleFastGradientClipping,
    GradSampleModuleExpandedWeights,
)
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import (
    DPLossFastGradientClipping,
    DPTensorFastGradientClipping,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import default_data_collator


pytestmark = [
    pytest.mark.filterwarnings("ignore:Secure RNG turned off.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Full backward hook is firing.*:UserWarning"),
]


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(
    os.environ.get(
        "VAULTGEMMA_LOSS_NOTEBOOK_OVERRIDE",
        ROOT
        / "workspaces/vaultgemma_vectorized/[VaultGemma]FineTuning_Vectorized.ipynb",
    )
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _loss_namespace() -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    function_nodes = []
    wanted = {
        "tokenize_and_mask",
        "validate_response_only_batch",
        "response_only_data_collator",
        "shift_response_tensors",
        "response_only_loss",
        "ghost_response_only_loss",
        "dispatch_response_only_loss",
        "forward_model_inputs",
        "forward_response_only_loss",
        "accumulate_record_weighted_loss",
        "finalize_record_weighted_loss",
        "optimizer_completed_logical_step",
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        sanitized = "\n".join(
            line
            for line in _source(cell).splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        tree = ast.parse(sanitized)
        function_nodes.extend(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        )
    found = {node.name for node in function_nodes}
    require(found == wanted, f"missing notebook loss functions: {sorted(wanted - found)}")
    module = ast.fix_missing_locations(ast.Module(body=function_nodes, type_ignores=[]))
    namespace = {
        "torch": torch,
        "default_data_collator": default_data_collator,
        "DPLossFastGradientClipping": DPLossFastGradientClipping,
        "GradSampleHooksFastGradientClipping": GradSampleHooksFastGradientClipping,
        "GradSampleModuleFastGradientClipping": GradSampleModuleFastGradientClipping,
        "DPOptimizerFastGradientClipping": DPOptimizerFastGradientClipping,
        "MAX_SEQUENCE_LENGTH": 256,
        "ALLOWED_GRAD_SAMPLE_MODES": {"hooks", "functorch", "ew", "ghost"},
    }
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


class _FakeTokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, **kwargs):
        self.calls += 1
        require(kwargs["truncation"] is True, "tokenization must truncate")
        require(kwargs["max_length"] == 256, "tokenization length changed")
        require(kwargs["padding"] == "max_length", "tokenization padding changed")
        require(kwargs["return_tensors"] == "pt", "tensor return mode changed")
        if self.calls == 1:
            return {
                "input_ids": torch.tensor(
                    [[10, 11, 12, 13, 0, 0], [20, 21, 22, 23, 24, 0]]
                ),
                "attention_mask": torch.tensor(
                    [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]]
                ),
            }
        return {
            "input_ids": torch.zeros((2, 6), dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]]
            ),
        }


def _synthetic_logits_and_labels():
    logits = torch.tensor(
        [
            [
                [5.0, -1.0, -2.0],
                [3.0, 0.0, -1.0],
                [-1.0, 3.0, 0.0],
                [0.0, -1.0, 3.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [-2.0, 4.0, 0.0],
                [0.0, 2.0, -1.0],
                [-1.0, 0.0, 1.0],
                [2.0, -2.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
        ],
        requires_grad=True,
    )
    labels = torch.tensor(
        [
            [-100, -100, 0, -100, -100],
            [-100, -100, 1, 2, -100],
        ]
    )
    return logits, labels


def _manual_record_and_global_losses(logits, labels):
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    token_losses = F.cross_entropy(
        shifted_logits.float().reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shifted_labels.shape)
    valid = shifted_labels.ne(-100)
    record_loss = (token_losses.sum(dim=1) / valid.sum(dim=1)).mean()
    global_loss = token_losses.sum() / valid.sum()
    return record_loss, global_loss


def test_tokenization_masks_prompt_and_padding_then_collator_preserves_labels():
    namespace = _loss_namespace()
    namespace["tokenize_and_mask"].__globals__["tokenizer"] = _FakeTokenizer()
    tokenized = namespace["tokenize_and_mask"](
        {"input": ["question one", "question two"], "output": ["a", "b"]}
    )
    expected = torch.tensor(
        [
            [-100, -100, 12, 13, -100, -100],
            [-100, -100, -100, 23, 24, -100],
        ]
    )
    require(torch.equal(tokenized["labels"], expected), "prompt/padding masks changed")

    features = [
        {key: value[index].tolist() for key, value in tokenized.items()}
        for index in range(2)
    ]
    batch = namespace["response_only_data_collator"](features)
    require(torch.equal(batch["labels"], expected), "collator overwrote prepared labels")
    require(
        torch.all(batch["labels"][batch["attention_mask"].eq(0)] == -100),
        "padding labels were restored",
    )

    bad_padding = [
        {key: list(value) for key, value in feature.items()} for feature in features
    ]
    bad_padding[0]["labels"][-1] = 0
    with pytest.raises(ValueError, match="padding labels must be -100"):
        namespace["response_only_data_collator"](bad_padding)

    bad_prompt = [
        {key: list(value) for key, value in feature.items()} for feature in features
    ]
    bad_prompt[0]["labels"] = [-100, 12, -100, 13, -100, -100]
    with pytest.raises(ValueError, match="non-prefix prompt mask"):
        namespace["response_only_data_collator"](bad_prompt)


def test_record_loss_shifts_once_and_is_not_global_token_mean():
    namespace = _loss_namespace()
    logits, labels = _synthetic_logits_and_labels()
    actual = namespace["response_only_loss"](logits, labels)
    expected_record, global_token_weighted = _manual_record_and_global_losses(logits, labels)
    require(torch.allclose(actual, expected_record), "record-level response loss mismatch")
    require(
        not torch.allclose(actual, global_token_weighted),
        "unequal response lengths collapsed to a global token mean",
    )

    shifted_logits, shifted_labels = namespace["shift_response_tensors"](logits, labels)
    require(torch.equal(shifted_logits, logits[:, :-1, :]), "logits shifted more than once")
    require(torch.equal(shifted_labels, labels[:, 1:]), "labels shifted more than once")


def test_zero_target_record_is_rejected_before_reduction():
    namespace = _loss_namespace()
    logits, labels = _synthetic_logits_and_labels()
    labels[0] = -100
    with pytest.raises(
        ValueError, match="every record must contain at least one response target token"
    ):
        namespace["response_only_loss"](logits, labels)


class _TinyOutput:
    def __init__(self, logits):
        self.logits = logits


class _TinyCausalLM(nn.Module):
    def __init__(self, vocabulary_size=7, hidden_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return _TinyOutput(self.output(self.embedding(input_ids)))


class _TinyHFSignatureModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        require(labels is None, "labels reached the HF-signature module")
        require(input_ids is not None, "input_ids is required")
        require(attention_mask is not None, "attention_mask is required")
        return _TinyOutput(self.projection(input_ids) * attention_mask.unsqueeze(-1))


def _make_private_tiny(mode):
    torch.manual_seed(123)
    model = _TinyCausalLM()
    reference_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loader = DataLoader(TensorDataset(torch.arange(2)), batch_size=2)
    result = PrivacyEngine().make_private(
        module=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(ignore_index=-100, reduction="mean"),
        data_loader=loader,
        noise_multiplier=0.0,
        max_grad_norm=1_000_000.0,
        poisson_sampling=False,
        grad_sample_mode=mode,
    )
    return reference_model, result


def test_ew_real_wrapper_requires_positional_primary_input():
    torch.manual_seed(123)
    module = _TinyHFSignatureModule()
    input_ids = torch.randn(2, 4, 3)
    attention_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    expected = module(input_ids=input_ids, attention_mask=attention_mask).logits
    wrapped = GradSampleModuleExpandedWeights(module)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'x'"):
        wrapped(input_ids=input_ids, attention_mask=attention_mask)
    actual = wrapped(input_ids, attention_mask=attention_mask).logits
    require(torch.allclose(actual, expected), "EW positional call changed logits")


@pytest.mark.parametrize("mode", ["hooks", "functorch", "ew"])
def test_shared_forward_helper_runs_real_opacus_wrappers(mode):
    namespace = _loss_namespace()
    reference_model, private_result = _make_private_tiny(mode)
    private_model = private_result[0]
    input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 2, -100], [-100, -100, 3, 4]])
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    expected_logits = reference_model(input_ids, attention_mask=attention_mask).logits
    actual_logits = namespace["forward_model_inputs"](private_model, batch).logits
    require(torch.allclose(actual_logits, expected_logits), f"{mode} logits changed")
    expected_loss = namespace["response_only_loss"](expected_logits, labels)
    actual_loss = namespace["response_only_loss"](actual_logits, labels)
    require(torch.allclose(actual_loss, expected_loss), f"{mode} response loss changed")


def test_official_ghost_criterion_runs_backward_step_and_matches_reference():
    namespace = _loss_namespace()
    reference_model, private_result = _make_private_tiny("ghost")
    private_model, private_optimizer, private_criterion, _ = private_result
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)
    input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 2, -100], [-100, -100, 3, 4]])

    reference_logits = reference_model(input_ids, attention_mask).logits
    reference_loss = namespace["response_only_loss"](reference_logits, labels)
    reference_loss.backward()
    reference_optimizer.step()

    private_logits = private_model(input_ids, attention_mask).logits
    wrapped = namespace["ghost_response_only_loss"](
        private_criterion,
        private_model,
        private_optimizer,
        private_logits,
        labels,
    )
    require(
        isinstance(wrapped, DPTensorFastGradientClipping),
        "ghost path did not use the official Opacus loss wrapper",
    )
    expected_record, _ = _manual_record_and_global_losses(private_logits, labels)
    require(wrapped.loss_per_sample.shape == (2,), "ghost privacy unit is not one record")
    require(
        torch.allclose(wrapped.detach(), expected_record),
        "ghost wrapper output differs from explicit record-level loss",
    )
    require(torch.allclose(wrapped.detach(), reference_loss.detach()), "ghost loss mismatch")
    wrapped.backward()
    private_optimizer.step()
    for private_parameter, reference_parameter in zip(
        private_model.parameters(), reference_model.parameters()
    ):
        require(
            torch.allclose(private_parameter, reference_parameter, atol=1e-6, rtol=1e-5),
            "zero-noise high-clip ghost update differs from record-mean reference",
        )

    zero_target_labels = labels.clone()
    zero_target_labels[0] = -100
    with pytest.raises(
        ValueError, match="every record must contain at least one response target token"
    ):
        namespace["ghost_response_only_loss"](
            private_criterion,
            private_model,
            private_optimizer,
            private_logits,
            zero_target_labels,
        )


def test_standard_hooks_wrapper_is_rejected_before_ghost_backward():
    namespace = _loss_namespace()
    _, private_result = _make_private_tiny("hooks")
    hooks_model, hooks_optimizer, _ = private_result
    arbitrary_criterion = DPLossFastGradientClipping(
        hooks_model,
        hooks_optimizer,
        nn.CrossEntropyLoss(ignore_index=-100, reduction="mean"),
        loss_reduction="mean",
    )
    input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 2, -100], [-100, -100, 3, 4]])
    logits = hooks_model(input_ids).logits
    with pytest.raises(TypeError, match="fast-gradient-clipping module"):
        namespace["ghost_response_only_loss"](
            arbitrary_criterion,
            hooks_model,
            hooks_optimizer,
            logits,
            labels,
        )


def test_ghost_criterion_must_match_the_live_module_and_optimizer():
    namespace = _loss_namespace()
    _, first_result = _make_private_tiny("ghost")
    _, second_result = _make_private_tiny("ghost")
    _, _, first_criterion, _ = first_result
    second_model, second_optimizer, _, _ = second_result
    input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 2, -100], [-100, -100, 3, 4]])
    logits = second_model(input_ids).logits
    with pytest.raises(ValueError, match="not bound to the live module and optimizer"):
        namespace["ghost_response_only_loss"](
            first_criterion,
            second_model,
            second_optimizer,
            logits,
            labels,
        )


def test_record_weighted_aggregation_handles_remainder_and_physical_chunks():
    namespace = _loss_namespace()
    accumulate = namespace["accumulate_record_weighted_loss"]
    finalize = namespace["finalize_record_weighted_loss"]

    loss_sum, record_count = 0.0, 0
    for batch_mean, batch_size in ((2.0, 128), (10.0, 32)):
        loss_sum, record_count = accumulate(
            loss_sum, record_count, batch_mean, batch_size
        )
    require(record_count == 160, "logical remainder record count mismatch")
    require(finalize(loss_sum, record_count) == pytest.approx(3.6), "remainder misweighted")
    require(finalize(loss_sum, record_count) != pytest.approx(6.0), "batch means averaged")

    chunk_sum, chunk_count = 0.0, 0
    for batch_mean in [2.0] * 8 + [10.0] * 2:
        chunk_sum, chunk_count = accumulate(chunk_sum, chunk_count, batch_mean, 16)
    require(chunk_count == 160, "physical chunk record count mismatch")
    require(
        finalize(chunk_sum, chunk_count) == pytest.approx(3.6),
        "physical chunks changed record weighting",
    )
    with pytest.raises(ValueError, match="record count must be positive"):
        finalize(0.0, 0)


def test_final_eval_is_physical_bounded_exact_and_side_effect_free():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    loader_source = _source(notebook["cells"][11])
    training_source = _source(notebook["cells"][15])
    final_eval = training_source[
        training_source.index("eval_started = time.perf_counter()\n") :
        training_source.index("require_contract(ADAPTER_PATH.is_dir()")
    ]

    for fragment in (
        "EVAL_BATCH_SIZE = MAX_PHYSICAL_BATCH_SIZE",
        "batch_size=EVAL_BATCH_SIZE",
        "shuffle=False",
        "drop_last=False",
        "require_contract(eval_record_count == EVAL_SIZE",
        "eval_accountant_steps_before = accountant_logical_steps(privacy_engine.accountant)",
        "eval_scheduler_steps_before = scheduler_step_count",
        "optimizer.zero_grad(set_to_none=True)",
        "accountant_logical_steps(privacy_engine.accountant) == eval_accountant_steps_before",
        "scheduler_step_count == eval_scheduler_steps_before",
    ):
        require(
            fragment in loader_source + final_eval,
            f"physical final-eval contract missing: {fragment}",
        )
    require("batch_size=per_device_train_batch_size" not in loader_source[loader_source.index("eval_dataloader = DataLoader(") :], "eval uses logical batch 128")
    for forbidden in ("optimizer.step(", "lr_scheduler.step(", ".backward("):
        require(forbidden not in final_eval, f"final eval mutates training state: {forbidden}")

    namespace = _loss_namespace()
    response_loss = namespace["response_only_loss"]
    accumulate = namespace["accumulate_record_weighted_loss"]
    finalize = namespace["finalize_record_weighted_loss"]
    base_logits, base_labels = _synthetic_logits_and_labels()
    logits = base_logits.detach().repeat(400, 1, 1)
    labels = base_labels.repeat(400, 1)

    def aggregate(chunk_size):
        total, count = 0.0, 0
        for start in range(0, 800, chunk_size):
            stop = min(start + chunk_size, 800)
            mean_loss = response_loss(logits[start:stop], labels[start:stop])
            total, count = accumulate(total, count, mean_loss.item(), stop - start)
        return finalize(total, count), count

    physical_loss, physical_count = aggregate(16)
    logical_loss, logical_count = aggregate(128)
    require(physical_count == logical_count == 800, "eval did not cover 800 records exactly once")
    require(physical_loss == pytest.approx(logical_loss, abs=1e-7), "physical chunks changed response-only loss")
    require(math.exp(physical_loss) == pytest.approx(math.exp(logical_loss), abs=1e-7), "physical chunks changed perplexity")

    forward_source = _source(notebook["cells"][9])
    for fragment in (
        'if key not in {"input_ids", "labels"}',
        'return module(batch["input_ids"], **model_kwargs)',
        "outputs = forward_model_inputs(module, batch)",
    ):
        require(fragment in forward_source, f"shared positional forward missing: {fragment}")


class _StepState:
    def __init__(self, skipped):
        self._is_last_step_skipped = skipped


def test_logical_step_gate_and_training_cell_use_record_weighting():
    namespace = _loss_namespace()
    completed = namespace["optimizer_completed_logical_step"]
    require(not completed(_StepState(True)), "skipped physical chunk counted as logical step")
    require(completed(_StepState(False)), "completed logical step was suppressed")
    require(completed(object()), "non-DP optimizer should complete its step")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    training_source = _source(notebook["cells"][15])
    for fragment in (
        "accumulate_record_weighted_loss(",
        "finalize_record_weighted_loss(",
        "logical_step_completed = optimizer_completed_logical_step(optimizer)",
        "if logical_step_completed:",
        "GHOST_LOSS_CRITERION",
    ):
        require(fragment in training_source, f"training integration missing: {fragment}")


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    def __init__(self, logits):
        self.logits = logits
        self.received = None

    def __call__(self, input_ids, **kwargs):
        require("labels" not in kwargs, "labels reached the Hugging Face model forward")
        self.received = {"input_ids": input_ids, **kwargs}
        return _FakeOutput(self.logits)


def test_forward_and_dispatch_never_use_hf_global_loss_or_silent_fallback():
    namespace = _loss_namespace()
    logits, labels = _synthetic_logits_and_labels()
    batch = {
        "input_ids": torch.ones((2, 5), dtype=torch.long),
        "attention_mask": torch.ones((2, 5), dtype=torch.long),
        "labels": labels,
    }
    model = _FakeModel(logits)
    actual = namespace["forward_response_only_loss"](
        model,
        batch,
        optimizer=None,
        grad_sample_mode="hooks",
        training=True,
        ghost_loss_criterion=None,
    )
    expected, _ = _manual_record_and_global_losses(logits, labels)
    require(torch.allclose(actual, expected), "training forward used the wrong loss")
    require(set(model.received) == {"input_ids", "attention_mask"}, "forward inputs changed")

    eval_model = _FakeModel(logits)
    eval_loss = namespace["forward_response_only_loss"](
        eval_model,
        batch,
        optimizer=None,
        grad_sample_mode="ghost",
        training=False,
        ghost_loss_criterion=None,
    )
    require(torch.allclose(eval_loss, expected), "eval did not use explicit response loss")

    with pytest.raises(RuntimeError, match="ghost mode requires"):
        namespace["dispatch_response_only_loss"](
            "ghost", None, None, None, logits, labels
        )
    with pytest.raises(ValueError, match="unsupported gradient-sample mode"):
        namespace["dispatch_response_only_loss"](
            "not-a-mode", None, None, None, logits, labels
        )


def test_loss_contract_fails_under_optimized_python_when_notebook_is_invalid(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = _source(cell)
            if "def response_only_loss" in source:
                cell["source"] = source.replace(
                    "return (token_losses.sum(dim=1) / counts).mean()",
                    "return token_losses.sum() / counts.sum()",
                )
                break
    invalid_notebook = tmp_path / "invalid-loss.ipynb"
    invalid_notebook.write_text(json.dumps(notebook), encoding="utf-8")
    environment = os.environ.copy()
    environment["VAULTGEMMA_LOSS_NOTEBOOK_OVERRIDE"] = str(invalid_notebook)
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "pytest",
            f"{__file__}::test_record_loss_shifts_once_and_is_not_global_token_mean",
            "-q",
            "--assert=plain",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    require(result.returncode != 0, result.stdout + result.stderr)
    require(
        "record-level response loss mismatch" in result.stdout + result.stderr,
        "optimized negative fixture did not identify the loss contract failure",
    )
