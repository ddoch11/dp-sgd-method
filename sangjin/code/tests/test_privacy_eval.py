from __future__ import annotations

import sys
from pathlib import Path

import torch
from datasets import Dataset

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from create_synthetic_canary_manifest import synthetic_answer
from evaluate_synthetic_canary import contains_code
from privacy_eval_common import (
    apply_member_canaries,
    canonical_json_sha256,
    matching_prefix_length,
    response_losses,
    token_edit_distance,
)


def test_token_edit_and_prefix_metrics() -> None:
    assert token_edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert token_edit_distance([1, 2, 3], [1, 4, 3]) == 1
    assert token_edit_distance([], [1, 2]) == 2
    assert matching_prefix_length([1, 2, 9], [1, 2, 3]) == 2


def test_code_matching_requires_digit_boundaries() -> None:
    assert contains_code("The code is 5831.", "5831")
    assert not contains_code("The code is 158319.", "5831")
    assert not contains_code("No code was generated.", "5831")


def test_synthetic_answer_repeats_only_the_fake_code() -> None:
    answer = synthetic_answer("SYN-P0001", "5831", 4)
    assert answer.count("5831") == 4
    assert "synthetic" in answer.lower()


def test_member_canary_replaces_one_train_record() -> None:
    selected = Dataset.from_dict(
        {
            "input": ["q0", "q1", "q2"],
            "output": ["a0", "a1", "a2"],
            "_source_index": [10, 11, 12],
        }
    )
    manifest = {
        "train_size": 2,
        "members": [
            {
                "canary_index": 0,
                "train_position": 1,
                "replaced_source_index": 11,
                "input": "synthetic q",
                "output": "synthetic a",
            }
        ],
    }
    replaced = apply_member_canaries(selected, manifest)
    assert replaced[0]["input"] == "q0"
    assert replaced[1]["input"] == "synthetic q"
    assert replaced[1]["_source_index"] == -1
    assert replaced[2]["input"] == "q2"


def test_response_loss_is_per_record_token_mean() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]],
            [[0.0, 3.0], [3.0, 0.0], [0.0, 3.0]],
        ]
    )
    labels = torch.tensor([[-100, 1, 0], [-100, 0, 1]])
    losses, sums, counts = response_losses(logits, labels)
    assert tuple(losses.shape) == (2,)
    assert torch.equal(counts, torch.tensor([2, 2]))
    assert torch.allclose(losses * counts, sums)


def test_canonical_hash_is_order_stable_for_mappings() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )
