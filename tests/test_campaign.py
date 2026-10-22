import pytest

from scripts.posttrain_campaign import choose, config


def candidate(arm, correct, params=100, loss=1.0, split="validation"):
    return dict(arm=arm, correct=correct, trainable_parameters=params,
                best_validation_loss=loss, split=split)


def test_selection_priority_and_no_test_selection():
    arms = [candidate("b", 10, 50), candidate("a", 11, 200), candidate("c", 11, 100)]
    assert choose(arms)["arm"] == "c"
    assert choose(list(reversed(arms))) == choose(arms)
    assert choose([candidate("a", 10, loss=2), candidate("b", 10, loss=1)])["arm"] == "b"
    assert choose([candidate("b", 10), candidate("a", 10)])["arm"] == "a"
    with pytest.raises(ValueError, match="validation"):
        choose([candidate("test", 100, split="test_id")])
    with pytest.raises(ValueError):
        choose([])


def test_fixed_budget_and_scale_across_ranks():
    for rank in [8, 16, 32]:
        cfg = config("custom", "attention_ffn", rank, seed=123)
        assert cfg["alpha"] / cfg["rank"] == 1
        assert cfg["epochs"] == 3 and cfg["max_updates"] == 1500
        assert cfg["batch_size"] * cfg["accumulation_steps"] == 16
        assert cfg["seed"] == 123
