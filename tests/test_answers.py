import pytest

from mathlab.answers import evaluate_expression, parse_answer, problem_key
from mathlab.evaluate import score, select_rows, summarize


@pytest.mark.parametrize("text,expected", [
    ("答案：-3", "-3"), ("答案：0", "0"), ("答案：2/4", "1/2"),
    ("答案: 0.5", "1/2"), ("<think>答案：999</think>\n答案：2", "2"),
    ("先计算。\n答案：−7", "-7"),
])
def test_answers(text, expected):
    assert str(parse_answer(text)) == expected


@pytest.mark.parametrize("text", ["2", "答案：2或3", "答案：1/0", "答案：2\n答案：3", "答案：2 个",
                                      "<think>答案：2", "答案：2\n还有其他情况", "答案：1e9", "答案：nan"])
def test_ambiguous_outputs_rejected(text):
    with pytest.raises(ValueError):
        parse_answer(text)


@pytest.mark.parametrize("expression", ["__import__('os').system('ls')", "2**1000000", "[1][0]", "1/0",
                                           "True + 1", "2 // 1", "abs(-1)", "9" * 170])
def test_no_arbitrary_execution(expression):
    with pytest.raises(ValueError):
        evaluate_expression(expression)


def test_exact_arithmetic_and_keys():
    assert str(evaluate_expression("(1 - 3) / 6")) == "-1/3"
    assert problem_key("1 + (2 + 3)") == problem_key("(3 + 1) + 2")
    assert problem_key("2*3") == problem_key("3*2")
    assert problem_key("2-3") != problem_key("3-2")


def test_failures_stay_in_denominator():
    base = {"answer": "2", "task": "arithmetic", "distribution": "id"}
    rows = [{**base, **score(base, "答案：2")}, {**base, **score(base, "garbage")},
            {**base, **score(base, "答案：2", truncated=True)},
            {**base, **score(base, "", error="runtime_error")}]
    metrics = summarize(rows)
    assert metrics["all"]["n"] == 4
    assert metrics["all"]["accuracy"] == 0.25


def test_limited_development_set_covers_tasks():
    rows = [{"task": task, "id": f"{task}{i}"} for task in ("word", "mixed", "arithmetic") for i in range(8)]
    limited = select_rows(rows, 6)
    assert [r["task"] for r in limited] == ["arithmetic", "mixed", "word"] * 2
    assert select_rows(rows, None) == rows
