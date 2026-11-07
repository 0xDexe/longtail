from longtail.data import parse_gsm8k_answer
from longtail.reward import correctness_reward, extract_answer, normalize


def test_normalize():
    assert normalize("1,234") == "1234"
    assert normalize("$18.00") == "18"
    assert normalize("3.50.") == "3.5"
    assert normalize("abc") is None


def test_boxed_preferred_over_trailing_numbers():
    assert extract_answer("so \\boxed{42} because 7 * 6 = 42 and 3") == "42"


def test_last_boxed_wins():
    assert extract_answer("\\boxed{1} wait, actually \\boxed{2}") == "2"


def test_thinking_is_ignored_when_closed():
    assert extract_answer("<think>maybe 10? no 12</think>The answer is \\boxed{12}.") == "12"


def test_truncated_thinking_scores_nothing():
    assert extract_answer("<think>step 1 ... 17 apples and then") is None


def test_fallback_to_last_number():
    assert extract_answer("She has 5 left, total 1,200 dollars") == "1200"


def test_reward():
    assert correctness_reward("<think>x</think>\\boxed{72}", "72") == 1.0
    assert correctness_reward("\\boxed{71}", "72") == 0.0
    assert correctness_reward("", "72") == 0.0


def test_gsm8k_answer_parse():
    assert parse_gsm8k_answer("Natalia sold 48/2 = 24 clips.\n#### 1,072") == "1072"
