from g1_speech.evaluation import edit_distance, error_rate, normalize_characters


def test_character_metric_normalizes_width_case_and_whitespace():
    reference = normalize_characters("Ａ B 机器人")
    hypothesis = normalize_characters("a b 机器认")

    assert edit_distance(reference, hypothesis) == 1
    assert error_rate(reference, hypothesis) == 1 / 5


def test_error_rate_is_undefined_for_empty_reference():
    assert error_rate([], ["x"]) is None
