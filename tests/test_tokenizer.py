import pytest

from mce.tokenizer import (
    EN,
    NUM,
    ZH,
    code_mixing_index,
    count_langs,
    poi_indices,
    switch_points,
    tokenize,
    token_texts,
)


def test_cjk_split_per_character_latin_per_word():
    toks = tokenize("我今日好busy")
    assert token_texts(toks) == ["我", "今", "日", "好", "busy"]
    assert [t.lang for t in toks] == [ZH, ZH, ZH, ZH, EN]


def test_digits_and_symbols_are_separate_languages():
    toks = tokenize("我 2026 年 present")
    langs = [t.lang for t in toks]
    assert langs == [ZH, NUM, ZH, EN]


def test_hyphenated_and_apostrophe_words_stay_whole():
    assert token_texts(tokenize("code-switching don't")) == ["code-switching", "don't"]


def test_count_langs():
    counts = count_langs(tokenize("我好busy今日"))
    assert counts[ZH] == 4
    assert counts[EN] == 1


def test_switch_points_marks_first_token_of_new_language():
    toks = tokenize("我好busy啊")
    # 我 好 busy 啊  ->  switch before 'busy' (idx 2) and before '啊' (idx 3)
    assert switch_points(toks) == [2, 3]


def test_digits_do_not_create_switch_points():
    toks = tokenize("我 2026 年")
    assert switch_points(toks) == []


def test_poi_window_covers_both_sides_of_the_boundary():
    toks = tokenize("我好busy啊")
    assert poi_indices(toks, window=1) == {1, 2, 3}
    assert poi_indices(toks, window=2) == {0, 1, 2, 3}


def test_poi_window_must_be_positive():
    with pytest.raises(ValueError):
        poi_indices(tokenize("我好busy"), window=0)


def test_cmi_zero_for_monolingual_and_higher_for_balanced():
    assert code_mixing_index(tokenize("我今日好攰")) == 0.0
    balanced = code_mixing_index(tokenize("我好 busy today ok"))
    assert 0.0 < balanced <= 0.5


def test_empty_string_is_no_tokens():
    assert tokenize("") == []
    assert code_mixing_index(tokenize("")) == 0.0


def test_whitespace_tokenisation_is_the_bug_this_package_avoids():
    """The regression this whole repo exists for.

    Whitespace splitting collapses the Cantonese span into a single token, so
    the reference length is tiny and the error rate explodes past 1.0. The MER
    tokenisation keeps the denominator honest.
    """
    ref = "我今日好busy要開好多meeting"
    hyp = "我今天好忙要開好多會議"

    naive_ref_len = len(ref.split())
    assert naive_ref_len == 1  # the entire utterance is one "word"

    mer_ref_len = len(tokenize(ref))
    assert mer_ref_len == 10  # 8 Chinese characters + 'busy' + 'meeting'
