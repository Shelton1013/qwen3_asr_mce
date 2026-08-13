import pytest

from mce.normalize import (
    NormalizeConfig,
    Normalizer,
    english_number_words_to_digits,
)


def plain(**kw) -> Normalizer:
    """A normaliser with script conversion off, so tests need no OpenCC."""
    cfg = NormalizeConfig(script=None, **kw)
    return Normalizer(cfg)


def test_lowercases_and_strips_punctuation():
    assert plain()("Hello, World!") == "hello world"


def test_cjk_punctuation_is_removed_too():
    assert plain()("我今日好攰，真係！") == "我今日好攰 真係"


def test_intra_word_punctuation_is_joined_not_split():
    # Consistent on both sides, so "don't" vs "dont" never costs an error.
    assert plain()("don't code-switch") == "dont codeswitch"


def test_sensevoice_decoder_tags_are_stripped():
    raw = "<|yue|><|NEUTRAL|><|Speech|><|woitn|>我今日好busy"
    assert plain()(raw) == "我今日好busy"


def test_whisper_special_tokens_are_stripped():
    raw = "<|startoftranscript|><|zh|><|transcribe|>今日開會<|endoftext|>"
    assert plain()(raw) == "今日開會"


def test_bracketed_annotations_are_removed():
    assert plain()("我 [laughter] 好攰") == "我 好攰"


def test_fullwidth_forms_are_folded():
    assert plain()("ＡＢＣ　１２３") == "abc 123"


def test_keep_case_and_keep_punct_are_honoured():
    n = plain(lowercase=False, remove_punct=False)
    assert n("Hello, World!") == "Hello, World!"


def test_drop_strings_are_removed_first():
    n = plain(drop_strings=["[MUSIC]"])
    assert n("[MUSIC] 我今日") == "我今日"


def test_none_and_empty_input():
    assert plain()(None) == ""
    assert plain()("") == ""


def test_script_conversion_requires_opencc():
    opencc = pytest.importorskip("opencc")  # noqa: F841
    n = Normalizer(NormalizeConfig(script="t2s"))
    # 「係」 is Traditional-only in this mapping direction; the point of the test
    # is that both sides land in one script, not the exact characters.
    assert n("開會") == n("开会")


class TestEnglishNumbers:
    def test_simple_units_and_tens(self):
        assert english_number_words_to_digits("twenty five") == "25"

    def test_scales(self):
        assert english_number_words_to_digits("three hundred and fifty") == "350"
        assert english_number_words_to_digits("two thousand") == "2000"

    def test_year_style_runs_are_left_alone(self):
        # "twenty twenty six" is a year, not 46. Refusing to guess is correct.
        assert english_number_words_to_digits("twenty twenty six") == "twenty twenty six"

    def test_non_numbers_pass_through(self):
        assert english_number_words_to_digits("hello world") == "hello world"

    def test_numbers_embedded_in_a_sentence(self):
        assert english_number_words_to_digits("i have five apples") == "i have 5 apples"

    def test_number_normalisation_is_off_by_default(self):
        assert NormalizeConfig().normalize_numbers is False
