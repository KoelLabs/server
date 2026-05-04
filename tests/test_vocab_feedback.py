import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.phoneme_utils import _ALL_MAPPINGS, _PHONEMES_TO_MASK
from src.feedback import _PHONEME_DESCRIPTIONS
from src.transcription import processor

REQUIRED_PHONEME_DESCRIPTION_FIELDS = {
    "phoneme": str,
    "primary": bool,
    "label": str,
    "plain_english": str,
    "description": list,
    "explanation": str,
    "primary_cue": str,
    "visual_cue": str,
    "feel_check": str,
    "anchor_words": list,
    "minimal_pairs": list,
    "drill_ladder": dict,
    "examples": list,
    "phonetic_spelling": str,
    "video": str,
}

OPTIONAL_PHONEME_DESCRIPTION_FIELDS = {
    "audio": str,
    "confused_phone": str,
    "minimal_pairs_by_confusion": dict,
    "sort_key": str,
    "voicing_tag": str,
}

ALLOWED_PHONEME_DESCRIPTION_FIELDS = set(REQUIRED_PHONEME_DESCRIPTION_FIELDS) | set(
    OPTIONAL_PHONEME_DESCRIPTION_FIELDS
)


def test_vocab_feedback_no_duplicates():
    assert len(set(p["phoneme"] for p in _PHONEME_DESCRIPTIONS)) == len(
        list(p["phoneme"] for p in _PHONEME_DESCRIPTIONS)
    )


def test_vocab_feedback_format():
    for feedback in _PHONEME_DESCRIPTIONS:
        assert set(feedback.keys()).issubset(ALLOWED_PHONEME_DESCRIPTION_FIELDS)

        for field, expected_type in REQUIRED_PHONEME_DESCRIPTION_FIELDS.items():
            assert field in feedback.keys()
            assert type(feedback[field]) == expected_type

        for field, expected_type in OPTIONAL_PHONEME_DESCRIPTION_FIELDS.items():
            if field in feedback:
                assert type(feedback[field]) == expected_type

        for word in feedback["anchor_words"]:
            assert type(word) == str

        for pair in feedback["minimal_pairs"]:
            assert type(pair) == list
            assert len(pair) == 2
            assert all(type(word) == str for word in pair)

        for pairs in feedback.get("minimal_pairs_by_confusion", {}).values():
            assert type(pairs) == list
            for pair in pairs:
                assert type(pair) == list
                assert len(pair) == 2
                assert all(type(word) == str for word in pair)

        for ladder_key in ("sound", "syllables", "words", "phrases"):
            assert ladder_key in feedback["drill_ladder"]
            assert type(feedback["drill_ladder"][ladder_key]) == list
            assert all(
                type(value) == str for value in feedback["drill_ladder"][ladder_key]
            )

        for example in feedback["examples"]:
            assert (
                "word" in example.keys()
                and type(example["word"]) == str
                and example["word"].count("*") % 2 == 0
            ), example["word"]
            assert (
                "phonetic_spelling" in example.keys()
                and type(example["phonetic_spelling"]) == str
            )


def test_vocab_feedback_primary_subset_matches_app_library():
    primary_phonemes = {p["phoneme"] for p in _PHONEME_DESCRIPTIONS if p["primary"]}
    assert len(primary_phonemes) == 39


def test_vocab_feedback_coverage():
    """Make sure the feedback covers exactly the vocab tokens of the model while taking into account that the model vocab will be mapped"""

    feedback_phonemes = set(p["phoneme"] for p in _PHONEME_DESCRIPTIONS)
    model_phonemes = set(
        _ALL_MAPPINGS.get(p, p) for p in processor.tokenizer.get_vocab().keys()
    ) - set(processor.tokenizer.all_special_tokens)

    in_feedback_not_in_model = feedback_phonemes.difference(model_phonemes)
    in_feedback_not_in_model -= (
        _PHONEMES_TO_MASK.keys()
    )  # we allow feedback to have extra explanations for tokens that will be masked away
    assert (
        len(in_feedback_not_in_model) == 0
    ), f"Feedback covers {in_feedback_not_in_model} not in model vocab"

    in_model_not_in_feedback = model_phonemes.difference(feedback_phonemes)
    assert (
        len(in_model_not_in_feedback) == 0
    ), f"model vocab has {in_model_not_in_feedback} that feedback does not"
