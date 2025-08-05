import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.phoneme_utils import _ALL_MAPPINGS, _PHONEMES_TO_MASK
from src.feedback import VOCAB_CONTENT
from src.server import processor


def test_vocab_feedback_no_duplicates():
    assert len(set(p["phoneme"] for p in VOCAB_CONTENT)) == len(
        list(p["phoneme"] for p in VOCAB_CONTENT)
    )


def test_vocab_feedback_format():
    for feedback in VOCAB_CONTENT:
        assert "phoneme" in feedback.keys() and type(feedback["phoneme"]) == str
        assert "explanation" in feedback.keys() and type(feedback["explanation"]) == str
        assert (
            "phonetic_spelling" in feedback.keys()
            and type(feedback["phonetic_spelling"]) == str
        )
        assert "video" in feedback.keys()
        assert (
            "description" in feedback.keys() and type(feedback["description"]) == list
        )
        assert "examples" in feedback.keys()
        for example in feedback["examples"]:
            assert (
                "word" in example.keys()
                and type(example["word"]) == str
                and example["word"].count("*") >= 2
                and example["word"].count("*") % 2 == 0
            ), example["word"]
            assert (
                "phonetic_spelling" in example.keys()
                and type(example["phonetic_spelling"]) == str
            )


def test_vocab_feedback_coverage():
    """Make sure the feedback covers exactly the vocab tokens of the model while taking into account that the model vocab will be mapped"""

    feedback_phonemes = set(p["phoneme"] for p in VOCAB_CONTENT)
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
