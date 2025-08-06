import os
import json
from typing import TypedDict

from phoneme_utils import (
    fer,
    grouped_weighted_needleman_wunsch,
    map_timestamped_phonemes,
    map_phones_by_word,
    TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T,
    TIMESTAMPED_PHONE_PAIRINGS_T,
    TIMESTAMPED_PHONES_BY_WORD_T,
    TIMESTAMPED_PHONES_T,
    WORD_T,
    PHONE_T,
)

_PHONEME_DESCRIPTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "phoneme_descriptions.json"
)
with open(_PHONEME_DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
    _PHONEME_DESCRIPTIONS = json.load(f)
_DESCRIPTIONS_BY_PHONEME = dict(
    (desc["phoneme"], desc) for desc in _PHONEME_DESCRIPTIONS
)


class Mistake(TypedDict):
    target: PHONE_T
    speech: set[PHONE_T]
    words: set[WORD_T]
    occurences_by_word: list[
        tuple[WORD_T, TIMESTAMPED_PHONE_PAIRINGS_T, list[float]]
    ]  # list of (word, paired_mistakes, severities)
    target_description: dict | None
    speech_description: list[dict | None]
    frequency: int
    total_severity: float


def top_phonetic_errors(
    phone_pairings_by_word: TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T, topk=3
):
    """
    Identify the topk mistakes grouped by type (insertion, deletion, substitution) and target phoneme.
    """
    insertion_mistakes, deletion_mistakes, substitution_mistakes, mistakes_by_target = (
        phonetic_errors(phone_pairings_by_word)
    )
    combined_mistakes_by_target: list[Mistake] = []
    for mistakes in mistakes_by_target.values():
        if mistakes[0]["target"] == "-":
            continue
        combined_mistakes_by_target.append(
            Mistake(
                target=mistakes[0]["target"],
                speech=set().union(*(m["speech"] for m in mistakes)),
                words=set().union(*(m["words"] for m in mistakes)),
                occurences_by_word=[
                    (
                        word,
                        [o for m in mistakes for o in m["occurences_by_word"][i][1]],
                        [o for m in mistakes for o in m["occurences_by_word"][i][2]],
                    )
                    for i, (word, _) in enumerate(phone_pairings_by_word)
                ],
                target_description=mistakes[0]["target_description"],
                speech_description=[
                    d for m in mistakes for d in m["speech_description"]
                ],
                frequency=sum(m["frequency"] for m in mistakes),
                total_severity=sum(m["total_severity"] for m in mistakes),
            )
        )

    def sortByFrequencyAndSeverity(mistake: Mistake):
        return -mistake["frequency"] - mistake["total_severity"]

    return {
        "topk_mistakes_by_target": sorted(
            combined_mistakes_by_target, key=sortByFrequencyAndSeverity
        )[:topk],
        "topk_insertion_mistakes": sorted(
            insertion_mistakes, key=sortByFrequencyAndSeverity
        )[:topk],
        "topk_deletion_mistakes": sorted(
            deletion_mistakes, key=sortByFrequencyAndSeverity
        )[:topk],
        "topk_substitution_mistakes": sorted(
            substitution_mistakes, key=sortByFrequencyAndSeverity
        )[:topk],
        "spoken_word_timestamps": [
            (word, paired[0][1][1], paired[-1][1][2])
            for word, paired in phone_pairings_by_word
        ],
    }


def phonetic_errors(
    phone_pairings_by_word: TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T,
) -> tuple[list[Mistake], list[Mistake], list[Mistake], dict[str, list[Mistake]]]:
    """Categorize errors into insertion, deletion, and substitution mistakes (return a list for each) and group by target phoneme (return a dictionary mapping target phoneme to matching mistakes)"""

    insertion_mistakes = []
    deletion_mistakes = []
    substitution_mistakes = []
    mistakes_by_target = {}

    mistakes: dict[str, Mistake] = {}
    for word_ix, (word, pairs) in enumerate(phone_pairings_by_word):
        for target, speech in pairs:
            target_phone, speech_phone = target[0], speech[0]
            if target_phone == speech_phone:
                continue

            key = f"{target_phone}-{speech_phone}"
            if key not in mistakes:
                mistakes[key] = mistake = Mistake(
                    target=target_phone,
                    speech=set([speech_phone]),
                    words=set(),
                    occurences_by_word=[
                        (word, [], []) for word, _ in phone_pairings_by_word
                    ],
                    target_description=_DESCRIPTIONS_BY_PHONEME.get(target_phone),
                    speech_description=[_DESCRIPTIONS_BY_PHONEME.get(speech_phone)],
                    frequency=0,
                    total_severity=0,
                )
                if target_phone == "-":
                    insertion_mistakes.append(mistake)
                elif speech_phone == "-":
                    deletion_mistakes.append(mistake)
                else:
                    substitution_mistakes.append(mistake)
                mistakes_by_target[target_phone] = mistakes_by_target.get(
                    target_phone, []
                ) + [mistake]
            mistakes[key]["frequency"] += 1
            severity = fer(speech_phone, target_phone)
            mistakes[key]["total_severity"] += severity
            mistakes[key]["words"].add(word)
            _, paired, severities = mistakes[key]["occurences_by_word"][word_ix]
            paired.append((target, speech))
            severities.append(severity)

    return (
        insertion_mistakes,
        deletion_mistakes,
        substitution_mistakes,
        mistakes_by_target,
    )


def pair_by_words(
    target_by_words: TIMESTAMPED_PHONES_BY_WORD_T, speech: TIMESTAMPED_PHONES_T
) -> TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T:
    """
    Pairs the target and speech by words.
    Returns an array of tuples with the (word, [(target_phoneme_timestamped, speech_phoneme_timestamped), ...])
    target_phoneme_timestamped = (phoneme, start_time, end_time)
    speech_phoneme_timestamped = (phoneme, start_time, end_time)
    """
    speech = map_timestamped_phonemes(speech)
    target_by_words = map_phones_by_word(target_by_words)

    return grouped_weighted_needleman_wunsch(target_by_words, speech)


def score_words_cer(
    phone_pairings_by_word: TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T,
) -> tuple[list[tuple[WORD_T, float]], float]:
    """
    This function scores the words based on the character error rate
    Returns a list of tuples with the (word, score) and an average score
    """
    word_scores = [
        (word, 1 - sum(1 for t, s in pairs if t[0] != s[0]) / len(pairs))
        for word, pairs in phone_pairings_by_word
    ]
    average_score = sum(score for _, score in word_scores) / len(phone_pairings_by_word)
    return word_scores, average_score


def get_unique_phonemes(
    phone_pairings_by_word: TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T,
) -> set[PHONE_T]:
    """Get the set of all phonemes in the phone_pairings_by_word"""
    return set(
        phone[0]
        for _, pairs in phone_pairings_by_word
        for target_phone, source_phone in pairs
        for phone in [target_phone, source_phone]
    )
