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


class FeedbackCard(TypedDict, total=False):
    type: str
    caption: str
    details: str
    words: list[WORD_T]
    video: str | None
    frequency: int
    total_severity: float
    target: PHONE_T
    speech: list[PHONE_T]
    target_description: dict | None
    speech_description: list[dict | None]


class AnalyzedWord(TypedDict):
    text: WORD_T
    score: float
    start: float
    end: float


class AnalysisResponse(TypedDict):
    words: list[AnalyzedWord]
    average_score: float
    spoken_word_timestamps: list[tuple[WORD_T, float, float]]
    feedback: list[FeedbackCard]


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


def _sort_by_frequency_and_severity(mistake: Mistake):
    freq_and_serverity = -mistake["frequency"] - mistake["total_severity"]
    if "-" in mistake["speech"]:
        freq_and_serverity += 100_000
    return freq_and_serverity


def _combine_target_mistakes(
    phone_pairings_by_word: TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T,
    mistakes_by_target: dict[str, list[Mistake]],
) -> list[Mistake]:
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
    return combined_mistakes_by_target


def _format_list(values: list[str]) -> str:
    if len(values) == 0:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _phone_label(phone: PHONE_T, description: dict | None) -> str:
    phonetic_spelling = description.get("phonetic_spelling") if description else None
    if phonetic_spelling:
        return f"/{phone}/ ({phonetic_spelling})"
    return f"/{phone}/"


def _ordered_words(mistake: Mistake) -> list[WORD_T]:
    return [
        word
        for word, occurences, _ in mistake["occurences_by_word"]
        if len(occurences) > 0
    ]


def _first_video(*descriptions: dict | None) -> str | None:
    for description in descriptions:
        if description and description.get("video"):
            return description["video"]
    return None


def _build_target_card(mistake: Mistake) -> FeedbackCard:
    target_description = mistake["target_description"]
    target_label = _phone_label(mistake["target"], target_description)
    spoken_phones = sorted(phone for phone in mistake["speech"] if phone != "-")
    spoken_descriptions: list[dict | None] = [
        description
        for description in mistake["speech_description"]
        if description and description.get("phoneme") in spoken_phones
    ]

    if len(spoken_phones) == 0:
        return FeedbackCard(
            type="under_enunciated",
            caption=f"You under-enunciated {target_label}.",
            details=(
                f"Focus on fully pronouncing {target_label}. "
                f"{target_description.get('explanation', '') if target_description else ''}"
            ).strip(),
            words=_ordered_words(mistake),
            video=_first_video(target_description),
            frequency=mistake["frequency"],
            total_severity=mistake["total_severity"],
            target=mistake["target"],
            speech=[],
            target_description=target_description,
            speech_description=[],
        )

    spoken_labels = [
        _phone_label(phone, _DESCRIPTIONS_BY_PHONEME.get(phone))
        for phone in spoken_phones
    ]
    return FeedbackCard(
        type="mispronunciation",
        caption=f"You're pronouncing {target_label} more like {_format_list(spoken_labels)}.",
        details=(
            f"Aim for {target_label} instead of {_format_list(spoken_labels)}. "
            f"{target_description.get('explanation', '') if target_description else ''}"
        ).strip(),
        words=_ordered_words(mistake),
        video=_first_video(target_description, *spoken_descriptions),
        frequency=mistake["frequency"],
        total_severity=mistake["total_severity"],
        target=mistake["target"],
        speech=spoken_phones,
        target_description=target_description,
        speech_description=spoken_descriptions,
    )


def _build_insertion_card(mistake: Mistake) -> FeedbackCard:
    inserted_phones = sorted(phone for phone in mistake["speech"] if phone != "-")
    inserted_descriptions = [
        _DESCRIPTIONS_BY_PHONEME.get(phone) for phone in inserted_phones
    ]
    inserted_labels = [
        _phone_label(phone, _DESCRIPTIONS_BY_PHONEME.get(phone))
        for phone in inserted_phones
    ]
    first_inserted_description = next(
        (description for description in inserted_descriptions if description), None
    )

    return FeedbackCard(
        type="extraneous_sound",
        caption=f"You tend to add {_format_list(inserted_labels)} sounds.",
        details=(
            f"An extra {_format_list(inserted_labels)} sound is getting added. "
            "Try to move directly between the surrounding sounds without inserting that extra sound. "
            f"{first_inserted_description.get('explanation', '') if first_inserted_description else ''}"
        ).strip(),
        words=_ordered_words(mistake),
        video=_first_video(*inserted_descriptions),
        frequency=mistake["frequency"],
        total_severity=mistake["total_severity"],
        target=mistake["target"],
        speech=inserted_phones,
        target_description=mistake["target_description"],
        speech_description=inserted_descriptions,
    )


def get_spoken_word_timestamps(
    phone_pairings_by_word: TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T,
) -> list[tuple[WORD_T, float, float]]:
    timestamps: list[tuple[WORD_T, float, float]] = []
    for word, pairings in phone_pairings_by_word:
        if len(pairings) == 0:
            timestamps.append((word, 0.0, 0.0))
            continue

        spoken_phones = [speech for _, speech in pairings]
        start = spoken_phones[0][1]
        end = spoken_phones[-1][2]
        timestamps.append((word, start, end))

    return timestamps


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


def build_analysis_response(
    phone_pairings_by_word: TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T,
    topk=3,
) -> AnalysisResponse:
    insertion_mistakes, _, _, mistakes_by_target = phonetic_errors(
        phone_pairings_by_word
    )
    combined_target_mistakes = _combine_target_mistakes(
        phone_pairings_by_word, mistakes_by_target
    )
    word_scores, average_score = score_words_cer(phone_pairings_by_word)
    spoken_word_timestamps = get_spoken_word_timestamps(phone_pairings_by_word)

    feedback_cards = [
        *[_build_target_card(mistake) for mistake in combined_target_mistakes],
        *[_build_insertion_card(mistake) for mistake in insertion_mistakes],
    ]
    feedback_cards = sorted(
        feedback_cards,
        key=lambda card: (
            -(card.get("frequency") or 0) - (card.get("total_severity") or 0)
        ),
    )[:topk]

    words = [
        AnalyzedWord(text=word, score=score, start=start, end=end)
        for (word, score), (_, start, end) in zip(word_scores, spoken_word_timestamps)
    ]

    return AnalysisResponse(
        words=words,
        average_score=average_score,
        spoken_word_timestamps=spoken_word_timestamps,
        feedback=feedback_cards,
    )


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
