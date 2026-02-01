from typing import TypeVar
from collections.abc import Sequence

import numpy as np

from string import punctuation
from g2p_en import G2p

g2p = G2p()

import panphon
import panphon.distance

# Create a panphon feature table
_ft = panphon.FeatureTable()
_panphon_dist = panphon.distance.Distance()
_inverse_double_weight_sum = 1 / (sum(_ft.weights) * 2)

# Types
PHONE_T = str
WORD_T = str
TIMESTAMPED_PHONE_T = tuple[PHONE_T, float, float]
TIMESTAMPED_PHONES_T = list[TIMESTAMPED_PHONE_T]
TIMESTAMPED_PHONES_BY_WORD_T = list[tuple[WORD_T, TIMESTAMPED_PHONES_T]]
TIMESTAMPED_PHONE_PAIRINGS_T = list[tuple[TIMESTAMPED_PHONE_T, TIMESTAMPED_PHONE_T]]
TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T = list[tuple[WORD_T, TIMESTAMPED_PHONE_PAIRINGS_T]]

# Phoneme mapping for panphon compatibility
_PANPHONE_MAPPINGS = {
    "ɝ": "ɜ˞",
    "ɚ": "ə˞",
    "ŋ̍": "ŋ̩",
    "ĩ": "ɪ̰",
}
# Temporary simplification of similar phonemes
_PHONEMES_TO_MASK = {
    "ʌ": "ə",
    "ɔ": "ɑ",
    "kʰ": "k",
    "sʰ": "s",
}
_ALL_MAPPINGS = {**_PANPHONE_MAPPINGS, **_PHONEMES_TO_MASK}


def map_timestamped_phonemes(timestamped: TIMESTAMPED_PHONES_T) -> TIMESTAMPED_PHONES_T:
    return [(_ALL_MAPPINGS.get(p, p), s, e) for p, s, e in timestamped]


def map_phones_by_word(
    phones_by_words: TIMESTAMPED_PHONES_BY_WORD_T,
) -> TIMESTAMPED_PHONES_BY_WORD_T:
    return [
        (w, [(_ALL_MAPPINGS.get(p, p), s, e) for p, s, e in phs])
        for w, phs in phones_by_words
    ]


def fer(prediction, ground_truth):
    """
    Feature Error Rate: the edits weighted by their acoustic features summed up and divided by the length of the ground truth.
    """
    return (
        _inverse_double_weight_sum
        * _panphon_dist.weighted_feature_edit_distance(ground_truth, prediction)
        / len(ground_truth)
    )


def _phoneme_to_vector(phoneme):
    """Convert a phoneme to a numerical feature vector"""
    vectors = _ft.word_to_vector_list(phoneme, numeric=True)
    if vectors:
        return np.array(vectors[0])  # Take the first vector if multiple exist
    else:
        raise ValueError(f"vector not found for phoneme: {phoneme}")


def _weighted_substitution_cost(x, y):
    return -abs(_panphon_dist.weighted_substitution_cost(x, y))


def _weighted_insertion_cost(x):
    return -abs(_panphon_dist.weighted_insertion_cost(x))


def _weighted_deletion_cost(x):
    return -abs(_panphon_dist.weighted_deletion_cost(x))


# ---- alignment functions ----
T = TypeVar("T")
L = TypeVar("L")


def _needleman_wunsch(
    seq1: Sequence[T],
    seq2: Sequence[L],
    substitution_func=lambda x, y: 0 if x == y else -1,
    deletion_func=lambda _: -1,
    insertion_func=lambda _: -1,
) -> tuple[Sequence[T], Sequence[L]]:
    """
    Get aligned sequences using the Needleman-Wunsch algorithm
    Example:
        seq1 = ['l', 'o', 'o', 'o', 'o', 'o', 'n', 'ɡ', 'e', 'e', 'r']
        seq2 = ['s', 'h', 'o', 'r', 'r', 't']
        aligned_seq1, aligned_seq2 = needleman_wunsch(seq1, seq2)
      Outputs:
        aligned_seq1: ['l', 'o', 'o', 'o', 'o', 'o', 'n', 'ɡ', 'e', 'e', 'r']
        aligned_seq2: ['-', '-', '-', 's', 'h', 'o', '-', '-', 'r', 'r', 't']
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1))

    # Initialize DP table
    for i in range(n + 1):
        dp[i][0] = i * deletion_func(seq1[i - 1]) if i > 0 else 0
    for j in range(m + 1):
        dp[0][j] = j * insertion_func(seq2[j - 1]) if j > 0 else 0

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + substitution_func(seq1[i - 1], seq2[j - 1])
            delete = dp[i - 1][j] + deletion_func(seq1[i - 1])
            insert = dp[i][j - 1] + insertion_func(seq2[j - 1])
            dp[i][j] = max(match, delete, insert)

    # Traceback to get alignment
    i, j = n, m
    aligned_seq1, aligned_seq2 = [], []

    while i > 0 or j > 0:
        current = dp[i][j]
        if (
            i > 0
            and j > 0
            and current
            == dp[i - 1][j - 1] + substitution_func(seq1[i - 1], seq2[j - 1])
        ):
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and current == dp[i - 1][j] + insertion_func(seq1[i - 1]):
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1
        else:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

    return list(reversed(aligned_seq1)), list(reversed(aligned_seq2))


def grouped_weighted_needleman_wunsch(
    grouped_seq: TIMESTAMPED_PHONES_BY_WORD_T, other_seq: TIMESTAMPED_PHONES_T
) -> TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T:
    """
    :func:`needleman_wunsch` weighted by feature error rate applied to grouped sequences
    """

    flattened_seq = [
        (_phoneme_to_vector(phone[0]), phone, word_ix)
        for word_ix, (_, phones) in enumerate(grouped_seq)
        for phone in phones
    ]
    other = [(_phoneme_to_vector(phone[0]), phone) for phone in other_seq]

    aligned, other_aligned = _needleman_wunsch(
        flattened_seq,
        other,
        lambda x, y: _weighted_substitution_cost(list(x[0]), list(y[0])),
        lambda x: _weighted_deletion_cost(list(x[0])),
        lambda x: _weighted_insertion_cost(list(x[0])),
    )

    grouped: TIMESTAMPED_PHONE_PAIRINGS_BY_WORD_T = [
        (word, []) for word, _ in grouped_seq
    ]
    prev, other_prev = ("-", 0.0, 0.0), ("-", 0.0, 0.0)
    prev_word_ix = 0
    for val, other_val in zip(aligned, other_aligned):
        phone, word_ix = (val[1], val[2]) if val != "-" else (prev, prev_word_ix)
        other_phone = other_val[1] if other_val != "-" else other_prev
        grouped[word_ix][1].append((phone, other_phone))
        prev = ("-", phone[1], phone[2])
        other_prev = ("-", other_phone[1], other_phone[2])
        prev_word_ix = word_ix

    return grouped


def remove_punctuation(text):
    return "".join([c for c in text if c not in punctuation])


def remove_length_diacritics(ipa_string: str):
    """Remove length diacritics from the IPA string"""
    return "".join(c for c in ipa_string if c not in {"ː", "ˑ"})


def remove_tones_and_stress(ipa_string: str):
    """Remove tones and stress from the IPA string"""
    tones_and_stresss = ["˨˩h", "˧ʔ˥", "˧˨ʔ", "ˈ", "ˌ", "˥", "˧", "˨", "˩", "˦"]  # fmt: skip
    for tone in tones_and_stresss:
        ipa_string = ipa_string.replace(tone, "")
    return ipa_string


def remove_tie_marker(ipa_string: str):
    """Remove tie marker from the IPA string"""
    return "".join({"͡": ""}.get(c, c) for c in ipa_string)


ARPABET2IPA = {'AA':'ɑ','AE':'æ','AH':'ʌ','AO':'ɔ','IX':'ɨ','AW':'aʊ','AX':'ə','AXR':'ɚ','AY':'aɪ','EH':'ɛ','ER':'ɝ','EY':'eɪ','IH':'ɪ','IY':'i','OW':'oʊ','OY':'ɔɪ','UH':'ʊ','UW':'u','UX':'ʉ','B':'b','CH':'tʃ','D':'d','DH':'ð','EL':'l̩','EM':'m̩','EN':'n̩','F':'f','G':'ɡ','HH':'h','H':'h','JH':'dʒ','K':'k','L':'l','M':'m','N':'n','NG':'ŋ','NX':'ɾ̃','P':'p','Q':'ʔ','R':'ɹ','S':'s','SH':'ʃ','T':'t','TH':'θ','V':'v','W':'w','WH':'ʍ','Y':'j','Z':'z','ZH':'ʒ','DX':'ɾ'}  # fmt: skip
def arpabet2ipa(arpabet_symbols: list[str]):
    return [ARPABET2IPA["".join(i for i in x if i.isalpha())] for x in arpabet_symbols]


def english2ipa(text: str):
    arpa = g2p(text)

    return list(
        map(
            lambda phoneme: remove_punctuation(
                remove_length_diacritics(
                    remove_tones_and_stress(remove_tie_marker(phoneme))
                )
            ),
            arpabet2ipa(arpa),
        )
    )
