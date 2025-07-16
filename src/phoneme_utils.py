# Conversion between different phonetic codes
# Modified from https://github.com/jhasegaw/phonecodes/blob/master/src/phonecodes.py

import sys

import panphon
import panphon.distance
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Create a panphon feature table
ft = panphon.FeatureTable()
panphon_dist = panphon.distance.Distance()
inverse_double_weight_sum = 1 / (sum(ft.weights) * 2)

IPA_SYMBOLS = [ipa for ipa, *_ in ft.segments]

# Phoneme mapping for panphon compatibility
# Some IPA phonemes have multiple Unicode representations. This maps them to
# the specific forms that panphon recognizes for proper feature vector generation.
PHONEME_MAPPINGS = {
    "ɝ": "ɜ˞",  # r-colored schwa (U+025D) -> schwa + r-coloring diacritic (U+025C + U+02DE)
    "ɚ": "ə˞",
}


def fer(prediction, ground_truth):
    """
    Feature Error Rate: the edits weighted by their acoustic features summed up and divided by the length of the ground truth.
    """
    return (
        inverse_double_weight_sum
        * panphon_dist.weighted_feature_edit_distance(ground_truth, prediction)
        / len(ground_truth)
    )


def map_phoneme_for_panphon(phoneme_string):
    """Map phonemes (or lists of phonemes) to their panphon-compatible forms.

    The original implementation returned a *list* of characters for an input
    string.  That made downstream code think a *single* phoneme was a list and
    therefore unhashable (e.g. when used as dict keys).  We now:

    1. If the input is a *string* representing one phoneme, replace any
       characters using `PHONEME_MAPPINGS` and return the *string*.
    2. If the input is a *list* (e.g. an existing list of phoneme strings), we
       map each element individually and return a *list* of strings – keeping
       the original container type intact.
    """

    return [PHONEME_MAPPINGS.get(ch, ch) for ch in phoneme_string]


# Convert a phoneme to a numerical feature vector
def phoneme_to_vector(phoneme):
    vectors = ft.word_to_vector_list(phoneme, numeric=True)
    if vectors:
        return np.array(vectors[0])  # Take the first vector if multiple exist
    return None  # Invalid phoneme


# Convert sequences of phonemes to sequences of vectors
def sequence_to_vectors(seq):
    vectors = []
    for p in seq:
        vec = phoneme_to_vector(p)
        vectors.append(vec)
    return vectors


def weighted_substitution_cost(x, y):
    return -abs(panphon_dist.weighted_substitution_cost(x, y))


def weighted_insertion_cost(x):
    return -abs(panphon_dist.weighted_insertion_cost(x))


def weighted_deletion_cost(x):
    return -abs(panphon_dist.weighted_deletion_cost(x))


# ---- alignment functions ----


def phrase_bounds(word_phone_pairings, timestamps, word_idx):
    """Return (start_time, end_time) for the three-word window around idx_word."""
    left_pairs = word_phone_pairings[max(0, word_idx - 1)][1]
    right_pairs = word_phone_pairings[min(len(word_phone_pairings) - 1, word_idx + 1)][
        1
    ]

    first_idx = left_pairs[0][2]  # speech index of first phoneme
    last_idx = right_pairs[-1][2]  # speech index of last phoneme
    start_time = timestamps[first_idx][1]  # timestamps = (phoneme, start, end)
    end_time = timestamps[last_idx][2]
    return start_time, end_time


def get_fastdtw_aligned_phoneme_lists(target, speech):
    """Get aligned phoneme lists for target and speech phonemes (even for grouped phones like kʰ)
    example:
    target = "loooonger"
    speech = "short"
    returns 3 parallel lists of equal length:
        aligned_target   – phonemes from the target
        aligned_speech   – phonemes from the speech sequence (with "-" for gaps)
        aligned_s_idx    – original index of the speech phoneme in the un-aligned sequence (None for gaps)
    Example:
        aligned_targets = (
            ['l', 'o', 'o', 'o', 'o', 'o', 'n', 'g', 'e', 'e'],
            ['s', 'h', 'o', 'o', 'o', 'o', 'r', 'r', 'r', 't'],
            [ 0 ,  1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8 ,  9 ]
        )
    """

    # Use FastDTW to get the alignment path
    target_vectors = sequence_to_vectors(target)
    speech_vectors = sequence_to_vectors(speech)

    if not target_vectors or not speech_vectors:
        # When either list is empty, return three empty lists so caller logic remains consistent
        return [], [], []

    distance, path = fastdtw(target_vectors, speech_vectors, dist=euclidean)

    # Create aligned sequences based on the path
    aligned_target = []
    aligned_speech = []
    aligned_s_idx = []
    for i, j in path:
        aligned_target.append(target[i] if i < len(target) else "-")
        aligned_speech.append(speech[j] if j < len(speech) else "-")
        aligned_s_idx.append(j if j < len(speech) else None)
    print(aligned_target)
    print(aligned_speech)
    print(aligned_s_idx)
    return aligned_target, aligned_speech, aligned_s_idx


def needleman_wunsch(
    seq1,
    seq2,
    substitution_func=lambda x, y: 0 if x == y else -1,
    deletetion_func=lambda _: -1,
    insertion_func=lambda _: -1,
):
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1))

    # Initialize DP table
    for i in range(n + 1):
        dp[i][0] = i * deletetion_func(seq1[i - 1]) if i > 0 else 0
    for j in range(m + 1):
        dp[0][j] = j * insertion_func(seq2[j - 1]) if j > 0 else 0

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + substitution_func(seq1[i - 1], seq2[j - 1])
            delete = dp[i - 1][j] + deletetion_func(seq1[i - 1])
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


def weighted_needleman_wunsch(seq1, seq2):
    vector_seq1 = sequence_to_vectors(seq1)
    vector_seq2 = sequence_to_vectors(seq2)
    aligned_seq1, aligned_seq2 = needleman_wunsch(
        [(s, v) for s, v in zip(seq1, vector_seq1)],
        [(s, v) for s, v in zip(seq2, vector_seq2)],
        lambda x, y: weighted_substitution_cost(list(x[1]), list(y[1])),
        lambda x: weighted_deletion_cost(list(x[1])),
        lambda x: weighted_insertion_cost(list(x[1])),
    )
    return [s if s == "-" else s[0] for s in aligned_seq1], [
        s if s == "-" else s[0] for s in aligned_seq2
    ]


def main(args):
    if len(args) < 3:
        print(
            "Usage: python ./scripts/forced_alignment/needleman_wunsch.py <weighted|unweighted> <seq1> <seq2>"
        )
        return
    weighted = args[0] == "weighted"
    seq1 = args[1]
    seq2 = args[2]
    if weighted:
        aligned_seq1, aligned_seq2 = weighted_needleman_wunsch(seq1, seq2)
    else:
        aligned_seq1, aligned_seq2 = needleman_wunsch(seq1, seq2)
    aligned_seq1 = "".join(aligned_seq1)
    aligned_seq2 = "".join(aligned_seq2)
    print(aligned_seq1)
    print(aligned_seq2)


if __name__ == "__main__":
    main(sys.argv[1:])
