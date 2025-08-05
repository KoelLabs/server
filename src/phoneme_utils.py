import numpy as np

import panphon
import panphon.distance

# Create a panphon feature table
ft = panphon.FeatureTable()
panphon_dist = panphon.distance.Distance()
inverse_double_weight_sum = 1 / (sum(ft.weights) * 2)

# Phoneme mapping for panphon compatibility
# Some IPA phonemes have multiple Unicode representations.
PANPHONE_MAPPINGS = {
    "ɝ": "ɜ˞",  # r-colored schwa (U+025D) -> schwa + r-coloring diacritic (U+025C + U+02DE)
    "ɚ": "ə˞",
    "g": "ɡ",  # model vocab has correct ɡ but we add this since its hard to distinguish
}
# just a few phonemes that are hard to distinguish, we mask them to the closest phoneme to improve scores and omitt unecessary feedback
PHONEMES_TO_MASK = {
    "ʌ": "ə",
    "ɔ": "ɑ",
    "kʰ": "k",
    "sʰ": "s",
}

ALL_MAPPINGS = {**PANPHONE_MAPPINGS, **PHONEMES_TO_MASK}


def fer(prediction, ground_truth):
    """
    Feature Error Rate: the edits weighted by their acoustic features summed up and divided by the length of the ground truth.
    """
    return (
        inverse_double_weight_sum
        * panphon_dist.weighted_feature_edit_distance(ground_truth, prediction)
        / len(ground_truth)
    )


# Convert a phoneme to a numerical feature vector
def phoneme_to_vector(phoneme):
    vectors = ft.word_to_vector_list(phoneme, numeric=True)
    if vectors:
        return np.array(vectors[0])  # Take the first vector if multiple exist
    else:
        raise ValueError(f"vector not found for phoneme: {phoneme}")


# Convert sequences of phonemes to sequences of vectors
def sequence_to_vectors(seq):
    return [phoneme_to_vector(p) for p in seq]


def weighted_substitution_cost(x, y):
    return -abs(panphon_dist.weighted_substitution_cost(x, y))


def weighted_insertion_cost(x):
    return -abs(panphon_dist.weighted_insertion_cost(x))


def weighted_deletion_cost(x):
    return -abs(panphon_dist.weighted_deletion_cost(x))


# ---- alignment functions ----
def needleman_wunsch(
    seq1,
    seq2,
    substitution_func=lambda x, y: 0 if x == y else -1,
    deletion_func=lambda _: -1,
    insertion_func=lambda _: -1,
):
    """Get aligned phoneme lists for target and speech phonemes (even for grouped phones like kʰ)
    example:
        target = ['l', 'o', 'o', 'o', 'o', 'o', 'n', 'ɡ', 'e', 'e', 'r']
        speech = ['s', 'h', 'o', 'r', 'r', 't']
    Example:
        target: ['l', 'o', 'o', 'o', 'o', 'o', 'n', 'ɡ', 'e', 'e', 'r']
        speech: ['-', '-', '-', 's', 'h', 'o', '-', '-', 'r', 'r', 't']
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


def weighted_needleman_wunsch(seq1, seq2, is_timestamp=False):
    """
    this function is needleman wunsch but weighted by feature error rate and can handle timestamped phonemes
    """
    if is_timestamp:
        vector_seq1 = sequence_to_vectors(
            [s[0] for s in seq1]
        )  # change the [(p, int, int)...] to [p, p...]
        vector_seq2 = sequence_to_vectors([s[0] for s in seq2])
    else:
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
