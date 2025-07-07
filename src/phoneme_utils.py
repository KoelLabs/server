# Import monkey patch first to fix ipapy collections import issue
import ipapy_monkey_patch

import sys
import ipapy
from ipapy.ipastring import IPAString
import panphon
import panphon.distance
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Create a panphon feature table
ft = panphon.FeatureTable()
panphon_dist = panphon.distance.Distance()

IPA_SYMBOLS = [ipa for ipa, *_ in ft.segments]


# Convert a phoneme to a numerical feature vector
def phoneme_to_vector(phoneme):
    vectors = ft.word_to_vector_list(phoneme, numeric=True)
    if vectors:
        return np.array(vectors[0])  # Take the first vector if multiple exist
    else:
        return None  # Invalid phoneme


# Convert sequences of phonemes to sequences of vectors
def sequence_to_vectors(seq):
    return [phoneme_to_vector(p) for p in seq if phoneme_to_vector(p) is not None]


def weighted_substitution_cost(x, y):
    return -abs(panphon_dist.weighted_substitution_cost(x, y))


def weighted_insertion_cost(x):
    return -abs(panphon_dist.weighted_insertion_cost(x))


def weighted_deletion_cost(x):
    return -abs(panphon_dist.weighted_deletion_cost(x))


def group_phonemes(phoneme_string):
    """
    Groups IPA characters into proper phoneme units.

    Args:
        phoneme_string (str): A string of IPA characters

    Returns:
        list: A list of grouped phonemes

    Example:
        group_phonemes("kʰɔlɪŋkʰɑɹdzɔɹðəweɪvʌvðəfjutʃɝ")
        # Returns: ['kʰ', 'ɔ', 'l', 'ɪ', 'ŋ', 'kʰ', 'ɑ', 'ɹ', 'd͡z', 'ɔ', 'ɹ', 'ð', 'ə', 'w', 'e', 'ɪ', 'v', 'ʌ', 'v', 'ð', 'ə', 'f', 'j', 'u', 't͡ʃ', 'ɝ']
    """
    if not is_valid_ipa(phoneme_string):
        print(
            f"Warning: removing invalid ipa characters from {phoneme_string}.",
            file=sys.stderr,
        )
    return string2symbols(canonize(phoneme_string, ignore=True), IPA_SYMBOLS)[0]


def is_valid_ipa(ipa_string):
    """Check if the given Unicode string is a valid IPA string"""
    for c in ipa_string:
        if not ipapy.is_valid_ipa(c):
            print(f"Invalid IPA character: {c}")
    return ipapy.is_valid_ipa(ipa_string)


def string2symbols(string, symbols):
    """
    Converts a string of symbols into a list of symbols, minimizing the number of untranslatable symbols,
    then minimizing the number of translated symbols.
    """
    N = len(string)
    symcost = 1  # path cost per translated symbol
    oovcost = len(string)  # path cost per untranslatable symbol
    maxsym = max(len(k) for k in symbols)  # max input symbol length
    # (pathcost to s[(n-m):n], n-m, translation[s[(n-m):m]], True/False)
    lattice = [(0, 0, "", True)]
    for n in range(1, N + 1):
        # Initialize on the assumption that s[n-1] is untranslatable
        lattice.append((oovcost + lattice[n - 1][0], n - 1, string[(n - 1) : n], False))
        # Search for translatable sequences s[(n-m):n], and keep the best
        for m in range(1, min(n + 1, maxsym + 1)):
            if (
                string[(n - m) : n] in symbols
                and symcost + lattice[n - m][0] < lattice[n][0]
            ):
                lattice[n] = (
                    symcost + lattice[n - m][0],
                    n - m,
                    string[(n - m) : n],
                    True,
                )
    # Back-trace
    tl = []
    translated = []
    n = N
    while n > 0:
        tl.append(lattice[n][2])
        translated.append(lattice[n][3])
        n = lattice[n][1]
    return (tl[::-1], translated[::-1])


def canonize(ipa_string, ignore=False):
    """canonize the Unicode representation of the IPA string"""
    return str(
        IPAString(unicode_string=ipa_string, ignore=ignore).canonical_representation
    )


# ---- alignment functions ----


def get_fastdtw_aligned_phoneme_lists(target, speech):
    """Get aligned phoneme lists for target and speech phonemes (even for grouped phones like kʰ)
    example:
    target = "loooonger"
    speech = "short"
    returns:
    aligned_targets = (['l', 'o', 'o', 'o', 'o', 'o', 'n', 'g', 'e', 'e'], ['s', 'h', 'o', 'o', 'o', 'o', 'r', 'r', 'r', 't'])
    """
    target_phonemes = group_phonemes(target)
    speech_phonemes = group_phonemes(speech)

    # Use FastDTW to get the alignment path
    target_vectors = sequence_to_vectors(target_phonemes)
    speech_vectors = sequence_to_vectors(speech_phonemes)

    if not target_vectors or not speech_vectors:
        return [], []

    distance, path = fastdtw(target_vectors, speech_vectors, dist=euclidean)

    # Create aligned sequences based on the path
    aligned_target = []
    aligned_speech = []
    for i, j in path:
        aligned_target.append(target_phonemes[i] if i < len(target_phonemes) else "-")
        aligned_speech.append(speech_phonemes[j] if j < len(speech_phonemes) else "-")

    return aligned_target, aligned_speech


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
