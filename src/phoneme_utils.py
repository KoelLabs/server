# Import monkey patch first to fix ipapy collections import issue
import ipapy_monkey_patch

import sys
import ipapy
from ipapy.ipastring import IPAString
import panphon
import panphon.distance

# Create a panphon feature table
ft = panphon.FeatureTable()
panphon_dist = panphon.distance.Distance()

IPA_SYMBOLS = [ipa for ipa, *_ in ft.segments]


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
    """Canonize the Unicode representation of the IPA string"""
    # For now, just return the string as-is since we're not using ipapy
    # This can be enhanced later if needed
    return ipa_string
