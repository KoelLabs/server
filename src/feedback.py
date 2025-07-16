import panphon
import panphon.distance
from panphon.distance import Distance
import numpy as np
import json
import os
from phoneme_utils import group_phonemes, get_fastdtw_aligned_phoneme_lists, fer

# Create a panphon feature table
ft = panphon.FeatureTable()
# Create a distance object for reuse
dist = Distance()

model_vocab_json = os.path.join(os.path.dirname(__file__), "model_vocab_feedback.json")


def user_phonetic_errors(target, target_by_words, speech, topk=3):
    """
    Build a frequency dictionary of phoneme mistakes.
    Returns {target_phoneme: (mistake_frequency, words_with_mistake, mistake_severities, phoneme_spoken_as)}
    example: {'k': (3, {1, 2}, 4.8, {'l', 'i'}, 8.4), 'o': (1, {5}, .9, {'i', 'e'})}
    """
    if len(speech) == 0:
        return {}

    word_phone_pairings = pair_by_words(target, target_by_words, speech)
    phoneme_mistake_freq = {}

    for word_idx, (word, pairs) in enumerate(word_phone_pairings):
        for target_phoneme, speech_phoneme, _ in pairs:
            if target_phoneme != speech_phoneme:
                if target_phoneme not in phoneme_mistake_freq:
                    phoneme_mistake_freq[target_phoneme] = (0, set(), 0.0, set(), 0.0)

                # update mistake count and word set and mistake severity and phoneme spoken as
                (
                    mistake_count,
                    word_set,
                    mistake_severities,
                    phoneme_spoken_as,
                    score,
                ) = phoneme_mistake_freq[target_phoneme]
                word_set.add(word_idx)
                mistake_count += 1
                mistake_severities += dist.feature_error_rate(
                    target_phoneme, speech_phoneme
                ) / (mistake_count)
                phoneme_spoken_as.add(speech_phoneme)
                score += mistake_severities + mistake_count

                phoneme_mistake_freq[target_phoneme] = (
                    mistake_count,
                    word_set,
                    mistake_severities,
                    phoneme_spoken_as,
                    score,
                )
    sorted_phoneme_mistake_freq = sorted(
        phoneme_mistake_freq.items(), key=lambda x: x[1][4], reverse=True
    )
    topk_phoneme_mistake_freq = dict(sorted_phoneme_mistake_freq[:topk])
    return topk_phoneme_mistake_freq


def pair_by_words(target, target_by_words, speech):
    """this function pairs the target and speech by words
    Returns an array of tuples of words where the index represents the word: (target_phoneme, speech_phoneme) tuples.
    example: [calling: [('ɔ', 'o'), ('l', 'i'), ('i', 'i'), ('ŋ', 'ŋ')], ...]
    """
    # Retrieve aligned sequences along with the speech-side indices so that we can
    # later map any phoneme back to its timestamp without extra scans.
    aligned_target, aligned_speech, aligned_idx = get_fastdtw_aligned_phoneme_lists(
        target, speech
    )

    paired = zip(aligned_target, aligned_speech, aligned_idx)

    pair_by_words = []
    pairs = iter(paired)
    cur_pair = next(pairs)
    start = []
    for word, phons in target_by_words:
        phons = list(phons)
        ps = start
        while len(phons) > 0:
            t, s, _ = cur_pair
            if t != phons[0]:
                phons.pop(0)
            ps.append(cur_pair)
            try:
                cur_pair = next(pairs)
            except StopIteration:
                break
        pair_by_words.append((word, ps[:-1]))
        start = [ps[-1]]
    # pair_by_words.append(start)

    return pair_by_words


def pairs_to_phonetic_phrases(pairs):
    """this function takes in a list of pairs and returns a list of phonetic phrases
    example: [('ɔ', 'o'), ('l', 'i'), ('i', 'i'), ('ŋ', 'ŋ')] -> ['ɔlilŋ', 'olilŋ', 'ɔlilŋ', 'olilŋ']
    """
    phonetic_phrases = []
    for pair in pairs:
        phonetic_phrases.append(pair[0] + pair[1])
    return phonetic_phrases


def score_words_cer(target, target_by_words, speech):
    if len(speech) == 0:
        return [[word, seq, "", 0] for word, seq in target_by_words], 0

    pbw = pair_by_words(target, target_by_words, speech)
    word_scores = []
    average_score = 0
    for word, pairs in pbw:
        cer = sum(1 for t, s, _ in pairs if t != s) / len(pairs)
        seq1 = "".join([t for t, _, _ in pairs])
        seq2 = "".join([s for _, s, _ in pairs])
        word_scores.append((word, seq1, seq2, (1 - cer / 2)))
        average_score += 1 - cer / 2
    average_score /= len(pbw)
    return word_scores, average_score


def score_words_wfed(target, target_by_words, speech):
    if len(speech) == 0:
        return [[word, seq, "", 0] for word, seq in target_by_words], 0

    pbw = pair_by_words(target, target_by_words, speech)
    word_scores = []
    average_score = 0
    for word, pairs in pbw:
        seq1 = "".join([t for t, _, _ in pairs])
        seq2 = "".join([s for _, s, _ in pairs])
        norm_score = (22 - fer(seq1, seq2)) / 22
        word_scores.append((word, seq1, seq2, norm_score**2))
        average_score += norm_score**2
    average_score /= len(pbw)
    return word_scores, average_score


def get_phoneme_feedback(phoneme):
    with open(model_vocab_json, "r", encoding="utf-8") as f:
        content = json.load(f)
    phoneme_feedback = next(
        (item for item in content if item["phoneme"] == phoneme), None
    )
    return phoneme_feedback


def phoneme_written_feedback(target, speech):
    """This function takes in the target and speech and returns a dictionary
    of phoneme: {explanation, phonetic-spelling} for ALL phonemes in the target
    and speech.
    """
    all_phoneme_feedback = {}
    all_speech_phonemes = list(set(target + speech))
    with open(model_vocab_json, "r", encoding="utf-8") as f:
        content = json.load(f)
    for phoneme in all_speech_phonemes:
        phoneme_feedback = next(
            (item for item in content if item["phoneme"] == phoneme), None
        )
        if phoneme_feedback:
            all_phoneme_feedback[phoneme] = {
                "explanation": phoneme_feedback["explanation"],
                "phonetic-spelling": phoneme_feedback["phonetic-spelling"],
                "video": phoneme_feedback["video"],
                "description": phoneme_feedback["description"],
                "example": phoneme_feedback["example"],
            }
        else:
            raise ValueError(f"Phoneme {phoneme} not found in model vocabulary")
    return all_phoneme_feedback
