import panphon
import panphon.distance
from panphon.distance import Distance
import numpy as np
import json
import os
from phoneme_utils import group_phonemes, get_fastdtw_aligned_phoneme_lists

# Create a panphon feature table
ft = panphon.FeatureTable()
# Create a distance object for reuse
dist = Distance()

model_vocab_json = os.path.join(os.path.dirname(__file__), "model_vocab_feedback.json")


def get_all_phoneme_pairs(target, target_by_words, speech):
    """
    Get all phoneme pairs from the alignment for building frequency dictionaries.
    Returns a dictionary of word: (target_phoneme, speech_phoneme) tuples.
    example: {'calling': [('ɔ', 'o'), ('l', 'i'), ('i', 'i'), ('ŋ', 'ŋ')], ...}
    """
    if len(speech) == 0:
        return {}
    
    pbw = pair_by_words(target, target_by_words, speech)
    all_pairs = {}
    
    for word, pairs in pbw:
        all_pairs[word] = pairs
    
    return all_pairs


def user_phonetic_errors(target, target_by_words, speech):
    """
    Build a frequency dictionary of phoneme mistakes.
    Returns {target_phoneme: (mistake_frequency, words_with_mistake, mistake_severities, phoneme_spoken_as)}
    example: {'k': (3, {'calling', 'crack'}, [2.3, .6], {'l', 'i'}), 'o': (1, {'cooler'}, [.3, .4], {'i', 'e'})}
    """
    if len(speech) == 0:
        return {}
    
    word_phone_pairings = get_all_phoneme_pairs(target, target_by_words, speech)
    phoneme_mistake_freq = {}
    print("word_phone_pairings", word_phone_pairings)
    for word, pairs in word_phone_pairings.items():
        for target_phoneme, speech_phoneme in pairs:
            if target_phoneme != speech_phoneme: 

                if target_phoneme not in phoneme_mistake_freq:
                    phoneme_mistake_freq[target_phoneme] = (0, set(), [], set())

                # update mistake count and word set and mistake severity and phoneme spoken as
                mistake_count, word_set, mistake_severities, phoneme_spoken_as = phoneme_mistake_freq[target_phoneme]
                word_set.add(word)
                mistake_severities.append(dist.feature_error_rate(target_phoneme, speech_phoneme))
                phoneme_spoken_as.add(speech_phoneme)
                phoneme_mistake_freq[target_phoneme] = (mistake_count + 1, word_set, mistake_severities, phoneme_spoken_as)

    return phoneme_mistake_freq



def pair_by_words(target, target_by_words, speech):
    ''' this function pairs the target and speech by words 
    example: [('calling', [('k', 'k'), ('ɔ', 'o'), ('l', 'l'), ('ɪ', 'i'), ('ŋ', 'ŋ')]), ...]'''
    # Get aligned phoneme lists with needleman wunsch
    aligned_target, aligned_speech = get_fastdtw_aligned_phoneme_lists(target, speech)
    
    if not aligned_target or not aligned_speech:
        return []
    
    # Now pair by words
    pair_by_words = []
    target_idx = 0
    
    for word, phons in target_by_words:
        # Group the phonemes for this word
        word_phonemes = group_phonemes(phons)
        word_pairs = []
        
        # Find pairs for this word
        for i in range(len(aligned_target)):
            if target_idx < len(aligned_target):
                t, s = aligned_target[target_idx], aligned_speech[target_idx]
                word_pairs.append((t, s))
                target_idx += 1
                
                # Check if we've processed all phonemes for this word
                if len(word_pairs) >= len(word_phonemes):
                    break
        
        pair_by_words.append((word, word_pairs))
    
    return pair_by_words


def score_words_cer(target, target_by_words, speech):
    if len(speech) == 0:
        return [[word, seq, "", 0] for word, seq in target_by_words], 0

    pbw = pair_by_words(target, target_by_words, speech)
    word_scores = []
    average_score = 0
    for word, pairs in pbw:
        cer = sum(1 for t, s in pairs if t != s) / len(pairs)
        seq1, seq2 = map(lambda x: "".join(x), zip(*pairs))
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
        seq1, seq2 = map(lambda x: "".join(x), zip(*pairs))
        norm_score = (
            22 - dist.weighted_feature_edit_distance(seq1, seq2)
        ) / 22
        word_scores.append((word, seq1, seq2, norm_score**2))
        average_score += norm_score**2
    average_score /= len(pbw)
    return word_scores, average_score

def phoneme_written_feedback(target, speech):
    ''' This function takes in the target and speech and returns a dictionary 
    of phoneme: {explanation, phonetic-spelling} for all phonemes in the target
    and speech. 
    '''
    all_phoneme_feedback = {}
    target_user_speech = group_phonemes(target + speech)
    with open(model_vocab_json, "r", encoding="utf-8") as f:
        content = json.load(f)
    for phoneme in target_user_speech: 
        phoneme_feedback = next((item for item in content if item["phoneme"] == phoneme), None)
        if phoneme_feedback:
            all_phoneme_feedback[phoneme] = {
                "explanation": phoneme_feedback["explanation"],
                "phonetic-spelling": phoneme_feedback["phonetic-spelling"],
            }
        else:
            print(f"Phoneme {phoneme} not found in model vocabulary")
    
    return all_phoneme_feedback


def feedback(target, target_by_words, speech, good_enough_threshold=0.4):
    if len(speech) == 0:
        return (
            [[word, "", "You didn't say anything!", 0] for word, _ in target_by_words],
            [],
            0,
        )
    pbw = pair_by_words(target, target_by_words, speech)
    word_feedbacks = []
    for word, pairs in pbw:
        wrongest_pair = pairs[0]
        wrongest_pair_dist = dist.weighted_feature_edit_distance(
            wrongest_pair[0], wrongest_pair[1]
        )
        for p in pairs:
            dist = dist.weighted_feature_edit_distance(
                p[0], p[1]
            )
            if dist > wrongest_pair_dist:
                wrongest_pair = p
                wrongest_pair_dist = dist
        if wrongest_pair_dist < good_enough_threshold:
            word_feedbacks.append(
                (
                    word,
                    'Your pronunciation of "' + word + '" is perfect!',
                    wrongest_pair_dist,
                )
            )
        else:
            target, speech = wrongest_pair
            t, s = sound_descriptions[target], sound_descriptions[speech]
            word_feedbacks.append(
                (
                    word,
                    f"""
                The actor made the '{t['phonemicSpelling']}' sound in "{word}" but you made the '{s['phonemicSpelling']}' sound.
                It is supposed to be {t['description'][0].lower() + t['description'][1:]}
                {t["exampleWord"]}
                """.strip(),
                    wrongest_pair_dist,
                )
            )
    top3 = sorted(word_feedbacks, key=lambda x: x[2], reverse=True)[:3]
    
    return word_feedbacks, top3


