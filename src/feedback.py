import os
import json
from phoneme_utils import (
    fer,
    weighted_needleman_wunsch,
    map_timestamped_phonemes,
    map_phones_by_word,
    validate_target_data,
)

MODEL_VOCAB_JSON = os.path.join(os.path.dirname(__file__), "model_vocab_feedback.json")

with open(MODEL_VOCAB_JSON, "r", encoding="utf-8") as f:
    VOCAB_CONTENT = json.load(f)


def user_phonetic_errors(word_phone_pairings, topk=3):
    """
    Build a frequency dictionary of phoneme mistakes AND sorts by score
    Returns {target_phoneme: (mistake_frequency, timestamps_of_user_phoneme_mistakes, timestamps_of_target_phoneme_mistakes, timestamps_of_user_phrase_mistakes, timestamps_of_target_phrase_mistakes, mistake_severities, phoneme_spoken_as, words_with_mistake, score, target_phoneme_written_feedback, speech_phoneme_written_feedback)}
    example: {'aɪ': (1, [(2.405345211581292, 2.425389755011136)], [(2.4764454113924055, 2.4965791139240507)], [(2.385300668151448, 2.425389755011136)], [(2.27510838607595, 2.8187183544303798)], 14.5, {'-'}, 15.5), 'n': (1, [(1.583518930957...}
    """
    phoneme_mistake_freq = {}
    phoneme_written_feedback_dict = phoneme_written_feedback(word_phone_pairings)

    for idx, (word, pairs) in enumerate(word_phone_pairings):
        for target_phoneme_tuple, speech_phoneme_tuple in pairs:
            target_phoneme, target_start_time, target_end_time = target_phoneme_tuple
            speech_phoneme, speech_start_time, speech_end_time = speech_phoneme_tuple
            if target_phoneme != speech_phoneme and target_phoneme != "-":
                if target_phoneme not in phoneme_mistake_freq:
                    phoneme_mistake_freq[target_phoneme] = (
                        0,
                        [],
                        [],
                        [],
                        [],
                        0,
                        set(),
                        set(),
                        0.0,
                        None,
                        None,
                    )

                # update mistake count and word set and mistake severity and phoneme spoken as
                (
                    mistake_count,
                    user_error_timestamps,
                    target_error_timestamps,
                    user_phrase_error_timestamps,
                    target_phrase_error_timestamps,
                    mistake_severities,
                    phoneme_spoken_as,
                    words_with_mistake,
                    score,
                    target_phoneme_written_feedback,
                    speech_phoneme_written_feedback,
                ) = phoneme_mistake_freq[target_phoneme]
                user_error_timestamps.append((speech_start_time, speech_end_time))
                target_error_timestamps.append((target_start_time, target_end_time))

                prev_word_first_pair = word_phone_pairings[max(0, idx - 1)][1][0]
                prev_target_word_start_time = prev_word_first_pair[0][
                    1
                ]  # target phone, start time
                prev_speech_word_start_time = prev_word_first_pair[1][
                    1
                ]  # speech phone, start time

                post_word_last_pair = word_phone_pairings[
                    min(idx + 1, len(word_phone_pairings) - 1)
                ][1][-1]
                post_target_word_end_time = post_word_last_pair[0][
                    2
                ]  # target phone, end time
                post_speech_word_end_time = post_word_last_pair[1][
                    2
                ]  # speech phone, start time

                user_phrase_error_timestamps.append(
                    (prev_speech_word_start_time, post_speech_word_end_time)
                )
                target_phrase_error_timestamps.append(
                    (prev_target_word_start_time, post_target_word_end_time)
                )

                mistake_count += 1
                mistake_severities += fer(target_phoneme, speech_phoneme) / (
                    mistake_count
                )
                phoneme_spoken_as.add(speech_phoneme)
                score += mistake_severities + mistake_count
                words_with_mistake.add(word)
                target_phoneme_written_feedback = phoneme_written_feedback_dict.get(
                    target_phoneme
                )
                speech_phoneme_written_feedback = phoneme_written_feedback_dict.get(
                    speech_phoneme
                )
                phoneme_mistake_freq[target_phoneme] = (
                    mistake_count,
                    user_error_timestamps,
                    target_error_timestamps,
                    user_phrase_error_timestamps,
                    target_phrase_error_timestamps,
                    mistake_severities,
                    phoneme_spoken_as,
                    words_with_mistake,
                    score,
                    target_phoneme_written_feedback,
                    speech_phoneme_written_feedback,
                )

    sorted_phoneme_mistake_freq = sorted(
        phoneme_mistake_freq.items(), key=lambda x: x[1][8], reverse=True
    )
    topk_phoneme_mistake_freq = dict(sorted_phoneme_mistake_freq[:topk])
    return topk_phoneme_mistake_freq


def pair_by_words(target_timestamped, target_by_words, speech_timestamped):
    """this function pairs the target and speech by words
    Returns an array of tuples with the (word, [(target_phoneme_timestamped, speech_phoneme_timestamped), ...])
    target_phoneme_timestamped = (phoneme, start_time, end_time)
    speech_phoneme_timestamped = (phoneme, start_time, end_time)
    """
    speech_timestamped = map_timestamped_phonemes(speech_timestamped)
    target_timestamped = map_timestamped_phonemes(target_timestamped)
    target_by_words = map_phones_by_word(target_by_words)
    validate_target_data(target_timestamped, target_by_words)

    aligned_target, aligned_speech = weighted_needleman_wunsch(
        target_timestamped, speech_timestamped, is_timestamp=True
    )
    paired = zip(aligned_target, aligned_speech)

    pairing_by_words = []
    pairs = iter(paired)
    cur_pair = next(pairs)
    start = [(("-", 0.0, 0.0), ("-", 0.0, 0.0))]
    for word, phons in target_by_words:
        phons = list(phons)
        ps = start
        while len(phons) > 0:
            t, s = cur_pair
            # every phoneme from needleman in phons has exactly one match to a phoneme t
            if t[0] == phons[0]:
                phons.pop(0)
            # if t or s is silence, the timestamp will be the same as the previous timestamp or 0,0
            if t[0] == "-":
                t = ("-", ps[-1][0][1], ps[-1][0][2])
            if s[0] == "-":
                s = ("-", ps[-1][1][1], ps[-1][1][2])
            # if the user says extra phonemes (t[0] == "-"), we just append the pair as well
            ps.append((t, s))
            try:
                cur_pair = next(pairs)
            except StopIteration:
                break
        pairing_by_words.append((word, ps[1:]))  # skip the first appended silence token
        start = [ps[-1]]

    return pairing_by_words


def score_words_cer(word_phone_pairings):
    """
    This function scores the words based on the character error rate
    Returns a list of tuples with the (word, target_phoneme_sequence, speech_phoneme_sequence, score)
    """
    word_scores = []
    average_score = 0
    for word, pairs in word_phone_pairings:
        cer = sum(1 for t, s in pairs if t[0] != s[0]) / len(pairs)
        seq1 = "".join([t[0] for t, _ in pairs])
        seq2 = "".join([s[0] for _, s in pairs])
        score = 1.0 - cer
        word_scores.append((word, seq1, seq2, score))
        average_score += score
    average_score /= len(word_phone_pairings)
    return word_scores, average_score


def score_words_wfed(word_phone_pairings):
    """
    This function scores the words based on the weighted feature error rate
    Returns a list of tuples with the (word, target_phoneme_sequence, speech_phoneme_sequence, score)
    """
    word_scores = []
    average_score = 0
    for word, pairs in word_phone_pairings:
        seq1 = "".join([t[0] for t, _ in pairs])
        seq2 = "".join([s[0] for _, s in pairs])
        fer_score = fer(seq1, seq2)
        score = 1 - fer_score
        word_scores.append((word, seq1, seq2, score))
        average_score += score
    average_score /= len(word_phone_pairings)
    return word_scores, average_score


def phoneme_written_feedback(word_phone_pairings):
    """This function takes in the target and speech and returns a dictionary
    of phoneme: {explanation, phonetic spelling} for ALL phonemes in the target
    and speech.
    """
    all_phoneme_feedback = {}
    # get all the speech phonemes at once on word_phone_pairings, not sending over transcript, reduce network cost
    all_speech_phonemes = get_phone_pairing_vocab(word_phone_pairings)

    for phoneme in all_speech_phonemes:
        phoneme_feedback = next(
            (item for item in VOCAB_CONTENT if item["phoneme"] == phoneme), None
        )
        if phoneme_feedback:
            all_phoneme_feedback[phoneme] = {
                "explanation": phoneme_feedback["explanation"],
                "phonetic spelling": phoneme_feedback["phonetic spelling"],
                "video": phoneme_feedback["video"],
                "description": phoneme_feedback["description"],
                "examples": phoneme_feedback["examples"],
            }
        else:
            raise ValueError(f"Phoneme {phoneme} not found in model vocabulary")
    return all_phoneme_feedback


def get_phone_pairing_vocab(word_phone_pairings):
    """
    This function gets the vocab of all the phonemes in the word_phone_pairings
    Returns a set of phonemes
    """
    vocab = set()
    for word, pairs in word_phone_pairings:
        for target_phoneme, speech_phoneme in pairs:
            vocab.add(target_phoneme[0]) if target_phoneme[0] != "-" else None
            vocab.add(speech_phoneme[0]) if speech_phoneme[0] != "-" else None
    return vocab
