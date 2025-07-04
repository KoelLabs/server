import panphon
import panphon.distance
from fastdtw import fastdtw
import numpy as np
from scipy.spatial.distance import euclidean
import json
import os

# Create a panphon feature table
ft = panphon.FeatureTable()


def load_sound_descriptions():
    """Load sound descriptions from the JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), "model_vocab_feedback.json")
    
    # Add a silence entry for special cases
    sound_descriptions = {
        "silence": {
            "phonemicSpelling": "silence",
            "description": "Mouth closed.",
            "exampleWord": "",
            "example_words": ["silence"],
        }
    }
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            phoneme = item.get("phoneme", "")
            if phoneme:
                # Convert JSON format to the expected format
                sound_descriptions[phoneme] = {
                    "phonemicSpelling": item.get("phonetic-spelling", ""),
                    "description": item.get("explanation", ""),
                    "exampleWord": item.get("example", []),
                    "example_words": item.get("example", [])
                }
    except Exception as e:
        print(f"Warning: Could not load sound descriptions from {json_path}: {e}")
    
    return sound_descriptions


# Load sound descriptions
sound_descriptions = load_sound_descriptions()


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


def fastdtw_phoneme_alignment(seq1, seq2):
    # Convert phoneme sequences to feature vector sequences
    seq1_vectors = sequence_to_vectors(seq1)
    seq2_vectors = sequence_to_vectors(seq2)

    if not seq1_vectors or not seq2_vectors:
        raise ValueError(
            "One or both sequences could not be converted to feature vectors."
        )

    # Use FastDTW with Euclidean distance on the vectors
    distance, path = fastdtw(seq1_vectors, seq2_vectors, dist=euclidean)

    # Align the original phoneme sequences based on the path
    aligned_seq1 = []
    aligned_seq2 = []
    for i, j in path:
        aligned_seq1.append(seq1[i] if i < len(seq1) else "-")
        aligned_seq2.append(seq2[j] if j < len(seq2) else "-")

    return "".join(aligned_seq1), "".join(aligned_seq2)


def pair_by_words(target, target_by_words, speech):
    paired = zip(*fastdtw_phoneme_alignment(target, speech))

    pair_by_words = []
    pairs = iter(paired)
    cur_pair = next(pairs)
    start = []
    for word, phons in target_by_words:
        phons = list(phons)
        ps = start
        while len(phons) > 0:
            t, s = cur_pair
            if t != phons[0]:
                phons.pop(0)
            ps.append(cur_pair)
            try:
                cur_pair = next(pairs)
            except StopIteration:
                break
        pair_by_words.append((word, ps[:-1]))
        start = [ps[-1]]

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
            22 - panphon.distance.Distance().weighted_feature_edit_distance(seq1, seq2)
        ) / 22
        word_scores.append((word, seq1, seq2, norm_score**2))
        average_score += norm_score**2
    average_score /= len(pbw)
    return word_scores, average_score


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
        wrongest_pair_dist = panphon.distance.Distance().weighted_feature_edit_distance(
            wrongest_pair[0], wrongest_pair[1]
        )
        for p in pairs:
            dist = panphon.distance.Distance().weighted_feature_edit_distance(
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


def side_by_side_description(target, target_by_words, speech):
    if len(speech) == 0:
        return [
            [
                word,
                [
                    (sound_descriptions[t], sound_descriptions["silence"])
                    for t in phonemes
                ],
            ]
            for word, phonemes in target_by_words
        ]

    pbw = pair_by_words(target, target_by_words, speech)
    vis = []
    for word, pairs in pbw:
        st = []
        for t, s in pairs:
            st.append(
                (
                    sound_descriptions[t],
                    sound_descriptions[s],
                )
            )
        vis.append((word, st))
    vis[0] = (
        vis[0][0],
        [
            (
                {
                    "phonemicSpelling": "silence",
                    "description": "Mouth closed.",
                    "exampleWord": "",
                    "example_words": ["silence"],
                },
                {
                    "phonemicSpelling": "silence",
                    "description": "Mouth closed.",
                    "exampleWord": "",
                    "example_words": ["silence"],
                },
            ),
            *vis[0][1],
        ],
    )
    return vis
