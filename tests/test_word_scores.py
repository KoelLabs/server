import json
import urllib.parse

TARGET = ['kʰ', 'ɔ', 'l', 'ɪ', 'ŋ', 'kʰ', 'ɑ', 'ɹ', 'd', 'd', 'z', 'ɑ', 'ɹ', 'ð', 'ə', 'w', 'eɪ', 'v', 'ə', 'v', 'ð', 'ə', 'f', 'j', 'u', 'tʃ', 'ɜ˞']  # fmt: skip
TARGET_BY_WORDS = [
    ["calling", ["kʰ", "ɔ", "l", "ɪ", "ŋ"]],
    ["cards", ["kʰ", "ɑ", "ɹ", "d", "z"]],
    ["are", ["ɑ", "ɹ"]],
    ["the", ["ð", "ə"]],
    ["wave", ["w", "eɪ", "v"]],
    ["of", ["ə", "v"]],
    ["the", ["ð", "ə"]],
    ["future", ["f", "j", "u", "tʃ", "ɜ˞"]],
]


def test_score_words_cer(client):
    # Use the same phonemes as target for a perfect match
    speech = [
        "kʰ",
        "ɔ",
        "l",
        "ɪ",
        "ŋ",
        "kʰ",
        "ɑ",
        "ɹ",
        "d",
        "d",
        "z",
        "ɑ",
        "ɹ",
        "ð",
        "ə",
        "w",
        "eɪ",
        "v",
        "ə",
        "v",
        "ð",
        "ə",
        "f",
        "k",
        "u",
        "t",
        "ɜ˞",
    ]

    # Hardcoded timestamps for speech (user's speech)
    speech_timestamped = [
        ("kʰ", 0.1, 0.12),
        ("ɔ", 0.12, 0.14),
        ("l", 0.14, 0.16),
        ("ɪ", 0.16, 0.18),
        ("ŋ", 0.18, 0.2),
        ("kʰ", 0.2, 0.22),
        ("ɑ", 0.22, 0.24),
        ("ɹ", 0.24, 0.26),
        ("d", 0.26, 0.28),
        ("d", 0.28, 0.3),
        ("z", 0.3, 0.32),
        ("ɑ", 0.32, 0.34),
        ("ɹ", 0.34, 0.36),
        ("ð", 0.36, 0.38),
        ("ə", 0.38, 0.4),
        ("w", 0.4, 0.42),
        ("eɪ", 0.42, 0.44),
        ("v", 0.44, 0.46),
        ("ə", 0.46, 0.48),
        ("v", 0.48, 0.5),
        ("ð", 0.5, 0.52),
        ("ə", 0.52, 0.54),
        ("f", 0.54, 0.56),
        ("k", 0.56, 0.58),
        ("u", 0.58, 0.6),
        ("t", 0.6, 0.62),
        ("ɜ˞", 0.62, 0.64),
    ]

    # Hardcoded timestamps for target (actor's speech)
    target_timestamped = [
        ("kʰ", 0.05, 0.07),
        ("ɔ", 0.07, 0.09),
        ("l", 0.09, 0.11),
        ("ɪ", 0.11, 0.13),
        ("ŋ", 0.13, 0.15),
        ("kʰ", 0.15, 0.17),
        ("ɑ", 0.17, 0.19),
        ("ɹ", 0.19, 0.21),
        ("d", 0.21, 0.23),
        ("d", 0.23, 0.25),
        ("z", 0.25, 0.27),
        ("ɑ", 0.27, 0.29),
        ("ɹ", 0.29, 0.31),
        ("ð", 0.31, 0.33),
        ("ə", 0.33, 0.35),
        ("w", 0.35, 0.37),
        ("eɪ", 0.37, 0.39),
        ("v", 0.39, 0.41),
        ("ə", 0.41, 0.43),
        ("v", 0.43, 0.45),
        ("ð", 0.45, 0.47),
        ("ə", 0.47, 0.49),
        ("f", 0.49, 0.51),
        ("j", 0.51, 0.53),
        ("u", 0.53, 0.55),
        ("tʃ", 0.55, 0.57),
        ("ɜ˞", 0.57, 0.59),
    ]

    # First, get word phone pairings
    target_timestamped_param = urllib.parse.quote(json.dumps(target_timestamped))
    target_by_words_param = urllib.parse.quote(json.dumps(TARGET_BY_WORDS))
    speech_timestamped_param = urllib.parse.quote(json.dumps(speech_timestamped))

    pair_response = client.get(
        f"/pair_by_words?target_timestamped={target_timestamped_param}&target_by_words={target_by_words_param}&speech_timestamped={speech_timestamped_param}"
    )
    assert pair_response.status_code == 200
    word_phone_pairings = json.loads(pair_response.data)

    # Now call score_words_cer with the word phone pairings
    word_phone_pairings_param = urllib.parse.quote(json.dumps(word_phone_pairings))
    response = client.get(
        f"/score_words_cer?word_phone_pairings={word_phone_pairings_param}"
    )
    print("response", response)
    assert response.status_code == 200
    data = json.loads(response.data)
    words, overall_score = data
    print("words", words)
    assert round(overall_score, 2) == 0.95
    
    # test that all words except the last one are perfect matches
    for eng_word, _, _, score in words[:-1]:
        assert score == 1.0
    words[-1][3] = 0.75
