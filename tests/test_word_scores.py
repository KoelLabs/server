import json
import urllib.parse

TARGET_BY_WORDS = [
    ("calling", ["kʰ", "ɔ", "l", "ɪ", "ŋ"]),
    ("cards", ["kʰ", "ɑ", "ɹ", "d", "z"]),
    ("are", ["ɑ", "ɹ"]),
    ("the", ["ð", "ə"]),
    ("wave", ["w", "eɪ", "v"]),
    ("of", ["ə", "v"]),
    ("the", ["ð", "ə"]),
    ("future", ["f", "j", "u", "tʃ", "ɜ˞"]),
]
_timestamp = 0
TARGET_TIMESTAMPED = [
    (phone, _timestamp, _timestamp := _timestamp + 1)
    for _, phones in TARGET_BY_WORDS
    for phone in phones
]


def test_score_words_cer(client):
    # test perfect match except the last word
    speech_timestamped = TARGET_TIMESTAMPED.copy()
    speech_timestamped[-1] = ("ə˞", 0, 0)

    # First, get word phone pairings
    target_timestamped_param = urllib.parse.quote(json.dumps(TARGET_TIMESTAMPED))
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
    assert round(overall_score, 2) == 0.97

    # test that all words except the last one are perfect matches
    for _, _, _, score in words[:-1]:
        assert score == 1.0
    assert words[-1][3] == 0.8
