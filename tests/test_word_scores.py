import json
import urllib.parse

# fmt: off
TARGET_BY_WORDS = [
    ("calling", [("kʰ", 0, 1), ("ɔ", 1, 2), ("l", 2, 3), ("ɪ", 3, 4), ("ŋ", 4, 5)]),
    ("cards", [("kʰ", 5, 6), ("ɑ", 6, 7), ("ɹ", 7, 8), ("d", 8, 9), ("z", 9, 10)]),
    ("are", [("ɑ", 10, 11), ("ɹ", 11, 12)]),
    ("the", [("ð", 12, 13), ("ə", 13, 14)]),
    ("wave", [("w", 14, 15), ("eɪ", 15, 16), ("v", 16, 17)]),
    ("of", [("ə", 17, 18), ("v", 18, 19)]),
    ("the", [("ð", 19, 20), ("ə", 20, 21)]),
    ("future", [("f", 21, 22), ("j", 22, 23), ("u", 23, 24), ("tʃ", 24, 25), ("ɜ˞", 25, 26)]),
]
# fmt: on


def test_score_words_cer(client):
    # test perfect match except the last word
    speech = [p for _, phones in TARGET_BY_WORDS for p in phones]
    speech[-1] = ("ə˞", 0, 0)

    target_by_words_param = urllib.parse.quote(json.dumps(TARGET_BY_WORDS))
    speech_param = urllib.parse.quote(json.dumps(speech))
    response = client.get(
        f"/score_words_cer?target_by_words={target_by_words_param}&speech={speech_param}"
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    words, overall_score = data
    assert round(overall_score, 2) == 0.97

    # test that all words except the last one are perfect matches
    for _, score in words[:-1]:
        assert score == 1.0
    assert words[-1][1] == 0.8
