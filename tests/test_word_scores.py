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
    speech = ['kʰ', 'ɔ', 'l', 'ɪ', 'ŋ', 'kʰ', 'ɑ', 'ɹ', 'd', 'd', 'z', 'ɑ', 'ɹ', 'ð', 'ə', 'w', 'eɪ', 'v', 'ə', 'v', 'ð', 'ə', 'f', 'k', 'u', 't', 'ɜ˞']  # fmt: skip

    # NOTE: I url encode the json parameters because the speech is a list, this may need fixing
    target_param = urllib.parse.quote(json.dumps(TARGET))
    tbw_param = urllib.parse.quote(json.dumps(TARGET_BY_WORDS))
    speech_param = urllib.parse.quote(json.dumps(speech))

    response = client.get(
        f"/score_words_cer?target={target_param}&tbw={tbw_param}&speech={speech_param}"
    )
    print("response", response)
    assert response.status_code == 200
    data = json.loads(response.data)
    words, overall_score = data
    print("words", words)
    assert round(overall_score, 2) == 0.97
    # test that all words except the last one are perfect matches
    for eng_word, _, _, score in words[:-1]:
        assert score == 1.0
    words[-1][3] = 0.75
