import json

TARGET = "ɔliŋkɑɹdsʔɑɹðəweɪvəvðifjutʃɹ"
TARGET_BY_WORD = [
    ["Calling", "ɔliŋ"],
    ["cards", "kɑɹdsʔ"],
    ["are", "ɑɹ"],
    ["the", "ðə"],
    ["wave", "weɪv"],
    ["of", "əv"],
    ["the", "ði"],
    ["future", "fjutʃɹ"],
]


def test_score_words_cer(client):
    response = client.get(
        f"/score_words_cer?target={TARGET}&tbw={json.dumps(TARGET_BY_WORD)}&speech=ɔliŋkɑɹdsʔɑɹðəweɪvəvðifjut"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    words, overall_score = data
    assert overall_score == 0.9875
    for _, _, _, score in words[:-1]:
        assert score == 1.0
    assert words[-1][3] == 0.9
