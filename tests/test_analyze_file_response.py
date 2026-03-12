import sys

sys.path.append("server/src")

from feedback import build_analysis_response


def test_build_analysis_response_formats_ready_to_use_feedback():
    phone_pairings_by_word = [
        (
            "cat",
            [
                (("k", 0.0, 0.1), ("k", 0.0, 0.1)),
                (("æ", 0.1, 0.2), ("-", 0.1, 0.1)),
                (("t", 0.2, 0.3), ("t", 0.2, 0.3)),
            ],
        ),
        (
            "thin",
            [
                (("θ", 0.3, 0.4), ("t", 0.3, 0.4)),
                (("ɪ", 0.4, 0.5), ("ɪ", 0.4, 0.5)),
                (("n", 0.5, 0.6), ("n", 0.5, 0.6)),
            ],
        ),
        (
            "blue",
            [
                (("-", 0.6, 0.6), ("ə", 0.6, 0.65)),
                (("b", 0.65, 0.75), ("b", 0.65, 0.75)),
                (("l", 0.75, 0.85), ("l", 0.75, 0.85)),
                (("u", 0.85, 0.95), ("u", 0.85, 0.95)),
            ],
        ),
    ]

    response = build_analysis_response(phone_pairings_by_word, topk=5)

    assert response["average_score"] < 1.0
    assert response["spoken_word_timestamps"] == [
        ("cat", 0.0, 0.3),
        ("thin", 0.3, 0.6),
        ("blue", 0.6, 0.95),
    ]
    assert response["words"][0]["text"] == "cat"
    assert response["words"][0]["start"] == 0.0
    assert response["words"][0]["end"] == 0.3

    captions = [item["caption"] for item in response["feedback"]]
    assert any(caption.startswith("You under-enunciated") for caption in captions)
    assert any(caption.startswith("You're pronouncing") for caption in captions)
    assert any(caption.startswith("You tend to add") for caption in captions)
    assert all("words" in item and len(item["words"]) > 0 for item in response["feedback"])
