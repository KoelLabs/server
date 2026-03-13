import sys
import io
import json

sys.path.append("server/src")

import numpy as np
from scipy.io import wavfile

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


def _build_test_wav_bytes(sample_rate: int = 16000) -> bytes:
    audio = np.zeros(sample_rate // 10, dtype=np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio)
    return buffer.getvalue()


def test_analyze_file_queues_audio_upload_with_metadata(client, monkeypatch):
    import server

    queued_uploads = []

    monkeypatch.setattr(server, "transcribe_timestamped", lambda audio, offset: [])
    monkeypatch.setattr(server, "pair_by_words", lambda target_by_words, transcription: [])
    monkeypatch.setattr(
        server,
        "build_analysis_response",
        lambda phone_pairings_by_word, topk=3: {
            "words": [],
            "average_score": 1.0,
            "spoken_word_timestamps": [],
            "feedback": [],
        },
    )
    monkeypatch.setattr(
        server,
        "_queue_audio_upload",
        lambda **kwargs: queued_uploads.append(kwargs),
    )

    wav_bytes = _build_test_wav_bytes()
    metadata = {"native_language": "Spanish", "user_id": "user-123"}
    response = client.post(
        "/analyze_file?target_words=%5B%22hello%22%5D",
        data={
            "file": (io.BytesIO(wav_bytes), "practice.wav"),
            "metadata": json.dumps(metadata),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["feedback"] == []
    assert len(queued_uploads) == 1
    assert queued_uploads[0]["audio_bytes"] == wav_bytes
    assert queued_uploads[0]["filename"] == "practice.wav"
    assert queued_uploads[0]["metadata"] == {
        **metadata,
        "target_sentence": "hello",
    }


def test_analyze_file_rejects_non_object_metadata(client):
    wav_bytes = _build_test_wav_bytes()

    response = client.post(
        "/analyze_file?target_words=%5B%22hello%22%5D",
        data={
            "file": (io.BytesIO(wav_bytes), "practice.wav"),
            "metadata": json.dumps(["not", "an", "object"]),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "metadata must be a JSON object" in response.get_json()["Malformatted arguments"]
