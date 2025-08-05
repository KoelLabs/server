import os
import json

import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS, cross_origin
from flask_sock import Sock
from flask.json.provider import _default as _json_default

from feedback import (
    score_words_cer,
    score_words_wfed,
    user_phonetic_errors,
    pair_by_words,
)

DEBUG = False

# Constants
SAMPLE_RATE = 16_000
NUM_SECONDS_PER_CHUNK = 2

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)  # allow CORS for all domains on all routes.
app.config["CORS_HEADERS"] = "Content-Type"
sock = Sock(app)


def json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    return _json_default(obj)


app.json.default = json_default

# Load Wav2Vec2 model
model_id = "KoelLabs/xlsr-english-01"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id)

model_vocab_json = os.path.join(os.path.dirname(__file__), "model_vocab_feedback.json")


def transcribe_audio(audio: np.ndarray) -> list[str]:
    """
    Transcribe audio and return both transcription and timestamp information.
    """
    transcription, _, _ = _run_inference(audio, model, processor)
    return transcription


def transcribe_timestamped(audio):
    transcription, duration_sec, predicted_ids = _run_inference(audio, model, processor)

    ids_w_time = [
        (i / len(predicted_ids) * duration_sec, _id)
        for i, _id in enumerate(predicted_ids)
    ]

    current_phoneme_id = processor.tokenizer.pad_token_id
    current_start_time = 0
    phonemes_with_time = []
    for time, _id in ids_w_time:
        if current_phoneme_id != _id:
            if current_phoneme_id != processor.tokenizer.pad_token_id:
                phonemes_with_time.append(
                    (
                        processor.decode(current_phoneme_id),
                        current_start_time,
                        time,
                    )
                )
            current_start_time = time
            current_phoneme_id = _id

    return transcription, phonemes_with_time


def _run_inference(audio, model, processor):
    input_values = (
        processor(
            audio,
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        .input_values.type(torch.float32)
        .to(model.device)
    )
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids)
    transcription = [
        t for t in tokens if t not in processor.tokenizer.all_special_tokens
    ]
    duration_sec = input_values.shape[1] / processor.feature_extractor.sampling_rate
    return transcription, duration_sec, predicted_ids


# server /
@app.route("/")
@cross_origin()
def index():
    return send_from_directory("static", "index.html")


# serve static files
@app.route("/<path:path>")
@cross_origin()
def send_static(path):
    return send_from_directory("static", path)


@app.route("/pair_by_words", methods=["GET"])
@cross_origin()
def get_pair_by_words():
    try:
        speech_timestamped = json.loads(request.args.get("speech_timestamped", "[]"))
        target_timestamped = json.loads(request.args.get("target_timestamped", "[]"))
        target_by_words = json.loads(request.args.get("target_by_words", "[]"))
    except Exception as e:
        return jsonify({"client called get_pair_by_words incorrectly": str(e)}), 400

    return jsonify(
        pair_by_words(target_timestamped, target_by_words, speech_timestamped)
    )


@app.route("/user_phonetic_errors", methods=["GET"])
@cross_origin()
def get_user_phonetic_errors():
    try:
        word_phone_pairings = json.loads(request.args.get("word_phone_pairings", "[]"))
    except Exception as e:
        return (
            jsonify({"client called get_user_phonetic_errors incorrectly": str(e)}),
            400,
        )
    return jsonify(user_phonetic_errors(word_phone_pairings))


# REST endpoint
@app.route("/score_words_cer", methods=["GET"])
@cross_origin()
def get_score_words_cer():
    try:
        word_phone_pairings = json.loads(request.args.get("word_phone_pairings", "[]"))
    except Exception as e:
        return jsonify({"client called get_score_words_cer incorrectly": str(e)}), 400
    word_scores = score_words_cer(word_phone_pairings)
    return jsonify(word_scores)


@app.route("/score_words_wfed", methods=["GET"])
@cross_origin()
def get_score_words_wfed():
    try:
        word_phone_pairings = json.loads(request.args.get("word_phone_pairings", "[]"))
    except Exception as e:
        return jsonify({"client called get_score_words_wfed incorrectly": str(e)}), 400
    word_scores = score_words_wfed(word_phone_pairings)
    return jsonify(word_scores)


@sock.route("/stream")
def stream(ws):
    buffer = b""  # Buffer to hold audio chunks

    full_transcription = []
    full_timestamps = []
    combined = np.array([], dtype=np.float32)
    while True:
        try:
            # Receive audio data from the client
            data = (
                ws.receive()
            )  # NOTE: anything left in the buffer will not be processed if client stops streaming
            if data == "stop":
                break

            if data:
                buffer += data
                # Process chunks when buffer reaches certain size
                if (
                    len(buffer) // 4 > SAMPLE_RATE * NUM_SECONDS_PER_CHUNK
                ):  # Adjust size for chunk processing

                    audio = np.frombuffer(buffer, dtype=np.float32)
                    combined = np.concatenate([combined, audio])

                    new_transcription, new_timestamps = transcribe_timestamped(audio)
                    full_transcription.extend(new_transcription)
                    full_timestamps.extend(new_timestamps)
                    ws.send(
                        json.dumps(
                            {
                                "speech_transcript": full_transcription,
                                "speech_timestamps": full_timestamps,
                            }
                        )
                    )
                    combined = np.array([], dtype=np.float32)

                    if DEBUG:
                        from scipy.io import wavfile

                        wavfile.write("audio.wav", SAMPLE_RATE, audio)
                        wavfile.write("combined.wav", SAMPLE_RATE, combined)

                    buffer = b""  # Clear the buffer
        except Exception as e:
            print(f"Error: {e}")
            print(f"Line: {e.__traceback__.tb_lineno if e.__traceback__ else -1}")
            break


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
