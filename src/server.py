import json

import numpy as np

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS, cross_origin
from flask_sock import Sock
from flask.json.provider import _default as _json_default

from feedback import (
    score_words_cer,
    top_phonetic_errors,
    pair_by_words,
)
from transcription import transcribe_timestamped, SAMPLE_RATE
from phoneme_utils import TIMESTAMPED_PHONES_T, TIMESTAMPED_PHONES_BY_WORD_T

# Constants
DEBUG = False
NUM_SECONDS_PER_CHUNK = 0.5

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)  # allow CORS for all domains on all routes.
app.config["CORS_HEADERS"] = "Content-Type"
sock = Sock(app)


# Extend JSON stringifying fallbacks
def json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    return _json_default(obj)


app.json.default = json_default  # type: ignore


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


@app.route("/top_phonetic_errors", methods=["GET"])
@cross_origin()
def get_top_phonetic_errors():
    try:
        speech: TIMESTAMPED_PHONES_T = json.loads(request.args.get("speech", "[]"))
        target_by_words: TIMESTAMPED_PHONES_BY_WORD_T = json.loads(
            request.args.get("target_by_words", "[]")
        )
        topk = int(json.loads(request.args.get("topk", "3")))
    except Exception as e:
        return jsonify({"Malformatted arguments": str(e)}), 400

    phone_pairings_by_word = pair_by_words(target_by_words, speech)
    return jsonify(top_phonetic_errors(phone_pairings_by_word, topk=topk))


@app.route("/score_words_cer", methods=["GET"])
@cross_origin()
def get_score_words_cer():
    try:
        speech: TIMESTAMPED_PHONES_T = json.loads(request.args.get("speech", "[]"))
        target_by_words: TIMESTAMPED_PHONES_BY_WORD_T = json.loads(
            request.args.get("target_by_words", "[]")
        )
    except Exception as e:
        return jsonify({"Malformatted arguments": str(e)}), 400

    phone_pairings_by_word = pair_by_words(target_by_words, speech)
    return jsonify(score_words_cer(phone_pairings_by_word))


@sock.route("/stream")
def stream(ws):
    buffer = b""  # Buffer to hold audio chunks

    full_transcription: TIMESTAMPED_PHONES_T = []
    accumulated_duration = 0
    combined = np.array([], dtype=np.float32)
    while True:
        try:
            # Receive audio data from the client
            data = ws.receive()
            if data and data != "stop":
                buffer += data

            # Process when buffer has at least one chunk in it or when we are done
            if (
                data == "stop"
                or len(buffer)
                >= SAMPLE_RATE * NUM_SECONDS_PER_CHUNK * np.dtype(np.float32).itemsize
            ):
                audio = np.frombuffer(buffer, dtype=np.float32)
                transcription = transcribe_timestamped(audio, accumulated_duration)
                accumulated_duration += len(audio) / SAMPLE_RATE
                full_transcription.extend(transcription)
                ws.send(json.dumps(full_transcription))

                if DEBUG:
                    from scipy.io import wavfile

                    wavfile.write("src/audio.wav", SAMPLE_RATE, audio)
                    combined = np.concatenate([combined, audio])
                    wavfile.write("src/combined.wav", SAMPLE_RATE, combined)

                if data == "stop":
                    break

                buffer = b""  # Clear the buffer
        except Exception as e:
            print(f"Error: {e}")
            print(f"Line: {e.__traceback__.tb_lineno if e.__traceback__ else -1}")
            break


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
