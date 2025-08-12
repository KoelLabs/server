import json
import numpy as np
import torch

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS, cross_origin
from flask_sock import Sock
from flask.json.provider import _default as _json_default

from feedback import (
    score_words_cer,
    top_phonetic_errors,
    pair_by_words,
)
from transcription import (
    extract_features_only,
    run_transformer_on_features,
    STRIDE_SIZE,
)
from phoneme_utils import TIMESTAMPED_PHONES_T, TIMESTAMPED_PHONES_BY_WORD_T

# Constants
DEBUG = False
TRANSFORMER_INTERVAL = 30

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
    buffer = b""
    feature_list = []
    total_samples_processed = 0

    while True:
        try:
            data = ws.receive()
            if data and data != "stop":
                buffer += data

            # Process 20ms chunks
            while len(buffer) >= STRIDE_SIZE * np.dtype(np.float32).itemsize:
                chunk_bytes = buffer[: STRIDE_SIZE * np.dtype(np.float32).itemsize]
                buffer = buffer[STRIDE_SIZE * np.dtype(np.float32).itemsize :]

                audio_chunk = np.frombuffer(chunk_bytes, dtype=np.float32)

                features, samples = extract_features_only(audio_chunk)
                feature_list.append(features)
                total_samples_processed += samples
                # accumulate features for 500ms (25 sets of 20ms), then send COMPLETE transcription from start
                if len(feature_list) % TRANSFORMER_INTERVAL == 0:
                    all_features = torch.cat(feature_list, dim=1)
                    full_transcription = run_transformer_on_features(
                        all_features, total_samples_processed
                    )
                    ws.send(json.dumps(full_transcription))

            if data == "stop":
                # Final update with any remaining features
                if feature_list:
                    all_features = torch.cat(feature_list, dim=1)
                    full_transcription = run_transformer_on_features(
                        all_features, total_samples_processed
                    )
                    ws.send(json.dumps(full_transcription))
                break

        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
