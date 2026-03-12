import json

import numpy as np

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS, cross_origin
from flask.json.provider import _default as _json_default

from feedback import (
    build_analysis_response,
    pair_by_words,
)
from transcription import transcribe_timestamped, SAMPLE_RATE
from phoneme_utils import (
    TIMESTAMPED_PHONES_BY_WORD_T,
    english2ipa,
)


# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)  # allow CORS for all domains on all routes.
app.config["CORS_HEADERS"] = "Content-Type"


# Extend JSON stringifying fallbacks
def json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    return _json_default(obj)


app.json.ensure_ascii = False  # type: ignore
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


@app.route("/analyze_file", methods=["POST"])
@cross_origin()
def analyze_file():
    try:
        target_words: list[str] = json.loads(request.args.get("target_words", "[]"))
        target_by_words: TIMESTAMPED_PHONES_BY_WORD_T = json.loads(
            request.args.get("target_by_words", "[]")
        )
        topk = int(json.loads(request.args.get("topk", "3")))
        assert (
            len(target_words) > 0 or len(target_by_words) > 0
        ), "must have some target_words"
    except Exception as e:
        return jsonify({"Malformatted arguments": str(e)}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    if len(target_by_words) == 0:
        target_by_words: TIMESTAMPED_PHONES_BY_WORD_T = [
            (word, [(p, 0, 0) for p in english2ipa(word)]) for word in target_words
        ]

    from scipy.io import wavfile
    import io

    f = request.files["file"]

    # Decode WAV file
    sr, audio = wavfile.read(io.BytesIO(f.read()))

    if sr != SAMPLE_RATE:
        return jsonify({"error": f"Expected {SAMPLE_RATE}Hz, got {sr}Hz"}), 400

    original_dtype = audio.dtype
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        if original_dtype == np.int16:
            audio /= 32768.0
        elif original_dtype == np.int32:
            audio /= 2147483648.0

    # mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    transcription = transcribe_timestamped(audio, 0.0)
    phone_pairings_by_word = pair_by_words(target_by_words, transcription)
    return jsonify(build_analysis_response(phone_pairings_by_word, topk=topk))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
