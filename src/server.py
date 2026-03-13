import io
import json
import mimetypes
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from scipy.io import wavfile


_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_S3_REQUIRED_ENV_VARS = (
    "AWS_S3_REGION",
    "AWS_S3_BUCKET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
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


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file(_ENV_PATH)


def _parse_request_metadata(raw_metadata: str | None) -> dict[str, Any]:
    if raw_metadata is None or raw_metadata.strip() == "":
        return {}

    parsed = json.loads(raw_metadata)
    if not isinstance(parsed, dict):
        raise ValueError("metadata must be a JSON object")

    return parsed


def _sanitize_s3_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for raw_key, raw_value in metadata.items():
        if raw_value is None:
            continue

        key = re.sub(r"[^a-z0-9-]+", "-", str(raw_key).strip().lower()).strip("-")
        if not key:
            continue

        value = str(raw_value).strip()
        if not value:
            continue

        sanitized[key[:128]] = value[:1024]

    return sanitized


def _target_sentence_from_request(
    target_words: list[str], target_by_words: TIMESTAMPED_PHONES_BY_WORD_T
) -> str:
    if len(target_words) > 0:
        return " ".join(target_words).strip()

    return " ".join(word for word, _ in target_by_words).strip()


def _get_s3_config() -> dict[str, str] | None:
    values = {key: os.getenv(key, "").strip() for key in _S3_REQUIRED_ENV_VARS}
    if not all(values.values()):
        missing = [key for key, value in values.items() if not value]
        app.logger.info("Skipping S3 upload; missing config: %s", ", ".join(missing))
        return None
    return values


def _build_s3_object_key(filename: str | None) -> str:
    extension = Path(filename or "").suffix.lower()
    if not extension:
        extension = ".bin"

    now = datetime.now(timezone.utc)
    return f"user-audio/{now:%Y/%m/%d/%H%M%S}-" f"{uuid.uuid4().hex}{extension}"


def _upload_audio_to_s3(
    *,
    audio_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    metadata: dict[str, Any],
) -> None:
    config = _get_s3_config()
    if config is None:
        return

    try:
        import boto3
    except ImportError:
        app.logger.warning("Skipping S3 upload; boto3 is not installed")
        return

    guessed_content_type = content_type or mimetypes.guess_type(filename or "")[0]
    extra_args: dict[str, Any] = {
        "Bucket": config["AWS_S3_BUCKET"],
        "Key": _build_s3_object_key(filename),
        "Body": audio_bytes,
        "Metadata": _sanitize_s3_metadata(metadata),
    }
    if guessed_content_type:
        extra_args["ContentType"] = guessed_content_type

    try:
        client = boto3.client(
            "s3",
            region_name=config["AWS_S3_REGION"],
            aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"],
        )
        client.put_object(**extra_args)
    except Exception:
        app.logger.exception("Failed to upload analyzed audio to S3")


def _queue_audio_upload(
    *,
    audio_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    metadata: dict[str, Any],
) -> None:
    threading.Thread(
        target=_upload_audio_to_s3,
        kwargs={
            "audio_bytes": audio_bytes,
            "filename": filename,
            "content_type": content_type,
            "metadata": metadata,
        },
        daemon=True,
    ).start()


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
        metadata = _parse_request_metadata(request.form.get("metadata"))
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
    metadata = {
        **metadata,
        "target_sentence": _target_sentence_from_request(target_words, target_by_words),
    }

    f = request.files["file"]
    file_bytes = f.read()

    # Decode WAV file
    sr, audio = wavfile.read(io.BytesIO(file_bytes))

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
    response = jsonify(build_analysis_response(phone_pairings_by_word, topk=topk))
    _queue_audio_upload(
        audio_bytes=file_bytes,
        filename=f.filename,
        content_type=f.mimetype,
        metadata=metadata,
    )
    return response


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
