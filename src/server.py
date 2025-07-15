from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS, cross_origin
from flask_sock import Sock
import torch
from transformers import AutoProcessor, AutoModelForCTC
import numpy as np
import os
import io, base64

from feedback import (
    score_words_cer,
    score_words_wfed,
    phoneme_written_feedback,
    user_phonetic_errors,
)
import json

# Note: map_phoneme_for_panphon_ipapy is used within feedback(), keep import
from phoneme_utils import phrase_bounds, map_phoneme_for_panphon_ipapy
import soundfile as sf
from scipy.io import wavfile

# Constants
SAMPLE_RATE = 16000
NUM_SECONDS_PER_CHUNK = 2
NUM_CHUNKS_ACCUMULATED = 5

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)  # allow CORS for all domains on all routes.
app.config["CORS_HEADERS"] = "Content-Type"
sock = Sock(app)

# Load Wav2Vec2 model
model_id = "KoelLabs/xlsr-english-01"
# model_id = "speech31/wav2vec2-large-english-TIMIT-phoneme_v3"
# model_id = "ginic/hyperparam_tuning_1_wav2vec2-large-xlsr-buckeye-ipa"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id)

model_vocab_json = os.path.join(os.path.dirname(__file__), "model_vocab_feedback.json")


def transcribe_audio(
    audio: np.ndarray, include_timestamps: bool = False
) -> "tuple[str, list]":
    """
    Transcribe audio and return both transcription and timestamp information.

    Returns:
        tuple: (transcription_string, timestamps_list)
        timestamps_list contains (start_time, end_time, token) for each token
    """
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids[0])
    transcription = [
        t for t in tokens if t not in processor.tokenizer.all_special_tokens
    ]
    print("ARUNA transcription", transcription)

    if include_timestamps:
        transcription_batch, phonemes_with_time_batch = transcribe_batch_timestamped(
            [(None, audio)], model, processor
        )
        return transcription, phonemes_with_time_batch[0]
    else:
        return transcription


def transcribe_batch_timestamped(batch, model, processor):
    input_values = (
        processor(
            [x[1] for x in batch],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        .input_values.type(torch.float32)
        .to(model.device)
    )
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids_batch = torch.argmax(logits, dim=-1)
    transcription_batch = [processor.decode(ids) for ids in predicted_ids_batch]

    # get the start and end timestamp for each phoneme
    phonemes_with_time_batch = []
    for predicted_ids in predicted_ids_batch:
        predicted_ids = predicted_ids.tolist()
        duration_sec = input_values.shape[1] / processor.feature_extractor.sampling_rate

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

        phonemes_with_time_batch.append(phonemes_with_time)

    return transcription_batch, phonemes_with_time_batch


def confidence_score(logits, predicted_ids) -> "tuple[np.ndarray, float]":
    scores = torch.nn.functional.softmax(logits, dim=-1)
    pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
    mask = torch.logical_and(
        predicted_ids.not_equal(processor.tokenizer.word_delimiter_token_id),
        predicted_ids.not_equal(processor.tokenizer.pad_token_id),
    )

    character_scores = pred_scores.masked_select(mask)
    total_average = torch.sum(character_scores) / len(character_scores)
    return character_scores.numpy(), total_average.float().item()


def get_error_audio_clip(word_phone_pairings, timestamps, speech_audio, word_idx):

    start, end = phrase_bounds(word_phone_pairings, timestamps, word_idx)
    clip = speech_audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]

    # Encode the clip as a WAV file in-memory and then base64 so it can be JSON-serialised.
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, clip)
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return audio_b64


def feedback(target, target_by_words, speech_audio, speech_phones, timestamps):
    if len(speech_phones) == 0:
        return []
    # retrieve main info
    user_phonetic_errors_dict = user_phonetic_errors(
        target, target_by_words, speech_phones, topk=3
    )

    feedback_items = []
    for phoneme, error_info in user_phonetic_errors_dict.items():
        num_mistakes, which_words, mistake_severities, phoneme_spoken_as, score = (
            error_info
        )
        target_phoneme_spelling = get_phoneme_feedback(phoneme)["phonetic-spelling"]
        target_phoneme_explanation = get_phoneme_feedback(phoneme)["explanation"]
        target_phoneme_video = get_phoneme_feedback(phoneme)["video"]
        # target_phoneme_audio = get_phoneme_feedback(phoneme)["audio"] # NOTE: this can be added once we create a page of all audio trancripts for videos
        speech_phoneme_words = target_by_words[which_words]
        speech_phoneme_spelling = get_phoneme_feedback(phoneme)["phonetic-spelling"]
        speech_phoneme_audio = get_error_audio_clip(
            target_by_words, timestamps, speech_audio, which_words
        )

        feedback_item = [
            target_phoneme_spelling,
            target_phoneme_explanation,
            target_phoneme_video,
            # target_phoneme_audio = None,
            speech_phoneme_words,
            speech_phoneme_spelling,
            speech_phoneme_audio,
        ]
        feedback_items.append(feedback_item)

    return feedback_items


def get_user_word_audio_clip(speech_word, timestamps, speech_audio):

    first_phoneme = speech_word[0]
    last_phoneme = speech_word[-1]
    print(timestamps)

    print(timestamps[first_phoneme], timestamps[last_phoneme])

    word_start_time = int(float(timestamps[first_phoneme][0]) * SAMPLE_RATE)
    word_end_time = int(float(timestamps[last_phoneme][1]) * SAMPLE_RATE)

    return speech_audio[word_start_time:word_end_time]


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


@app.route("/phoneme_written_feedback", methods=["GET"])
@cross_origin()
def get_phoneme_feedback():
    try:
        target = request.args.get("target", "").strip()
        speech = request.args.get("speech", "").strip()

        result = phoneme_written_feedback(target, speech)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/user_phonetic_errors", methods=["GET"])
@cross_origin()
def get_user_phonetic_errors():
    try:
        target = request.args.get("target", "").strip()
        target_by_word = json.loads(request.args.get("tbw") or "null")
        speech = request.args.get("speech", "").strip()

        result = user_phonetic_errors(target, target_by_word, speech)

        result_sorted_freq = sorted(result.items(), key=lambda x: x[1][0], reverse=True)
        # Convert sets to lists for JSON serialization
        serializable_result = {}
        for phoneme, (count, words, severities, spoken_as) in result_sorted_freq:
            serializable_result[phoneme] = [
                count,
                list(words),
                severities,
                list(spoken_as),
            ]

        return jsonify(serializable_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# REST endpoint
@app.route("/score_words_cer", methods=["GET"])
@cross_origin()
def get_score_words_cer():
    target = request.args.get("target", "").strip()
    target_by_word = json.loads(request.args.get("tbw") or "null")
    speech = request.args.get("speech", "").strip()
    if not speech:
        return jsonify([[[word, seq, "", 0] for word, seq in target_by_word], 0])
    word_scores = score_words_cer(target, target_by_word, speech)
    return jsonify(word_scores)


@app.route("/score_words_wfed", methods=["GET"])
@cross_origin()
def get_score_words_wfed():
    target = request.args.get("target", "").strip()
    target_by_word = json.loads(request.args.get("tbw") or "null")
    speech = request.args.get("speech", "").strip()
    if not speech:
        return jsonify([[[word, seq, "", 0] for word, seq in target_by_word], 0])
    word_scores = score_words_wfed(target, target_by_word, speech)
    return jsonify(word_scores)


@app.route("/feedback", methods=["POST"])
@cross_origin()
def get_feedback():
    # 1. pull the inputs out of the request
    target = request.form["target"].strip()
    target_by_words = json.loads(request.form["tbw"])
    audio_bytes = request.files["audio"].read()
    speech_audio = np.frombuffer(audio_bytes, dtype=np.float32)

    print(
        f"Feedback endpoint: audio length = {len(speech_audio)} samples ({len(speech_audio)/SAMPLE_RATE:.2f} seconds)"
    )

    # Check if audio is too short (less than 0.1 seconds)
    if len(speech_audio) < SAMPLE_RATE * 0.1:
        print(f"Audio too short: {len(speech_audio)} samples")
        return jsonify({"error": "Audio too short to process"}), 400

    # 2. ONE call to the model
    speech_phones, timestamps = transcribe_audio(
        speech_audio, include_timestamps=True
    )  # speech_phones is already a LIST of phoneme strings
    speech_phones = map_phoneme_for_panphon_ipapy(speech_phones)
    # 3. continue the pipeline without ever calling transcribe_audio again
    out = feedback(
        target,
        target_by_words,
        speech_audio,  # raw audio (for clipping)
        speech_phones,
        timestamps,  # per-phoneme timing info
    )
    return jsonify(out)


# WebSocket endpoint for transcription
@sock.route("/stream")
def stream(ws):
    buffer = b""  # Buffer to hold audio chunks

    full_transcription = ""
    combined = np.array([], dtype=np.float32)
    num_chunks_accumulated = 0
    transcription = ""
    while True:
        try:
            # Receive audio data from the client
            data = ws.receive()
            if data:
                buffer += data
                # Process chunks when buffer reaches certain size
                if (
                    len(buffer) // 4 > SAMPLE_RATE * NUM_SECONDS_PER_CHUNK
                ):  # Adjust size for chunk processing
                    num_chunks_accumulated += 1

                    audio = np.frombuffer(buffer, dtype=np.float32)
                    combined = np.concatenate([combined, audio])

                    if num_chunks_accumulated < NUM_CHUNKS_ACCUMULATED:
                        transcription += transcribe_audio(audio)
                        ws.send(full_transcription + transcription)
                    else:
                        full_transcription += transcribe_audio(combined)
                        ws.send(full_transcription)
                        combined = np.array([], dtype=np.float32)
                        num_chunks_accumulated = 0
                        transcription = ""

                    buffer = b""  # Clear the buffer
        except Exception as e:
            break


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
