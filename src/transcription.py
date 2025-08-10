import torch
import numpy as np
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from phoneme_utils import TIMESTAMPED_PHONES_T

SAMPLE_RATE = 16_000

# Load Wav2Vec2 model
model_id = "KoelLabs/xlsr-english-01"
processor: Wav2Vec2Processor = AutoProcessor.from_pretrained(model_id)
model: Wav2Vec2ForCTC = AutoModelForCTC.from_pretrained(model_id)
assert processor.feature_extractor.sampling_rate == SAMPLE_RATE
MIN_LEN_SAMPLES = (
    400  # computed from model.config.conv_kernel and model.config.conv_stride
)


def extract_features_only(audio: np.ndarray):
    """Extract CNN features and project to encoder hidden size (transformer-ready)."""
    # True raw sample count before any padding
    raw_sample_count = int(np.asarray(audio).shape[-1])
    if raw_sample_count < MIN_LEN_SAMPLES:
        audio = np.pad(audio, (0, MIN_LEN_SAMPLES - raw_sample_count), mode="constant")
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs.input_values.type(torch.float32).to(model.device)
    with torch.no_grad():
        conv_feats = model.wav2vec2.feature_extractor(input_values)  # (B, C, T')
        conv_feats_t = conv_feats.transpose(1, 2)  # (B, T', C)
        # Project to hidden size for transformer; also returns normalized conv features
        features, normed_conv_feats = model.wav2vec2.feature_projection(conv_feats_t)
    # Return transformer-ready features and original (unpadded) input length in samples
    return features, raw_sample_count


def run_transformer_on_features(features, total_audio_samples, time_offset=0.0):
    """Run transformer and decode"""
    #  slowest step
    with torch.no_grad():
        encoder_outputs = model.wav2vec2.encoder(features)
        logits = model.lm_head(encoder_outputs[0])

    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
    # Use original audio length in samples to compute duration
    duration_sec = total_audio_samples / processor.feature_extractor.sampling_rate
    ids_w_time = [
        (time_offset + i / len(predicted_ids) * duration_sec, _id)
        for i, _id in enumerate(predicted_ids)
    ]
    current_phoneme_id = processor.tokenizer.pad_token_id
    current_start_time = 0
    phonemes_with_time = []
    for timestamp, _id in ids_w_time:
        if current_phoneme_id != _id:
            if current_phoneme_id != processor.tokenizer.pad_token_id:
                phonemes_with_time.append(
                    (
                        processor.decode(current_phoneme_id),
                        current_start_time,
                        timestamp,
                    )
                )

            current_start_time = timestamp
            current_phoneme_id = _id
    return phonemes_with_time


def transcribe_timestamped(audio: np.ndarray, time_offset=0.0) -> TIMESTAMPED_PHONES_T:
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
    duration_sec = input_values.shape[1] / processor.feature_extractor.sampling_rate

    ids_w_time = [
        (time_offset + i / len(predicted_ids) * duration_sec, _id)
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

    return phonemes_with_time
