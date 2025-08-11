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


def _calculate_cnn_window(model: Wav2Vec2ForCTC):
    receptive_field = 1
    stride = 1
    for conv_layer in model.wav2vec2.feature_extractor.conv_layers:
        assert hasattr(conv_layer, "conv")
        conv = conv_layer.conv
        assert isinstance(conv, torch.nn.Conv1d)
        receptive_field += (conv.kernel_size[0] - 1) * stride
        stride *= conv.stride[0]
    return receptive_field, stride


RECEPTIVE_FIELD_SIZE, STRIDE_SIZE = _calculate_cnn_window(model)


def extract_features_only(audio: np.ndarray):
    """Extract CNN features and project to encoder hidden size (transformer-ready)."""
    # True raw sample count before any padding
    raw_sample_count = int(np.asarray(audio).shape[-1])

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=RECEPTIVE_FIELD_SIZE,
    )
    input_values = inputs.input_values.type(torch.float32).to(model.device)
    with torch.no_grad():
        conv_feats = model.wav2vec2.feature_extractor(input_values)  # (B, C, T')
        conv_feats_t = conv_feats.transpose(1, 2)  # (B, T', C)
        # Project to hidden size for transformer; also returns normalized conv features
        features, normed_conv_feats = model.wav2vec2.feature_projection(conv_feats_t)
    # Return transformer-ready features and original (unpadded) input length in samples
    return features, raw_sample_count


def run_transformer_on_features(
    features: torch.Tensor, total_audio_samples: int, time_offset: float = 0.0
) -> TIMESTAMPED_PHONES_T:
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
