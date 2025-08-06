import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC
from phoneme_utils import TIMESTAMPED_PHONES_T

SAMPLE_RATE = 16_000

# Load Wav2Vec2 model
model_id = "KoelLabs/xlsr-english-01"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id)
assert processor.feature_extractor.sampling_rate == SAMPLE_RATE


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
