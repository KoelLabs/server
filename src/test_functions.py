import sys
import os
from scipy.io import wavfile
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server import get_user_word_audio_clip, transcribe_audio
from src.audio_utils import audio_wav_file_to_array
from src.server import SAMPLE_RATE

speech_sample = "./test_samples/aruna-calls-cards.wav"

# Read WAV file and convert to float32 array
speech_audio = audio_wav_file_to_array(speech_sample)
print(speech_audio.shape)

target = "kʰɔlɪŋkʰɑɹdzɑɹðəweɪvəvðəfjutʃɝ"
target_by_words = [
    ["calling", ["kʰ", "ɔ", "l", "ɪ", "ŋ"]],
    ["cards", ["kʰ", "ɑ", "ɹ", "d", "z"]],
    ["are", ["ɑ", "ɹ"]],
    ["the", ["ð", "ə"]],
    ["wave", ["w", "eɪ", "v"]],
    ["of", ["ə", "v"]],
    ["the", ["ð", "ə"]],
    ["future", ["f", "j", "u", "tʃ", "ɝ"]],
]
speech_transcript = [
    "kʰ",
    "ɔ",
    "l",
    "ɪ",
    "ŋ",
    "kʰ",
    "ɑ",
    "ɹ",
    "d",
    "d",
    "z",
    "ɑ",
    "ɹ",
    "ð",
    "ə",
    "w",
    "eɪ",
    "v",
    "ə",
    "v",
    "ð",
    "ə",
    "f",
    "j",
    "u",
    "tʃ",
    "ɝ",
]


transcription, timestamps = transcribe_audio(speech_audio, include_timestamps=True)
# test word clipping
print("transcription", transcription)
new_speech = get_user_word_audio_clip(
    ["kʰ", "ɑ", "ɹ", "d", "z"], timestamps, speech_audio
)

wavfile.write(
    "test_samples/aruna-calls-cards-sample-cards.wav", SAMPLE_RATE, new_speech
)
