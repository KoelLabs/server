import numpy as np
import sounddevice as sd
from scipy.io import wavfile

TARGET_SAMPLE_RATE = 16000


def audio_resample(array, src_sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    """Resample audio array to target sample rate"""
    if src_sample_rate == target_sample_rate:
        return array
    return np.interp(
        np.linspace(
            0,
            len(array),
            int(len(array) * target_sample_rate / src_sample_rate),
        ),
        np.arange(len(array)),
        array,
    ).astype(np.int16)


def audio_array_play(input_array, sample_rate=TARGET_SAMPLE_RATE):
    """Play audio array using sounddevice"""
    sd.play(input_array, sample_rate)
    sd.wait()


def audio_wav_file_play(input_path, start_sec=None, end_sec=None):
    """Play audio from WAV file using sounddevice"""
    print(start_sec, end_sec)
    rate, data = wavfile.read(input_path)
    start = int(float(start_sec) * rate) if start_sec else 0
    end = int(float(end_sec) * rate) if end_sec else len(data)
    data = data[start:end]
    audio_array_play(data, rate)


def audio_wav_file_crop(input_path, start_sec, end_sec, output_path):
    """Crop audio from WAV file using sounddevice"""
    rate, data = wavfile.read(input_path)
    start = int(float(start_sec) * rate)
    end = int(float(end_sec) * rate)
    data = data[start:end]
    audio_array_to_wav_file(data, output_path, rate)


def audio_array_to_wav_file(
    input_array, output_path, output_sample_rate=TARGET_SAMPLE_RATE
):
    """Write audio array to WAV file"""
    wavfile.write(output_path, output_sample_rate, input_array)
