import numpy as np
import sounddevice as sd
from scipy.io import wavfile

TARGET_SAMPLE_RATE = 16000
WAV_HEADER_SIZE = 44  # Standard WAV header size


def audio_resample(array, src_sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    """Resample audio array to target sample rate"""
    if src_sample_rate == target_sample_rate:
        return array.astype(np.float32)
    resampled = np.interp(
        np.linspace(
            0,
            len(array),
            int(len(array) * target_sample_rate / src_sample_rate),
        ),
        np.arange(len(array)),
        array,
    ).astype(
        np.float32
    )  # Ensure output is float32
    return resampled


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


def audio_bytes_to_array(
    data,
    src_sample_rate=None,
    target_sample_rate=TARGET_SAMPLE_RATE,
    output_orig_sample_rate=False,
):
    # Verify WAV format by checking RIFF header
    if not data[:4] == b"RIFF":
        raise ValueError(
            "Input must be WAV format - missing RIFF header. First 4 bytes are: "
            + data[:4].decode("utf-8")
        )
    if src_sample_rate == None:
        # read 32 bit integer from bytes 25-28 in header
        src_sample_rate = int.from_bytes(data[24:28], byteorder="little")
    # read bits per sample from bytes 35-36 in header
    bits_per_sample = int.from_bytes(data[34:36], byteorder="little")
    dtype = np.int16 if bits_per_sample == 16 else np.int32
    # read number of channels from bytes 23-24 in header
    num_channels = int.from_bytes(data[22:24], byteorder="little")
    data = data[WAV_HEADER_SIZE:]
    audio = np.frombuffer(data, dtype=dtype).astype(np.float32)  # Convert to float32
    # average in chunks of num_channels
    if num_channels > 1:
        if len(audio) % num_channels != 0:
            audio = audio[: -(len(audio) % num_channels)]
        audio = audio.reshape(-1, num_channels)
        audio = np.mean(audio, axis=1).astype(np.float32)  # Keep as float32
    audio = audio_resample(audio, src_sample_rate, target_sample_rate)
    # Ensure final output is float32
    audio = audio.astype(np.float32)
    if output_orig_sample_rate:
        return audio, src_sample_rate
    return audio


def audio_wav_file_to_array(
    input_path,
    target_sample_rate=TARGET_SAMPLE_RATE,
    output_orig_sample_rate=False,
):
    """Read WAV file and convert to float32 array suitable for ASR model"""
    rate, data = wavfile.read(input_path)

    # Convert to float32
    data = data.astype(np.float32)

    # Normalize to [-1, 1] range if needed
    if data.max() > 1.0:
        data = data / 32768.0  # Normalize 16-bit audio

    # Resample if needed
    if rate != target_sample_rate:
        data = audio_resample(data, rate, target_sample_rate)

    if output_orig_sample_rate:
        return data, rate
    return data
