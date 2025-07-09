#!/usr/bin/env python3
"""
Test script to demonstrate phoneme frequency dictionary with random sequences.
"""
import numpy as np
import scipy.io.wavfile as wavfile
from feedback import get_all_phoneme_pairs, user_phonetic_errors, phoneme_written_feedback, get_fastdtw_aligned_phoneme_lists, group_phonemes
from server import transcribe_audio, get_user_word_audio_sample
rando_audio = "/Users/arunasrivastava/Koel/koellabs.com/server/src/aruna-er-short.wav"
target_phonemes = "kʰɔlɪŋkʰɑɹdzɑɹðəweɪvəvðəfjutʃɝ"
target_by_words = [
        ['calling', 'kʰɔlɪŋ'],
        ['cards', 'kʰɑɹdz'],
        ['are', 'ɑɹ'],
        ['the', 'ðʌ'],
        ['wave', 'weɪv'],
        ['of', 'əv'],
        ['the', 'ðʌ'],
        ['future', 'fjutʃɝ']
]
speech_phonemes = "kʰɔlɪŋbɑɹdsɑɹðʌweɪvʌvðʌfjupɝ"
    # Build frequency dictionary
sample_rate, audio = wavfile.read(rando_audio)
# int 16 float32
audio = audio.astype(np.float32)
transcribed, timestamps = transcribe_audio(audio, include_timestamps=True)
transscript_er = "oʊvɝðbɹdʒoʊvɛɹðʌɹɪdʒ"
print("sampling rate", sample_rate)
phone, start, end = timestamps[0]
phone_next, start_next, end_next = timestamps[1]
print(start)
start_sample = int(start * sample_rate)
end_sample = int(end_next * sample_rate)
wavfile.write("aruna-er-short-oʊvɝ.wav", sample_rate, audio[start_sample:end_sample])

# mistake_freq = user_phonetic_errors(target_phonemes, target_by_words, speech_phonemes)
# contents = phoneme_written_feedback(target_phonemes, speech_phonemes)

# for phoneme, info in mistake_freq.items():
#     incorrect_phones = info[3]
#     words = info[1]
#     for incorrect_phone in incorrect_phones:
#         print(f"{contents[phoneme]["phonetic-spelling"]}: found in the words {words} sounds like you pronounce it as {contents[incorrect_phone]["phonetic-spelling"]}")
    
# print("written feedback", phoneme_written_feedback(target_phonemes, speech_phonemes, "model_vocab_feedback.json"))

# sort the mistake_freq by count and then most concerning mistakes (based on Feature error rate panphon)


