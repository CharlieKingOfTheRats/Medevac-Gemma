"""
MedASR Domain Dataset Generator
Generates synthetic speech from statements, applies helicopter, radio, white noise, and wind,
optionally applies pitch/speed augmentations, and outputs JSON labels for Hugging Face.
"""

import os
import random
import json
from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf
from TTS.api import TTS

# -----------------------
# CONFIG
# -----------------------

STATEMENTS_FILE = "data/clean_statements.txt"   # 300 statements, one per line
CLEAN_AUDIO_DIR = "data/audio_clean"
NOISY_AUDIO_DIR = "data/audio_noisy"
DATASET_JSON = "data/dataset.json"

HELICOPTER_NOISE = "data/noise/helicopter.wav"
RADIO_NOISE = "data/noise/static.wav"       # your "radio" static
# Optional: wind/white noise files can be generated if missing
WHITE_NOISE = "data/noise/white.wav"
WIND_NOISE = "data/noise/wind.wav"

# Augmentation parameters
PITCH_SHIFT_STEPS = [-1, 0, 1]      # semitones
SPEED_FACTORS = [0.95, 1.0, 1.05]  # stretch factors
VOLUME_NOISE_DB = (-10, -5)         # range for helicopter/radio volume
VOLUME_AUX_DB = (-20, -10)          # range for white/wind volume

# -----------------------
# SETUP
# -----------------------

os.makedirs(CLEAN_AUDIO_DIR, exist_ok=True)
os.makedirs(NOISY_AUDIO_DIR, exist_ok=True)
os.makedirs(os.path.dirname(WHITE_NOISE), exist_ok=True)

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# -----------------------
# GENERATE WHITE/WIND NOISE IF MISSING
# -----------------------
def generate_noise(file_path, duration_ms):
    """Generates white noise of specified duration (ms)"""
    sr = 16000
    samples = np.random.normal(0, 0.02, int(sr * (duration_ms / 1000.0)))
    sf.write(file_path, samples, sr)
    print(f"Generated noise file: {file_path}")

for noise_file in [WHITE_NOISE, WIND_NOISE]:
    if not os.path.exists(noise_file):
        generate_noise(noise_file, duration_ms=3000)  # 3 sec default, will loop later

# -----------------------
# LOAD STATEMENTS
# -----------------------

with open(STATEMENTS_FILE) as f:
    statements = [line.strip() for line in f.readlines()]

# -----------------------
# FUNCTIONS
# -----------------------

def augment_audio(y, sr):
    """Apply random pitch shift and speed change"""
    pitch_shift = random.choice(PITCH_SHIFT_STEPS)
    speed_factor = random.choice(SPEED_FACTORS)
    
    if pitch_shift != 0:
        y = librosa.effects.pitch_shift(y, sr, n_steps=pitch_shift)
    if speed_factor != 1.0:
        y = librosa.effects.time_stretch(y, speed_factor)
    return y

def add_noise(clean_path, noise_files, out_path):
    clean = AudioSegment.from_wav(clean_path)
    mix = clean
    for noise_file, vol_range in noise_files:
        noise = AudioSegment.from_wav(noise_file)
        # Loop noise to match clean audio duration
        noise = noise * ((len(clean) // len(noise)) + 1)
        noise = noise[:len(clean)]
        # Random volume
        vol = random.randint(vol_range[0], vol_range[1])
        mix = mix.overlay(noise + vol)
    mix.export(out_path, format="wav")

# -----------------------
# MAIN PROCESS
# -----------------------

dataset_entries = []
noise_variants = [
    ("_heli.wav", [(HELICOPTER_NOISE, VOLUME_NOISE_DB)]),
    ("_radio.wav", [(RADIO_NOISE, VOLUME_NOISE_DB)]),
    ("_white.wav", [(WHITE_NOISE, VOLUME_AUX_DB)]),
    ("_wind.wav", [(WIND_NOISE, VOLUME_AUX_DB)])
]

for i, text in enumerate(statements[:300]):  # Use first 300 rows
    print(f"[{i+1}/300] Generating clean TTS audio...")
    clean_path = os.path.join(CLEAN_AUDIO_DIR, f"{i:03d}.wav")
    
    # Generate base TTS audio
    tts.tts_to_file(text=text, file_path=clean_path)
    
    # Optional: Load and apply pitch/speed augmentation
    y, sr = librosa.load(clean_path, sr=None)
    y_aug = augment_audio(y, sr)
    sf.write(clean_path, y_aug, sr)
    
    # Create 4 noisy versions per clean file
    for suffix, noise_list in noise_variants:
        noisy_path = os.path.join(NOISY_AUDIO_DIR, f"{i:03d}{suffix}")
        add_noise(clean_path, noise_list, noisy_path)
        dataset_entries.append({
            "path": noisy_path,
            "transcription": text
        })

# -----------------------
# SAVE JSON MANIFEST
# -----------------------

with open(DATASET_JSON, "w") as f:
    json.dump(dataset_entries, f, indent=2)

print(f"\nDataset generation complete! Total entries: {len(dataset_entries)}")