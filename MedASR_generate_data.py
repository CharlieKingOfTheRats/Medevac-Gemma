"""
MedASR Domain Dataset Generator
Generates synthetic speech from statements, applies helicopter + radio noise,
optionally applies pitch/speed augmentations, and outputs JSON labels for Hugging Face.
"""

import os
import random
import json
from pydub import AudioSegment
import librosa
import soundfile as sf
from TTS.api import TTS

# -----------------------
# CONFIG
# -----------------------

STATEMENTS_FILE = "data/clean_statements.txt"   # 100 statements, one per line
CLEAN_AUDIO_DIR = "data/audio_clean"
NOISY_AUDIO_DIR = "data/audio_noisy"
DATASET_JSON = "data/dataset.json"

HELICOPTER_NOISE = "data/noise/helicopter.wav"
STATIC_NOISE = "data/noise/static.wav"

# Augmentation parameters
PITCH_SHIFT_STEPS = [-1, 0, 1]      # semitones
SPEED_FACTORS = [0.95, 1.0, 1.05]  # stretch factors
VOLUME_NOISE_DB = (-10, -5)         # range for helicopter volume relative to clean audio
VOLUME_STATIC_DB = (-20, -10)       # range for static volume relative to clean audio

# -----------------------
# SETUP
# -----------------------

os.makedirs(CLEAN_AUDIO_DIR, exist_ok=True)
os.makedirs(NOISY_AUDIO_DIR, exist_ok=True)

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

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

def add_noise(clean_path, heli_path, static_path, out_path):
    clean = AudioSegment.from_wav(clean_path)
    heli = AudioSegment.from_wav(heli_path)
    static = AudioSegment.from_wav(static_path)

    # Loop noises to match clean audio duration
    heli = heli * ((len(clean) // len(heli)) + 1)
    static = static * ((len(clean) // len(static)) + 1)
    heli = heli[:len(clean)]
    static = static[:len(clean)]

    # Random volume adjustments
    heli_vol = random.randint(VOLUME_NOISE_DB[0], VOLUME_NOISE_DB[1])
    static_vol = random.randint(VOLUME_STATIC_DB[0], VOLUME_STATIC_DB[1])

    mix = clean.overlay(heli + heli_vol)
    mix = mix.overlay(static + static_vol)

    mix.export(out_path, format="wav")

# -----------------------
# MAIN PROCESS
# -----------------------

dataset_entries = []

for i, text in enumerate(statements):
    print(f"[{i+1}/{len(statements)}] Generating clean TTS audio...")
    clean_path = os.path.join(CLEAN_AUDIO_DIR, f"{i:03d}.wav")
    
    # Generate base TTS audio
    tts.tts_to_file(text=text, file_path=clean_path)
    
    # Optional: Load and apply pitch/speed augmentation
    y, sr = librosa.load(clean_path, sr=None)
    y_aug = augment_audio(y, sr)
    sf.write(clean_path, y_aug, sr)
    
    # Add helicopter + static noise
    noisy_path = os.path.join(NOISY_AUDIO_DIR, f"{i:03d}.wav")
    add_noise(clean_path, HELICOPTER_NOISE, STATIC_NOISE, noisy_path)
    
    # Append JSON entry
    dataset_entries.append({
        "path": noisy_path,
        "transcription": text
    })

# -----------------------
# SAVE JSON MANIFEST
# -----------------------

with open(DATASET_JSON, "w") as f:
    json.dump(dataset_entries, f, indent=2)

print(f"\nDataset generation complete! JSON manifest saved to {DATASET_JSON}")
