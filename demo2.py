#!/usr/bin/env python3
"""
MedEvac-Gemma Live Demo
Plays audio → ASR → LLM → TTS (all out loud)
Requires: llama-server running on port 8080
"""

import subprocess
import time
import os
import requests
from transformers import pipeline
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# =========================
# CONFIGURATION
# =========================
AUDIO_FILE = "/Users/fiercecoyote/medevac-gemma/audio/Demo2.wav"
LLM_URL = "http://localhost:8080/completion"
ASR_MODEL_PATH = "/Users/fiercecoyote/medevac-gemma/medevac-gemma/medasr-mil"
MAX_TOKENS = 90 # was 128, reduced for faster response
TEMP = 0.7

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """You are MedEvac-Gemma, a Tactical Combat Casualty Care (TCCC) AI assistant.

Provide ONLY this format with NO additional text. STOP after WARNING. Do not add notes, confirmations, or additional instructions.

ASSESSMENT:
[One sentence: patient status and injuries]

ACTION:
[Numbered list: 3-4 immediate interventions]

WARNING:
[One sentence: critical concern]

"""

# =========================
# UTILITIES
# =========================

def play_audio(path):
    """Play audio file using macOS afplay"""
    print("\n▶ Playing casualty audio...")
    subprocess.run(["afplay", path], check=True)

def speak(text):
    """Convert text to speech using macOS say command"""
    print("\n🗣 AI Response (speaking)...")
    subprocess.run(["say", "-v", "Alex", text])

def run_llm(prompt):
    """Query persistent llama.cpp server via HTTP"""
    start = time.time()
    
    try:
        response = requests.post(
            LLM_URL,
            json={
                "prompt": prompt,
                "n_predict": MAX_TOKENS,
                "temperature": TEMP,
                "stop": ["\n\n\n"],
                "cache_prompt": True
            },
            timeout=30
        )
        response.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ LLM server error: {e}")
        return None, 0
    
    latency = time.time() - start
    return response.json()["content"].strip(), latency

def check_server():
    """Verify llama-server is running"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_internet():
    """Check if internet is available (we DON'T need it!)"""
    try:
        requests.get("https://www.google.com", timeout=2)
        return True
    except:
        return False

def warm_asr(asr):
    """Pre-warm ASR model with silent audio"""
    print("  ⚡ Initializing ASR model...")
    silent_audio = np.zeros(16000, dtype=np.float32)
    _ = asr({"array": silent_audio, "sampling_rate": 16000})
    print("  ✓ ASR ready")

def clean_transcription(text):
    """Remove <epsilon> tokens and clean up the transcription"""
    import re
    
    # Remove special tokens
    text = text.replace("<epsilon>", "")
    text = text.replace("<s>", "").replace("</s>", "")
    text = text.replace("[", "").replace("]", "")
    
    # Remove duplicate consecutive characters
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    # Remove duplicate consecutive words
    words = text.split()
    cleaned_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i-1].lower():
            cleaned_words.append(word)
    text = " ".join(cleaned_words)
    
    # Clean up extra spaces
    text = " ".join(text.split())
    return text.strip()

# =========================
# MAIN DEMO
# =========================

def main():
    print("=" * 60)
    print("MEDEVAC-GEMMA SYSTEM")
    print("=" * 60)
    
    # Pre-flight checks
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        return
    
    # Check internet (we don't need it, but show we're offline-capable)
    internet = check_internet()
    if internet:
        print("⚠ Internet detected (not required for operation)")
    else:
        print("✓ Running offline (no internet connection)")
    
    if not check_server():
        print("❌ LLM server not running!")
        print("\nStart server in another terminal:")
        print("  ./start_llm_server.sh")
        return
    
    print("✓ LLM server online")
    
    # Load ASR model
    print("\n📡 Loading MedASR-mil locally...")
    device = 0 if torch.backends.mps.is_available() else -1
    
    asr = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_PATH,
        device=device,
        trust_remote_code=True
    )
    
    # Warm up ASR
    warm_asr(asr)
    
    print("\n" + "=" * 60)
    print("SYSTEM READY - STANDBY FOR CASUALTY REPORT")
    print("=" * 60)
    
    # Play casualty audio
    play_audio(AUDIO_FILE)
    
    # ASR Transcription
    print("\n🎧 Transcribing...")
    t0 = time.time()
    asr_result = asr(AUDIO_FILE, chunk_length_s=20, stride_length_s=2)
    asr_text = clean_transcription(asr_result["text"])
    asr_time = time.time() - t0
    
    # LLM Processing
    prompt = f"{SYSTEM_PROMPT}\n\nINPUT:\n{asr_text}\n\nOUTPUT:\n"
    print("🤖 Analyzing with MedGemma-4B-TCCC...")
    llm_out, llm_time = run_llm(prompt)
    
    if llm_out is None:
        print("❌ LLM failed to respond")
        return
    
    print("\n" + "=" * 60)
    print("TCCC ASSESSMENT")
    print("=" * 60)
    print(llm_out)
    print("=" * 60)
    
    # Text-to-Speech
    speak(llm_out)
    
    # Summary
    print("\n" + "=" * 60)
    print("ASSESSMENT COMPLETE")
    print("=" * 60)
    print(f"Total Response Time: {asr_time + llm_time:.2f}s")
    print("=" * 60)
    print("\nSystem ready for next casualty report...\n")

if __name__ == "__main__":
    main()