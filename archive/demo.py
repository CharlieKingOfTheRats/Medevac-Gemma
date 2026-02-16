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

# =========================
# CONFIGURATION
# =========================
AUDIO_FILE = "audio/casualty.wav"
LLM_URL = "http://localhost:8080/completion"
ASR_MODEL_PATH = "./models/medasr-mil"
MAX_TOKENS = 128  # ← Reduced for faster TCCC responses
TEMP = 0.7

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """You are a Tactical Combat Casualty Care (TCCC) AI assistant.

Respond ONLY in this format:

ASSESSMENT:
ACTION:
WARNING:
"""

# =========================
# UTILITIES
# =========================

def play_audio(path):
    """Play audio file using macOS afplay"""
    print(f"\n🔊 Playing audio: {path}")
    subprocess.run(["afplay", path], check=True)

def speak(text):
    """Convert text to speech using macOS say command"""
    print("\n🗣️ AI Response (speaking out loud)...")
    # Use Alex voice for better quality (optional)
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
                "stop": ["\n\n\n"],  # Stop at triple newline
                "cache_prompt": True  # Enable prompt caching
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

def warm_asr(asr):
    """Pre-warm ASR model with silent audio"""
    print("   ⚡ Warming ASR model...")
    silent_audio = np.zeros(16000, dtype=np.float32)  # 1 sec @ 16kHz
    _ = asr({"array": silent_audio, "sampling_rate": 16000})
    print("   ✅ ASR ready")

# =========================
# MAIN DEMO
# =========================

def main():
    print("=" * 60)
    print("🚑 MedEvac-Gemma Live Demo")
    print("=" * 60)
    
    # 1️⃣ Pre-flight checks
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        return
    
    if not check_server():
        print("❌ LLM server not running!")
        print("\nStart server in another terminal:")
        print("  ./start_llm_server.sh")
        return
    
    print("✅ LLM server online")
    
    # 2️⃣ Load ASR model
    print("\n🧠 Loading MedASR-mil...")
    device = 0 if torch.backends.mps.is_available() else -1
    
    asr = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_PATH,
        device=device
    )
    
    # 3️⃣ Warm up ASR
    warm_asr(asr)
    
    print("\n" + "=" * 60)
    print("🎬 STARTING DEMO")
    print("=" * 60)
    
    # 4️⃣ Play casualty audio OUT LOUD
    play_audio(AUDIO_FILE)
    
    # 5️⃣ ASR Transcription
    print("\n🎧 Transcribing casualty report...")
    t0 = time.time()
    asr_result = asr(AUDIO_FILE, chunk_length_s=20, stride_length_s=2)
    asr_text = asr_result["text"].strip()
    asr_time = time.time() - t0
    
    print("\n📝 TRANSCRIPTION:")
    print("-" * 60)
    print(asr_text)
    print("-" * 60)
    print(f"⏱️  ASR latency: {asr_time:.2f}s")
    
    # 6️⃣ LLM Processing
    prompt = f"{SYSTEM_PROMPT}\n\nINPUT:\n{asr_text}\n\nOUTPUT:\n"
    print("\n🤖 Analyzing with MedGemma-TCCC...")
    llm_out, llm_time = run_llm(prompt)
    
    if llm_out is None:
        print("❌ LLM failed to respond")
        return
    
    print("\n📋 AI ASSESSMENT:")
    print("-" * 60)
    print(llm_out)
    print("-" * 60)
    print(f"⏱️  LLM latency: {llm_time:.2f}s")
    
    # 7️⃣ Text-to-Speech OUT LOUD
    speak(llm_out)
    
    # 8️⃣ Summary
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
    print(f"📊 Total Processing Time: {asr_time + llm_time:.2f}s")
    print(f"   - ASR:  {asr_time:.2f}s")
    print(f"   - LLM:  {llm_time:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()

# Make executable
# chmod +x demo.py

# How to run
# Terminal 1: Start server (once)
# ./start_llm_server.sh

# Terminal 2: Run demo (can run multiple times)
# python3 demo.py
# 
