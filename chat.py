#!/usr/bin/env python3
"""
MedEvac-Gemma Push-to-Talk Demo
Record audio → ASR → LLM → TTS (all out loud)
Press and hold SPACE to record, release to process
Press Q to quit
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
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
import tempfile

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# =========================
# CONFIGURATION
# =========================
LLM_URL = "http://localhost:8080/completion"
ASR_MODEL_PATH = "/Users/fiercecoyote/medevac-gemma/medevac-gemma/medasr-mil"
MAX_TOKENS = 90 # This can be adjusted based on expected response length and latency requirements
TEMP = 0.7
SAMPLE_RATE = 16000

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
# GLOBAL STATE
# =========================
is_recording = False
recording_data = []
asr_pipeline = None
quit_flag = False

# =========================
# UTILITIES
# =========================

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
    """Check if internet is available"""
    try:
        requests.get("https://www.google.com", timeout=2)
        return True
    except:
        return False

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

def audio_callback(indata, frames, time_info, status):
    """Callback for audio recording"""
    global recording_data
    if is_recording:
        recording_data.append(indata.copy())

def on_press(key):
    """Handle key press"""
    global is_recording, recording_data
    
    try:
        if key == keyboard.Key.space and not is_recording:
            is_recording = True
            recording_data = []
            print("\n🔴 RECORDING... (release SPACE to stop)")
        elif key.char == 'q':
            global quit_flag
            quit_flag = True
            return False
    except AttributeError:
        pass

def on_release(key):
    """Handle key release"""
    global is_recording
    
    if key == keyboard.Key.space and is_recording:
        is_recording = False
        print("⏹ Recording stopped, processing...")
        process_recording()

def process_recording():
    """Process the recorded audio"""
    global recording_data, asr_pipeline
    
    if len(recording_data) == 0:
        print("❌ No audio recorded")
        return
    
    # Combine audio chunks
    audio = np.concatenate(recording_data, axis=0)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio, SAMPLE_RATE)
    temp_file.close()
    
    try:
        # Transcribe
        print("🎧 Transcribing...")
        t0 = time.time()
        asr_result = asr_pipeline(temp_file.name, chunk_length_s=20, stride_length_s=2)
        asr_text = clean_transcription(asr_result["text"])
        asr_time = time.time() - t0
        
        # print(f"\n📝 Transcribed: {asr_text}") # optional
        
        # Get LLM response
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
        print(f"\nProcessing time: {asr_time + llm_time:.2f}s")
        
        # Speak response
        speak(llm_out)
        
    finally:
        # Clean up temp file
        os.unlink(temp_file.name)
    
    print("\n💬 Ready for next input (SPACE to talk, Q to quit)...")

def initialize_system():
    """Initialize ASR model and check systems"""
    global asr_pipeline
    
    print("=" * 60)
    print("MEDEVAC-GEMMA PUSH-TO-TALK SYSTEM")
    print("=" * 60)
    
    # Check internet
    internet = check_internet()
    if internet:
        print("⚠ Internet detected (not required for operation)")
    else:
        print("✓ Running offline (no internet connection)")
    
    # Check LLM server
    if not check_server():
        print("❌ LLM server not running!")
        print("\nStart server in another terminal:")
        print("  ./start_llm_server.sh")
        return False
    
    print("✓ LLM server online")
    
    # Load ASR model
    print("\n📡 Loading MedASR-mil locally...")
    device = 0 if torch.backends.mps.is_available() else -1
    
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_PATH,
        device=device,
        trust_remote_code=True
    )
    
    print("✓ ASR ready")
    
    return True

# =========================
# MAIN
# =========================

def main():
    if not initialize_system():
        return
    
    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE - Hold to record, release to process")
    print("  Q     - Quit")
    print("\n💬 Ready for input (SPACE to talk)...\n")
    
    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback
    )
    
    # Start keyboard listener
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    
    with stream:
        listener.start()
        
        # Keep running until quit
        while not quit_flag:
            time.sleep(0.1)
        
        listener.stop()
    
    print("\n" + "=" * 60)
    print("SYSTEM SHUTDOWN")
    print("=" * 60)

if __name__ == "__main__":
    main()
