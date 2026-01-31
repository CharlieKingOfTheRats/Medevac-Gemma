# ⛑️ Medevac-Gemma

**Offline, voice-activated Tactical Combat Casualty Care (TCCC) AI assistant**
Built for military combat medics, field experimentation, and applied medical AI research.

Medevac-Gemma is a fully local **speech-to-speech** system that listens to a medic’s spoken casualty report, transcribes it using a military-tuned ASR model, reasons over it with a fine-tuned MedGemma LLM, and responds with **structured, radio-concise TCCC guidance** — all **without internet access**.

---

## ✨ Key Features

* 🎙 **Hands-free voice activation**

  * Wake word: **“MedEvac-Gemma”**
  * End transmission with: **“over”**

* 🧠 **Medical reasoning via MedGemma**

  * GGUF-based, quantized LLM
  * Structured output (Assessment / Action / Warning)
  * Deterministic, low-temperature responses

* 🗣 **Speech-to-speech loop**

  * Medic → AI → Medic → AI (continuous conversation)
  * macOS TTS for spoken responses

* 📴 **100% offline**

  * No cloud calls
  * No telemetry
  * No network dependency

* 🍎 **Optimized for Apple Silicon**

  * Tested on Mac mini M1
  * llama.cpp with Metal GPU offload

---

## 🧩 System Architecture

```
Microphone
   ↓
Military ASR (CharlieKingOfTheRats/medasr-mil)
   ↓
Conversation Manager
   ↓
MedGemma 1.5 4B Fine-tuned TCCC LLM (CharlieKingOfTheRats/medgemma-1.5-4b-tccc-q4)
   ↓
Structured TCCC Response
   ↓
macOS Text-to-Speech
```

---

## 📁 Project Structure

```text
Medevac-Gemma/
├── main.py        # Voice loop + conversation orchestration
├── audio.py       # Continuous microphone capture
├── stt.py         # Speech-to-text (medasr-mil)
├── llm.py         # MedGemma + llama.cpp interface
└── models/
    └── medgemma/
        └── medgemma-1.5-4b-it-Q4_K_M.gguf
```

---

## 🛠 Requirements

### Hardware

* Apple Silicon Mac (M1/M2/M3 recommended)
* Microphone (built-in is fine)

### Software

* Python 3.9+
* `llama.cpp` built with Metal support
* macOS (for `say` TTS)

---

## 📦 Python Dependencies

Install once:

```bash
pip install sounddevice numpy faster-whisper
```

macOS may prompt for microphone permissions — allow access.

---

## 🦙 llama.cpp Setup

You **do not** need to reinstall llama.cpp if it already works.

This project expects the `llama-simple-chat` binary, typically at:

```text
~/fiercecoyote/llama.cpp/build/bin/llama-simple-chat
```

Confirm it works:

```bash
/path/to/llama-simple-chat \
  -m /path/to/medgemma-1.5-4b-it-Q4_K_M.gguf \
  -c 2048 \
  -ngl 35
```

Update `llm.py` with **absolute paths** to:

* `LLAMA_BIN`
* `MODEL_PATH`

---

## ▶️ Running Medevac-Gemma

From the project root:

```bash
python main.py
```

### Voice Flow

1. Program starts listening silently
2. Medic says:

   > **“MedEvac-Gemma”**
3. System responds:

   > *“Go ahead”*
4. Medic speaks scenario
5. Medic says:

   > **“over”**
6. AI responds with structured TCCC guidance (spoken + printed)
7. Repeat as needed

---

## 📋 Output Format

The AI responds **only** in the following structure:

```
ASSESSMENT:
<brief medical assessment>

ACTION:
<step-by-step immediate actions>

WARNING:
<critical risks or red flags>
```

Designed for:

* radio brevity
* cognitive load reduction
* field usability

---

## 📴 Offline Operation

Medevac-Gemma runs fully offline once models are present locally:

* ✅ Local ASR model
* ✅ Local LLM (GGUF)
* ✅ Local inference via llama.cpp

No Wi-Fi, cellular, or cloud services required.

---

## ⚠️ Disclaimer

This project is **experimental** and **not a medical device**.

It is intended for:

* research
* training
* prototyping
* human-in-the-loop decision support

All medical decisions remain the responsibility of the human operator.

---

## 🚀 Future Work

Planned or possible extensions:

* Push-to-talk fallback mode
* Voice Activity Detection (VAD)
* Streaming token-level TTS
* Persistent llama.cpp process
* Encrypted conversation logging
* Jetson / ARM deployment
* Body-worn or vehicle-mounted integration

---

## 👤 Author

**CharlieKingOfTheRats**
Applied AI • Medical AI • Defense-adjacent systems

---
