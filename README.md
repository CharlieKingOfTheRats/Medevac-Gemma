# ⛑️ MedEvac-Gemma

**Offline, Push-to-Talk Tactical Combat Casualty Care (TCCC) AI Assistant**

Built for military combat medics, disaster response, and applied medical AI research.

MedEvac-Gemma is a fully local speech-to-speech system that processes a medic's spoken casualty report, transcribes it using a military-tuned ASR model, reasons over it with a fine-tuned MedGemma LLM, and responds with structured, radio-concise TCCC guidance — all without internet access.

---

## ✨ Key Features

- 🎙 **Push-to-Talk Interface** - Hold SPACE to record, release to process (no wake word needed)
- 🧠 **Medical Reasoning via MedGemma** - Fine-tuned GGUF LLM with structured TCCC output
- 🗣 **Speech-to-Speech Pipeline** - Complete audio input → AI guidance → audio output
- 📴 **100% Offline** - No cloud calls, no telemetry, no network dependency
- 🍎 **Optimized for Apple Silicon** - Tested on Mac mini M1 with Metal GPU acceleration
- ⚡ **Sub-7s Response Time** - ASR (0.3s) + LLM (3-5s) + TTS (1s)

---

## 🧩 System Architecture

```
Microphone (Push-to-Talk)
   ↓
Military ASR (medasr-mil)
   ↓
MedGemma 1.5 4B TCCC LLM (quantized 4-bit)
   ↓
Structured TCCC Response
   ↓
macOS Text-to-Speech
```

**Pipeline:** Audio → Transcription → Medical Reasoning → Spoken Guidance (4-6s total)

---

## 📁 Project Structure

```
Medevac-Gemma/
├── demo.py                    # Main demo script (recommended)
├── demo_clean.py              # Polished demo with minimal output
├── demo_chat.py               # Interactive push-to-talk chat mode
├── start_llm_server.sh        # llama-server launcher
├── setup.sh                   # Virtual environment setup
├── requirements.txt           # Python dependencies
├── audio/
│   ├── Demo1.wav             # Moderate helicopter noise scenario
│   └── Demo2.wav             # Heavy noise + fragmented speech
├── notebooks/
│   ├── MedASR_Training.ipynb         # ASR fine-tuning notebook
│   ├── MedGemma_Training.ipynb       # LLM fine-tuning notebook
│   └── Evaluation_Notebook.ipynb    # Performance evaluation
└── models/
    └── medgemma-tccc-q4.gguf         # Fine-tuned, quantized LLM
```

---

## 🛠 Requirements

### Hardware
- **Apple Silicon Mac** (M1/M2/M3/M4 recommended)
- **16GB+ RAM** (for optimal LLM performance)
- **Microphone** (built-in or external)

### Software
- **Python 3.10+** (3.11 recommended)
- **llama.cpp** with Metal support
- **macOS** (for native TTS)
- **ffmpeg** (for audio processing)

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/CharlieKingOfTheRats/Medevac-Gemma.git
cd Medevac-Gemma
```

### 2. Install Dependencies

```bash
chmod +x setup.sh
./setup.sh
```

This creates a virtual environment and installs:
- PyTorch (with Metal support)
- Transformers
- Audio processing libraries (sounddevice, soundfile)
- HTTP client (requests)

### 3. Install ffmpeg

```bash
brew install ffmpeg
```

### 4. Download Models

Models are hosted on HuggingFace:

**ASR Model:**
```bash
# Auto-downloads on first run, or manually:
# Location: ~/.cache/huggingface/hub/
```
Model: [CharlieKingOfTheRats/medasr-mil](https://huggingface.co/CharlieKingOfTheRats/medasr-mil)

**LLM Model:**
Download quantized GGUF model:
- [medgemma-1.5-4b-tccc-q4.gguf](https://huggingface.co/CharlieKingOfTheRats/medgemma-1.5-4b-tccc-q4)
- Place in `./models/medgemma-tccc-q4.gguf`

---

## ▶️ Running MedEvac-Gemma

### Option 1: Pre-recorded Demo (Recommended for First Run)

**Terminal 1** - Start LLM Server:
```bash
chmod +x start_llm_server.sh
./start_llm_server.sh
```

**Terminal 2** - Run Demo:
```bash
source venv/bin/activate
python3 demo.py
```

This plays pre-recorded scenarios and shows system responses.

---

### Option 2: Interactive Push-to-Talk Chat

**Terminal 1** - Start LLM Server:
```bash
./start_llm_server.sh
```

**Terminal 2** - Run Interactive Chat:
```bash
source venv/bin/activate
python3 demo_chat.py
```

**Controls:**
- **Hold SPACE** - Record your casualty report
- **Release SPACE** - Process and get AI response
- **Press Q** - Quit

---

## 🎤 Usage Example

```
============================================================
MEDEVAC-GEMMA SYSTEM READY
============================================================

[Hold SPACE and speak]
"Male casualty, blast injury to left leg, heavy bleeding 
controlled with tourniquet. Patient conscious, breathing 
rapidly, weak pulse."
[Release SPACE]

AI ASSESSMENT (spoken + printed):
------------------------------------------------------------
ASSESSMENT:
Male patient with controlled left leg hemorrhage via tourniquet.
Patient conscious, breathing rapidly, weak pulse indicates shock.

ACTION:
1. Maintain tourniquet, monitor for re-bleeding
2. Administer oxygen if available
3. Monitor vital signs every 5 minutes
4. Prepare for immediate evacuation

WARNING:
Rapid pulse indicates potential shock. Monitor for changes in consciousness.
------------------------------------------------------------
⏱ Processing time: 5.2s
```

---

## 📊 Demo Scenarios

Two scenarios included to demonstrate robustness:

**Demo1.wav** - Moderate Helicopter Noise
- Standard casualty report
- Intermittent rotor and radio static
- Tests ASR under stress-distorted speech

**Demo2.wav** - Heavy Noise + Fragmented Speech
- Penetrating trauma scenario
- Loud continuous rotor noise
- Severely fragmented transmission
- Tests ASR limits and LLM robustness

---

## 🧪 Training Notebooks

All training and evaluation notebooks are in `notebooks/`:

### ASR Training
- **Dataset:** [medasr-military-1300](https://huggingface.co/datasets/CharlieKingOfTheRats/medasr-military-1300)
- **Base Model:** google/medasr
- **Fine-tuning:** 19 epochs on synthetic combat audio
- **Result:** 64% WER reduction vs baseline

### LLM Training
- **Dataset:** [medgemma_tccc](https://huggingface.co/datasets/CharlieKingOfTheRats/medgemma_tccc)
- **Base Model:** google/medgemma-1.5-4b-it
- **Fine-tuning:** LoRA (r=16, alpha=32), 3 epochs
- **Result:** 21% TCCC protocol coverage improvement

### Evaluation
- **Metrics:** WER, TCCC Score, Latency, Failure Analysis
- **Test Set:** n=30 samples with varied acoustic conditions
- **Notebook:** Full reproducible evaluation pipeline

---

## 📋 Output Format

AI responses follow structured TCCC format:

```
ASSESSMENT:
[Brief patient status and injuries]

ACTION:
[Numbered list of immediate interventions]

WARNING:
[Critical safety concerns or time-sensitive issues]
```

Designed for:
- Radio brevity
- Cognitive load reduction
- Field usability
- TCCC protocol compliance

---

## 📴 Offline Operation

MedEvac-Gemma runs fully offline once models are loaded:

✅ Local ASR model (cached after first download)  
✅ Local LLM (GGUF file)  
✅ Local inference via llama.cpp  
✅ No Wi-Fi, cellular, or cloud services required

Perfect for:
- Denied communications environments
- GPS-denied operations
- Austere medical facilities
- Disaster response zones

---

## 🔧 Configuration

Edit `demo.py` or `start_llm_server.sh` to customize:

**ASR Settings:**
```python
ASR_MODEL_PATH = "CharlieKingOfTheRats/medasr-mil"
```

**LLM Settings:**
```bash
-m ./models/medgemma-tccc-q4.gguf  # Model path
-ngl 99                             # GPU layers
-c 1024                             # Context size
-t 6                                # Threads
```

**Audio Settings:**
```python
SAMPLE_RATE = 16000
```

---

## 🎯 Performance Metrics

| Metric | Custom | Baseline | Improvement |
|--------|--------|----------|-------------|
| WER (ASR) | 0.144 ± 0.043 | 0.402 ± 0.052 | **64.2%** |
| TCCC Score | 0.610 ± 0.079 | 0.502 ± 0.085 | **21.5%** |
| Latency | 4-6s | - | Sub-7s target |

---

## ⚠️ Disclaimer

**This project is experimental and not a medical device.**

It is intended for:
- Research and development
- Training and education
- Prototyping and demonstration
- Human-in-the-loop decision support

**All medical decisions remain the responsibility of the human operator.**

---

## 🚀 Future Work

Planned enhancements:

- [ ] Specialized medical TTS model (recommended for Google HAI-DEF)
- [ ] Multi-casualty triage mode
- [ ] Voice authentication for OPSEC
- [ ] Bi-directional radio integration
- [ ] Extended battery optimization
- [ ] Ruggedized hardware deployment
- [ ] Real-time vital sign integration

---

## 📚 Citations

If you use this work, please cite:

```bibtex
@software{medevac_gemma_2025,
  author = {Donnelly, Charles},
  title = {MedEvac-Gemma: Edge-Deployed Speech-to-Speech Medical AI},
  year = {2025},
  url = {https://github.com/CharlieKingOfTheRats/Medevac-Gemma}
}
```

**Models:**
- ASR: [CharlieKingOfTheRats/medasr-mil](https://huggingface.co/CharlieKingOfTheRats/medasr-mil)
- LLM: [CharlieKingOfTheRats/medgemma-1.5-4b-tccc-lora](https://huggingface.co/CharlieKingOfTheRats/medgemma-1.5-4b-tccc-lora)

**Datasets:**
- ASR: [medasr-military-1300](https://huggingface.co/datasets/CharlieKingOfTheRats/medasr-military-1300)
- LLM: [medgemma_tccc](https://huggingface.co/datasets/CharlieKingOfTheRats/medgemma_tccc)

---

## 👤 Author

**Charles Donnelly**  
Applied AI • Defense Systems

For questions or collaboration: [GitHub Issues](https://github.com/CharlieKingOfTheRats/Medevac-Gemma/issues)

---

## 📄 License

Apache 2.0 - See LICENSE file for details

---

## 🏆 Acknowledgments

Built using:
- **Google Health AI Developer Foundations (HAI-DEF)** - MedGemma and MedASR base models
- **llama.cpp** - Efficient LLM inference
- **Transformers** - HuggingFace model ecosystem

Part of **The MedGemma Impact Challenge** submission.
