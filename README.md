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
├── demo1.py                       # Pre-recorded demo script
├── demo2.py                       # Alternate demo scenario
├── chat.py                        # Interactive push-to-talk chat mode
├── start_llm_server.sh            # llama-server launcher
├── requirements.txt               # Python dependencies
├── audio/                         # Demo audio files
│   ├── Demo1.wav                  # Moderate helicopter noise scenario
│   └── Demo2.wav                  # Heavy noise + fragmented speech
├── medevac-gemma_notebooks/       # Training, evaluation, and utility notebooks
│   ├── medasr_noise/              # Noise augmentation assets/scripts
│   ├── Lora_to_gguf.ipynb         # Convert LoRA adapter → GGUF format
│   ├── MedASR_Dataset_Generator.ipynb    # Synthetic ASR dataset generation
│   ├── MedGemma_finetune_final.ipynb     # LLM fine-tuning notebook
│   ├── medasr_military_fine_tune_w.ipynb # ASR fine-tuning notebook
│   ├── medasr_prompts.csv         # ASR training prompt templates
│   └── full_eval 2.csv            # Full evaluation results
└── archive/                       # Legacy scripts and earlier iterations
```

> **Note:** Models are not included in the repo. See [Download Models](#-download-models) below.

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

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This installs:
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
- Place in `./models/medgemma-tccc-q4.gguf` (create the `models/` directory if needed)

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
python3 demo1.py
```

For the alternate scenario:
```bash
python3 demo2.py
```

---

### Option 2: Interactive Push-to-Talk Chat

**Terminal 1** - Start LLM Server:
```bash
./start_llm_server.sh
```

**Terminal 2** - Run Interactive Chat:
```bash
source venv/bin/activate
python3 chat.py
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

YouTube demo: https://m.youtube.com/watch?v=uNWsgeHPV0M

---

## 🧪 Training & Evaluation Notebooks

All notebooks are in `medevac-gemma_notebooks/`:

### ASR Fine-Tuning (`medasr_military_fine_tune_w.ipynb`)
- **Dataset:** [medasr-military-1300](https://huggingface.co/datasets/CharlieKingOfTheRats/medasr-military-1300)
- **Base Model:** google/medasr
- **Fine-tuning:** 19 epochs on synthetic combat audio
- **Result:** 64% WER reduction vs baseline

### LLM Fine-Tuning (`MedGemma_finetune_final.ipynb`)
- **Dataset:** [medgemma_tccc](https://huggingface.co/datasets/CharlieKingOfTheRats/medgemma_tccc)
- **Base Model:** google/medgemma-1.5-4b-it
- **Fine-tuning:** LoRA (r=16, alpha=32), 3 epochs
- **Result:** 21% TCCC protocol coverage improvement

### Dataset Generation (`MedASR_Dataset_Generator.ipynb`)
- Generates synthetic combat audio training data
- Includes noise augmentation via `medasr_noise/`
- Prompt templates in `medasr_prompts.csv`

### LoRA → GGUF Conversion (`Lora_to_gguf.ipynb`)
- Converts trained LoRA adapter to quantized GGUF format for llama.cpp deployment

### Evaluation Results (`full_eval 2.csv`)
- Full reproducible evaluation pipeline output
- Metrics: WER, TCCC Score, Latency, Failure Analysis
- Test Set: n=30 samples with varied acoustic conditions

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

Designed for radio brevity, cognitive load reduction, field usability, and TCCC protocol compliance.

---

## 📴 Offline Operation

MedEvac-Gemma runs fully offline once models are loaded:

✅ Local ASR model (cached after first download)  
✅ Local LLM (GGUF file)  
✅ Local inference via llama.cpp  
✅ No Wi-Fi, cellular, or cloud services required

Perfect for denied communications environments, GPS-denied operations, austere medical facilities, and disaster response zones.

---

## 🔧 Configuration

Edit `demo1.py` or `start_llm_server.sh` to customize:

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

## 📊 Performance Metrics

| Metric | Custom | Baseline | Improvement |
|--------|--------|----------|-------------|
| WER (ASR) | 0.144 ± 0.043 | 0.402 ± 0.052 | **64.2%** |
| TCCC Score | 0.610 ± 0.079 | 0.502 ± 0.085 | **21.5%** |
| Latency | 4-6s | - | Sub-7s target |

Full evaluation data available in `medevac-gemma_notebooks/full_eval 2.csv`.

---

## ⚠️ Disclaimer

**This project is experimental and not a medical device.**

It is intended for research and development, training and education, prototyping and demonstration, and human-in-the-loop decision support.

**All medical decisions remain the responsibility of the human operator.**

---

## 🚀 Future Work

- [ ] Specialized medical TTS model (recommended for Google HAI-DEF)
- [ ] Multi-casualty triage mode
- [ ] Voice authentication for OPSEC
- [ ] Bi-directional radio integration
- [ ] Extended battery optimization
- [ ] Ruggedized hardware deployment
- [ ] Real-time vital sign integration

---

## 📚 Citations

```bibtex
@software{medevac_gemma_2025,
  author = {Donnelly, Charles},
  title = {MedEvac-Gemma: Edge-Deployed Speech-to-Speech Medical AI},
  year = {2026},
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
