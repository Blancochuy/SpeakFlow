# 🎙️ SpeakFlow

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)

**SpeakFlow** is a powerful real-time speech transcription and translation tool that seamlessly captures system audio, transcribes it using OpenAI's Whisper large-v3-turbo model, and provides instant translations. Built for speed with CUDA acceleration and an intuitive interface.

---

## 📹 Demo

Check out the demo video below:
# Transcription
[![Watch the video](https://img.youtube.com/vi/-lypvEV-zt4/mqdefault.jpg)](https://youtu.be/-lypvEV-zt4)

# Translation
[![Watch the video](https://img.youtube.com/vi/ytvD_oIRmM4/mqdefault.jpg)](https://youtu.be/ytvD_oIRmM4)

---

## ✨ Key Features

- 🎯 **Zero-latency Capture**: Records system audio in real-time via loopback
- 🌍 **Multilingual**: Translates to English, Spanish, French, and German
- 🔊 **Smart Audio Processing**: Automatic noise reduction and speech enhancement
- ⚡ **GPU Accelerated**: Optimized performance with CUDA support
- 💫 **Sleek Interface**: Smooth typing animations and word counting
- 🧠 **Intelligent Processing**: Removes duplicates and handles overlapping text

---

## 🚀 Quick Start

### 📌 Prerequisites
Make sure you have **Python 3.8+** installed.

```bash
python --version
```

### 🔽 Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/realtime-transcription.git
cd realtime-transcription
```

### 📦 Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 🔧 Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- `torch`, `torchaudio`, `transformers`
- `pyaudiowpatch` (for loopback recording)
- `noisereduce` (for noise reduction)
- `tkinter` (for GUI)
- `langdetect` (for automatic language detection)

### ⚡ Step 4: Run the Application
```bash
python main.py
```

---

## 🎛️ Usage
1. **Start the transcription**: Click the `Start` button to begin transcribing.
2. **Select output language**: Choose **English, Spanish, French, or German** from the dropdown menu.
3. **View live transcription**: The transcribed text will appear with a **typing animation** in real-time.
4. **Stop the transcription**: Click the `Stop` button to end the session.
5. **Clear text**: Click the `Clear` button to reset the transcription area.

---

## 📁 Project Structure

```
SpeakFLow/
│
├── main.py               # Entry point (instantiates manager and GUI)
├── transcription_manager.py  # Contains the TranscriptionManager class
├── gui_app.py            # Contains the Application class
├── audio_utils.py        # Contains helper functions (e.g., remove_repeated_words, remove_overlap)
└── logging_config.py     # (optional) Handles logging configuration
```

---

## ⚡ Optimizations
🔹 **GPU Acceleration**: Uses **CUDA (torch.float16)** for faster transcription.  
🔹 **Multi-threading**: Runs **audio recording, transcription, and UI updates in parallel**.  
🔹 **Noise Reduction**: Filters out background noise for **clearer** transcriptions.  
🔹 **Language Detection**: Uses `langdetect` to **automatically detect the language** of the audio.  

---

## 🛠️ Troubleshooting

### ❌ Issue: "No WASAPI loopback device found"
- Ensure your **system audio device** supports loopback recording.
- On **Windows**, run:
  ```bash
  python -c "import pyaudiowpatch; print(pyaudiowpatch.PyAudio().get_default_wasapi_loopback())"
  ```
- If it returns `None`, try changing your **default audio device**.

### ❌ Issue: "CUDA not being used"
- Check if PyTorch detects your **GPU**:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- If it returns `False`, ensure **CUDA & cuDNN** are installed correctly.

### ❌ Issue: "Transcription is slow"
- Reduce **batch size** in `config.py`:
  ```python
  BATCH_SIZE = 2  # Lower batch size for faster response
  ```
- Try **disabling translation** by setting the language to `English`.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 💡 Future Improvements
✅ Add **real-time microphone support**.  
✅ Optimize UI responsiveness for **faster text updates**.  

---

## 🌟 Credits
- Built with **Python, PyTorch, Hugging Face Transformers, and Tkinter**.
- Uses **OpenAI's Whisper model** for high-accuracy transcription.
- Thanks to **Helsinki-NLP** for their open-source translation models.

---

### 🚀 **Start transcribing in real-time today!**
```bash
python main.py