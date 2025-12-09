# Audio Transcription & Language Detection

This project allows you to **transcribe audio** or **detect the language** of a given audio file automatically. It uses the **Whisper model** `OvozifyLabs/whisper-small-uz-v1` from Hugging Face and supports three languages:

- English
- Russian
- Uzbek

---

## Requirements

- **Python 3.10** is required for this project.

---

## Setup Instructions

| Step | Windows | Ubuntu/Linux |
|------|---------|--------------|
| 1. Create a Virtual Environment | `py -3.10 -m venv venv` | `python3.10 -m venv venv` |
| 2. Activate Virtual Environment | `venv\Scripts\activate` | `source venv/bin/activate` |
| 3. Install Required Packages (GPU) | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| 3. Install Required Packages (CPU) | `pip install -r requirements_cpu.txt` | `pip install -r requirements_cpu.txt` |
| 4. Run the Project UI | `streamlit run app.py` | `streamlit run app.py` |

---

## Notes

- Make sure your Python version is exactly **3.10** to avoid compatibility issues.
- The project automatically downloads the Whisper model from Hugging Face if it is not available locally and converts it to OpenAI Whisper module supported format.
- Supported audio input types: microphone input or uploaded audio files.

---

Enjoy transcribing and detecting languages with ease! 🎤📝

