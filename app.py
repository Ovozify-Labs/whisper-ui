import streamlit as st
import whisper
import tempfile
import os
import shutil
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from converter import convert

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set page config
st.set_page_config(
    page_title="Whisper Audio Transcription",
    page_icon="🎤",
    layout="wide"
)

# Title and description
st.title("🎤 Audio Transcription with Whisper")
st.markdown("Upload an audio file to transcribe or detect its language using OpenAI's Whisper model.")

# Initialize session state
if "mode" not in st.session_state:
    st.session_state.mode = None

# Buttons to switch modes
st.session_state.mode = st.radio("Select mode: ", ("Transcribe", "Detect language"))
        
model_name = "OvozifyLabs/whisper-small-uz-v1"

# Load model with caching
@st.cache_resource
def load_and_convert_model(model_name: str):
    if os.path.exists("models/whisper"):
        return whisper.load_model("models/whisper", device=device)
    else:
        # Download huggingface repo → local folder
        repo_dir = snapshot_download(repo_id=model_name)

        # Convert to Whisper supported format (your function)
        convert(repo_dir, base="small", out="models/whisper", cpu=False)

        # Delete HuggingFace model after conversion
        # shutil.rmtree(repo_dir, ignore_errors=True)

        # Load converted OpenAI Whisper-format model
        return whisper.load_model("models/whisper", device=device)

if st.session_state.mode == "Transcribe":
    st.header("📝 Transcribe Audio")
    audio_file_transcribe = st.file_uploader(
        "Choose an audio file for transcription",
        type=["mp3", "wav", "m4a", "flac", "ogg"],
        key="transcribe_file"
    )

    if audio_file_transcribe is not None:
        st.audio(audio_file_transcribe)
        if st.button("Transcribe", key="transcribe_btn"):
            with st.spinner(f"Loading {model_name} model..."):
                model = load_and_convert_model(model_name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file_transcribe.name).suffix) as tmp_file:
                tmp_file.write(audio_file_transcribe.read())
                tmp_path = tmp_file.name

            try:
                with st.spinner("Transcribing audio..."):
                    result = model.transcribe(tmp_path)
                st.success("Transcription completed!")
                st.text_area("Transcription Result", result["text"], height=200)
                st.info(f"Detected Language: {result['language']}")
            finally:
                os.unlink(tmp_path)

# --- Language Detection UI ---
elif st.session_state.mode == "Detect language":
    st.header("🌐 Detect Language")
    audio_file_detect = st.file_uploader(
        "Choose an audio file for language detection",
        type=["mp3", "wav", "m4a", "flac", "ogg"],
        key="detect_file"
    )

    if audio_file_detect is not None:
        st.audio(audio_file_detect)
        if st.button("Detect Language", key="detect_btn"):
            with st.spinner(f"Loading {model_name} model..."):
                model = load_and_convert_model(model_name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file_detect.name).suffix) as tmp_file:
                tmp_file.write(audio_file_detect.read())
                tmp_path = tmp_file.name

            try:
                with st.spinner("Detecting language..."):
                    audio = whisper.load_audio(tmp_path)
                    audio = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(audio).to(model.device)
                    _, probs = model.detect_language(mel)
                    detected_lang = max(probs, key=probs.get)

                st.success("Language detection completed!")
                st.markdown(f"### Detected Language: {detected_lang.upper()}")

                st.subheader("Language Probabilities (Top 5):")
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                for lang, prob in sorted_probs:
                    st.progress(prob, text=f"{lang}: {prob:.2%}")
            finally:
                os.unlink(tmp_path)