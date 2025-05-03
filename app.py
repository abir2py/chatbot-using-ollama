import streamlit as st
import whisper
import io
from streamlit_mic_recorder import mic_recorder, speech_to_text

@st.cache_resource
def load_model(model_size):
    return whisper.load_model(model_size)

def robust_speech_recognition(audio_bytes, model_size="medium", language=None):
    """Transcribe audio using Whisper."""
    model = load_model(model_size)
    
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)

    result = model.transcribe("temp_audio.wav", language=language, fp16=False)
    return result["text"]

st.title("Speech Recognition App")

option = st.radio("Select Input Method", ["Upload an Audio File", "Record from Microphone"])

if option == "Upload an Audio File":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        model_size = st.selectbox("Select Model Size", ["tiny", "base", "small", "medium", "large"])
        language = st.text_input("Enter Language Code (optional, e.g., 'en', 'fr')")

        if st.button("Transcribe"):
            with st.spinner("Transcribing... Please wait."):
                try:
                    transcription = robust_speech_recognition(uploaded_file.getbuffer(), model_size=model_size, language=language)
                    st.subheader("Transcription:")
                    st.write(transcription)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

elif option == "Record from Microphone":
    audio = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="🛑 Stop Recording")

    if audio:
        st.write("Audio recorded!")
        st.audio(audio["bytes"], format=audio["format"])  # Play the recorded audio

        model_size = st.selectbox("Select Model Size", ["tiny", "base", "small", "medium", "large"], key="record_model_size")
        language = st.text_input("Enter Language Code (optional, e.g., 'en', 'fr')", key="record_language")

        if st.button("Transcribe", key="record_transcribe"):
            with st.spinner("Transcribing... Please wait."):
                try:
                    transcription = robust_speech_recognition(audio["bytes"], model_size=model_size, language=language)
                    st.subheader("Transcription:")
                    st.write(transcription)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
