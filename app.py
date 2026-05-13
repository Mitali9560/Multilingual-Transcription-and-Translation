import uuid
import gradio as gr
import librosa
import soundfile as sf

from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from gtts import gTTS

# -----------------------------
# Load Whisper model (CPU)
# -----------------------------
whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# -----------------------------
# Audio Preprocessing
# -----------------------------
def preprocess_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    out_path = "clean_audio.wav"
    sf.write(out_path, audio, 16000)
    return out_path

# -----------------------------
# Speech → English (Whisper)
# -----------------------------
def speech_to_text(audio_path):
    segments, _ = whisper_model.transcribe(
        audio_path,
        task="translate"  # ANY language → English
    )
    return " ".join(seg.text for seg in segments)

# -----------------------------
# Text Translation (Auto-detect)
# -----------------------------
def translate_text(text, target_lang):
    return GoogleTranslator(
        source="auto",
        target=target_lang
    ).translate(text)

# -----------------------------
# Text → Speech
# -----------------------------
def text_to_speech(text, lang):
    filename = f"{uuid.uuid4()}.mp3"
    gTTS(text=text, lang=lang).save(filename)
    return filename

# -----------------------------
# Unified Processing Function
# -----------------------------
def process_input(
    input_mode,
    text_input,
    audio_input,
    output_language,
    output_mode
):
    try:
        if input_mode == "Text":
            if not text_input or not text_input.strip():
                return "Please enter text.", None
            clean_text = text_input.strip()

        else:  # Audio
            if audio_input is None:
                return "Please upload audio.", None
            clean_audio = preprocess_audio(audio_input)
            clean_text = speech_to_text(clean_audio)

        translated_text = translate_text(clean_text, output_language)

        if output_mode == "Text":
            return translated_text, None

        audio_file = text_to_speech(translated_text, output_language)
        return translated_text, audio_file

    except Exception as e:
        return f"ERROR: {str(e)}", None

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Multilingual Text & Audio App") as app:
    gr.Markdown("## 🌍 Multilingual Text & Audio Processing")

    with gr.Tabs():
        with gr.Tab("Text Input"):
            text_input = gr.Textbox(label="Enter text (any language)")
            out_lang_text = gr.Dropdown(
                ["en", "hi", "te", "ta"],
                value="en",
                label="Output Language"
            )
            out_mode_text = gr.Radio(
                ["Text", "Audio"],
                value="Text",
                label="Output Mode"
            )
            submit_text = gr.Button("Submit Text")

        with gr.Tab("Audio Input"):
            audio_input = gr.Audio(
                type="filepath",
                label="Upload audio (any language)"
            )
            out_lang_audio = gr.Dropdown(
                ["en", "hi", "te", "ta"],
                value="en",
                label="Output Language"
            )
            out_mode_audio = gr.Radio(
                ["Text", "Audio"],
                value="Text",
                label="Output Mode"
            )
            submit_audio = gr.Button("Submit Audio")

    output_text = gr.Textbox(label="Output Text", lines=5)
    output_audio = gr.Audio(label="Output Audio")

    submit_text.click(
        fn=lambda t, l, m: process_input(
            "Text", t, None, l, m
        ),
        inputs=[text_input, out_lang_text, out_mode_text],
        outputs=[output_text, output_audio]
    )

    submit_audio.click(
        fn=lambda a, l, m: process_input(
            "Audio", None, a, l, m
        ),
        inputs=[audio_input, out_lang_audio, out_mode_audio],
        outputs=[output_text, output_audio]
    )

app.launch()
