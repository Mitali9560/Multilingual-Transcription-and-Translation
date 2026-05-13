
🌍 Multilingual Text & Audio Application
This application supports multilingual text and audio processing using modern AI models.

✨ Features
📝 Text Input
Accepts text in any language
Automatically detects input language
Translates to:
English (en)
Hindi (hi)
Telugu (te)
Tamil (ta)
Output as text or speech
🎤 Audio Input
Accepts audio in any language
Audio preprocessing:
Stereo → Mono
Resampled to 16 kHz
Uses Whisper to convert speech → English
Translates English to selected language
Output as text or speech
🧠 Architecture Overview
Text Pipeline
Text → Language Auto-detection → Translation → Text / TTS

Audio Pipeline
Audio → Preprocessing → Whisper (translate) → English → Translation → Text / TTS

🚀 Deployment
This app is deployed using Gradio on Hugging Face Spaces.

No backend server required
CPU-only
Public sharable link
🛠️ Tech Stack
Gradio
Faster-Whisper
GoogleTranslator (deep-translator)
gTTS
Librosa
📌 Future Improvements
Fine-tuned ASR model
Fine-tuned TTS model
Additional languages
RAG-based document Q&A
