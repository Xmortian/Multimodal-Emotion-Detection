import torch
from transformers import pipeline
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

def test_speech_and_text_emotion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running text and audio models on: {device}")

    # 1. Load Text Emotion Classifier (GoEmotions fine-tuned model)
    print("[INFO] Loading Text Emotion Classifier from Hugging Face...")
    text_classifier = pipeline(
        "text-classification",
        model="michelleli99/emotion_text_classifier",
        top_k=None,
        device=0 if device == "cuda" else -1
    )

    # 2. Load Faster-Whisper Model for Speech Recognition
    print("[INFO] Loading Whisper Tiny for real-time speech transcription...")
    whisper_model = WhisperModel("tiny.en", device=device, compute_type="float16" if device == "cuda" else "int8")

    # 3. Quick Text Emotion Inference Test
    sample_texts = [
        "I absolutely love working on machine learning projects!",
        "This is completely terrible and ruining my entire day.",
        "Wait, what just happened? I did not expect that."
    ]

    print("\n--- TEXT EMOTION CLASSIFICATION TEST ---")
    for text in sample_texts:
        predictions = text_classifier(text)[0]
        top_pred = max(predictions, key=lambda x: x['score'])
        print(f"Text: '{text}'")
        print(f"  --> Detected Emotion: {top_pred['label']} ({top_pred['score']:.2f})\n")

    # 4. Live Microphone Audio Test (Recording 4 seconds)
    duration = 4  # seconds
    sample_rate = 16000
    
    print("\n--- LIVE MICROPHONE TEST ---")
    print(f"[ACTION] Speak into your mic now! Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("[INFO] Recording complete. Transcribing audio...")

    # Save temporary audio file for Whisper
    temp_wav = "temp_test_audio.wav"
    wav.write(temp_wav, sample_rate, (audio_data * 32767).astype(np.int16))

    # Transcribe speech to text
    segments, _ = whisper_model.transcribe(temp_wav, beam_size=1)
    transcribed_text = " ".join([segment.text for segment in segments]).strip()

    print(f"\n[SPEECH RESULT] Transcribed Text: '{transcribed_text}'")

    if transcribed_text:
        text_preds = text_classifier(transcribed_text)[0]
        top_text_emotion = max(text_preds, key=lambda x: x['score'])
        print(f"[EMOTION RESULT] Text Emotion: {top_text_emotion['label'].upper()} ({top_text_emotion['score']:.2f})")
    else:
        print("[WARNING] No speech detected in recording.")

if __name__ == "__main__":
    test_speech_and_text_emotion()