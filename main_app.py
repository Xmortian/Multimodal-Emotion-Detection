import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import pipeline
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import urllib.request
import threading
import queue
import time
import os
from collections import deque
from fusion_engine import EmotionFusionEngine

# ---------------------------------------------------------
# 1. CONFIGURATION & DEVICE SETUP
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PTH = r"C:\Users\USER\Desktop\CSE427 Project\best_emotion_resnet18.pth"
TEMP_WAV = "temp_live_audio.wav"
SAMPLE_RATE = 16000
AUDIO_DURATION = 4.0  # 3-second audio buffer

# Rolling window buffer to smooth facial prediction flickering (10 frames)
SMOOTHING_WINDOW_SIZE = 10
vision_prob_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)

print(f"[INFO] Running compute device: {DEVICE}")

# ---------------------------------------------------------
# 2. LOAD TRAINED FACIAL RESNET-18 MODEL & FACE DETECTOR
# ---------------------------------------------------------
print("[INFO] Loading trained ResNet-18 facial emotion model...")
if not os.path.exists(MODEL_PTH):
    MODEL_PTH = "best_emotion_resnet18.pth"

checkpoint = torch.load(MODEL_PTH, map_location=DEVICE)
VISION_CLASSES = checkpoint.get('classes', ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

vision_model = models.resnet18(weights=None)
num_ftrs = vision_model.fc.in_features
vision_model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, len(VISION_CLASSES))
)
vision_model.load_state_dict(checkpoint['model_state_dict'])
vision_model.to(DEVICE)
vision_model.eval()

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

HAAR_FILE = "haarcascade_frontalface_default.xml"
if not os.path.exists(HAAR_FILE):
    print("[INFO] Downloading face detection cascade XML locally...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    try:
        urllib.request.urlretrieve(url, HAAR_FILE)
    except Exception as e:
        print(f"[WARNING] Could not download Haar cascade: {e}")

face_cascade = cv2.CascadeClassifier(HAAR_FILE)
if face_cascade.empty():
    sys_haar = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(sys_haar)

# ---------------------------------------------------------
# 3. LOAD TEXT EMOTION CLASSIFIER & WHISPER
# ---------------------------------------------------------
print("[INFO] Loading Hugging Face text emotion pipeline...")
text_classifier = pipeline(
    "text-classification",
    model="michelleli99/emotion_text_classifier",
    top_k=None,
    device=0 if DEVICE == "cuda" else -1
)

print("[INFO] Loading Whisper Tiny for speech transcription...")
whisper_model = WhisperModel("tiny.en", device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")

fusion_engine = EmotionFusionEngine(incongruity_threshold=0.45)

# ---------------------------------------------------------
# 4. MULTITHREADED AUDIO WORKER
# ---------------------------------------------------------
audio_queue = queue.Queue()
latest_text_result = {"text": "Listening...", "raw_preds": [{'label': 'neutral', 'score': 0.99}]}

def audio_worker():
    global latest_text_result
    while True:
        try:
            audio_data = audio_queue.get()
            if audio_data is None:
                break
            
            audio_data = np.nan_to_num(audio_data)
            audio_clipped = np.clip(audio_data, -1.0, 1.0)
            audio_pcm = (audio_clipped * 32767).astype(np.int16)
            
            wav.write(TEMP_WAV, SAMPLE_RATE, audio_pcm)
            
            segments, _ = whisper_model.transcribe(TEMP_WAV, beam_size=1)
            text = " ".join([seg.text for seg in segments]).strip()
            
            if text:
                preds = text_classifier(text)[0]
                latest_text_result = {"text": text, "raw_preds": preds}
            
            audio_queue.task_done()
        except Exception as e:
            print(f"[AUDIO WORKER WARNING] {e}")

threading.Thread(target=audio_worker, daemon=True).start()

# ---------------------------------------------------------
# 5. UI DASHBOARD RENDERING
# ---------------------------------------------------------
def draw_ui_dashboard(frame, vision_raw, text_info, fusion_res):
    h, w, _ = frame.shape
    dashboard_w = 440
    canvas = np.zeros((h, w + dashboard_w, 3), dtype=np.uint8)
    
    canvas[0:h, 0:w] = frame
    
    panel_x = w
    cv2.rectangle(canvas, (panel_x, 0), (panel_x + dashboard_w, h), (20, 20, 20), -1)
    
    cv2.putText(canvas, "MULTIMODAL EMOTION UI", (panel_x + 15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.line(canvas, (panel_x + 15, 45), (panel_x + dashboard_w - 15, 45), (100, 100, 100), 1)

    # 1. Vision Verdict
    cv2.putText(canvas, "FACIAL EMOTION (ResNet-18)", (panel_x + 15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1)
    cv2.putText(canvas, f"Verdict: {fusion_res['vision_top']} ({fusion_res['vision_conf']*100:.1f}%)", 
                (panel_x + 15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 2. Spoken Text Verdict
    cv2.putText(canvas, "SPOKEN SPEECH (Whisper + HF)", (panel_x + 15, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1)
    
    disp_text = text_info['text']
    if len(disp_text) > 36:
        disp_text = disp_text[:33] + "..."
        
    cv2.putText(canvas, f"Text: \"{disp_text}\"", (panel_x + 15, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(canvas, f"Verdict: {fusion_res['text_top']} ({fusion_res['text_conf']*100:.1f}%)", 
                (panel_x + 15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 3. Discrepancy Meter
    cv2.line(canvas, (panel_x + 15, 215), (panel_x + dashboard_w - 15, 215), (60, 60, 60), 1)
    cv2.putText(canvas, f"Incongruity Index: {fusion_res['discrepancy_score']:.2f}", 
                (panel_x + 15, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    bar_w = int(fusion_res['discrepancy_score'] * (dashboard_w - 30))
    bar_color = (0, 0, 255) if fusion_res['is_contradiction'] else (0, 255, 0)
    cv2.rectangle(canvas, (panel_x + 15, 250), (panel_x + dashboard_w - 15, 262), (50, 50, 50), -1)
    cv2.rectangle(canvas, (panel_x + 15, 250), (panel_x + 15 + bar_w, 262), bar_color, -1)

    # 4. Final Fused Verdict Box
    cv2.putText(canvas, "FINAL FUSED VERDICT", (panel_x + 15, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
    
    box_color = (0, 0, 180) if fusion_res['badge_color'] == 'RED' else (0, 140, 0)
    cv2.rectangle(canvas, (panel_x + 15, 310), (panel_x + dashboard_w - 15, 370), box_color, -1)
    cv2.putText(canvas, fusion_res['final_verdict'], (panel_x + 25, 345),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Status Badge
    badge_str = "ALERT: CONTRADICTION DETECTED" if fusion_res['is_contradiction'] else "STATUS: ALIGNED / NORMAL"
    cv2.putText(canvas, badge_str, (panel_x + 15, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if fusion_res['is_contradiction'] else (0, 255, 0), 1)

    return canvas

# ---------------------------------------------------------
# 6. MAIN LIVE LOOP
# ---------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam!")
        return

    print("\n[INFO] Starting live system... Press 'q' in the window to exit.")
    last_audio_record_time = time.time()
    raw_probs = np.ones(len(VISION_CLASSES), dtype=np.float32) / len(VISION_CLASSES)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            (x, y, bw, bh) = faces[0]
            
            # Extract face crop
            face_crop = frame[y:y+bh, x:x+bw]
            try:
                face_tensor = img_transform(face_crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = vision_model(face_tensor)
                    raw_probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            except Exception:
                pass
            
            # Push raw frame probabilities to history buffer for Temporal Smoothing
            vision_prob_history.append(raw_probs)
            
            # Compute average probability vector across buffer
            smoothed_probs = np.mean(vision_prob_history, axis=0)
            vision_raw = {cls: float(smoothed_probs[i]) for i, cls in enumerate(VISION_CLASSES)}

            # Run fusion engine to get final verdict for bounding box display
            fusion_res = fusion_engine.analyze(vision_raw, latest_text_result['raw_preds'])

            # Draw green box around face
            box_color = (0, 0, 255) if fusion_res['is_contradiction'] else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)

            # --- DRAW FINAL VERDICT TEXT BESIDE / ABOVE THE GREEN FACE BOX ---
            verdict_text = f"Verdict: {fusion_res['final_verdict']}"
            
            # Text background rectangle above bounding box
            text_size = cv2.getTextSize(verdict_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            text_y = max(y - 10, 25)
            cv2.rectangle(frame, (x, text_y - text_size[1] - 8), (x + text_size[0] + 12, text_y + 4), (0, 0, 0), -1)
            cv2.putText(frame, verdict_text, (x + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        else:
            # Fallback if no face detected in frame
            smoothed_probs = np.mean(vision_prob_history, axis=0) if len(vision_prob_history) > 0 else raw_probs
            vision_raw = {cls: float(smoothed_probs[i]) for i, cls in enumerate(VISION_CLASSES)}
            fusion_res = fusion_engine.analyze(vision_raw, latest_text_result['raw_preds'])

        # Non-blocking audio capture
        if time.time() - last_audio_record_time >= AUDIO_DURATION:
            last_audio_record_time = time.time()
            
            def record_chunk():
                try:
                    audio_data = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE), 
                                        samplerate=SAMPLE_RATE, channels=1, dtype='float32')
                    sd.wait()
                    audio_queue.put(audio_data)
                except Exception as e:
                    print(f"[MIC RECORD WARNING] {e}")

            threading.Thread(target=record_chunk, daemon=True).start()

        dashboard = draw_ui_dashboard(frame, vision_raw, latest_text_result, fusion_res)

        cv2.imshow("CSE427 Multimodal Emotion Recognition System", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(TEMP_WAV):
        os.remove(TEMP_WAV)

if __name__ == "__main__":
    main()