Multimodal Facial & Spoken Emotion Recognition System with Incongruity Detection
Overview
This project implements a real-time Multimodal Emotion Recognition Engine developed for CSE427 (Machine Learning & Computer Vision). The system processes visual facial expressions alongside transcribed speech in real time to yield a unified emotional verdict.

A key highlight of this architecture is its Incongruity & Contradiction Engine, which detects discrepancies between what a user says and their facial expression (e.g., detecting sarcasm, masked frustration, or suppressed sadness).

Architecture & System Pipeline
                              ┌───────────────────────────┐
                              │    Live Webcam Stream     │
                              └─────────────┬─────────────┘
                                            │
                                            ▼
                              ┌───────────────────────────┐
                              │ Haar Cascade Face Detect  │
                              └─────────────┬─────────────┘
                                            │
                                            ▼
                              ┌───────────────────────────┐
                              │ ResNet-18 Vision Model    │
                              │ (Temporal 10-Frame EMA)   │
                              └─────────────┬─────────────┘
                                            │
  ┌───────────────────────────┐             │
  │   Live Microphone Input   │             │
  └─────────────┬─────────────┘             │
                │                           │
                ▼                           │
  ┌───────────────────────────┐             │
  │  Whisper Tiny (STT Chunk) │             │
  └─────────────┬─────────────┘             │
                │                           │
                ▼                           │
  ┌───────────────────────────┐             │
  │ HuggingFace Text Pipeline │             │
  └─────────────┬─────────────┘             │
                │                           │
                └─────────────┬─────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │ Contradiction Fusion Engine │
               │   (Cosine Discrepancy Matrix)│
               └──────────────┬──────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │    Real-Time OpenCV UI      │
               └─────────────────────────────┘
Technical Stack & Frameworks
Language: Python 3.12 / 3.13

Computer Vision: OpenCV (cv2), PyTorch, Torchvision

Natural Language Processing & Speech: Hugging Face transformers (michelleli99/emotion_text_classifier), faster-whisper

Audio Capture: sounddevice, scipy

Hardware Acceleration: CUDA FP16 Inference (Optimized for NVIDIA GeForce GTX/RTX GPUs)

Technical Highlights & Solutions
1. High-Accuracy Facial Classifier
Backbone: ResNet-18 fine-tuned on ~50,000 standardized facial images across 7 core classes (angry, disgust, fear, happy, neutral, sad, surprise).

Validation Accuracy: Achieved 79.33% validation accuracy on test splits.

Temporal Prediction Smoothing: Applied an Exponential Moving Average (EMA) buffer over a 10-frame sliding window (deque(maxlen=10)) to eliminate rapid frame-to-frame prediction flickering on camera.

2. Multi-Threaded Audio Pipeline
Asynchronous Buffer: Speech is sampled continuously in non-blocking 3.0-second chunks using Python's threading and queue.Queue.

Robust Signal Normalization: Included dynamic array clipping (np.clip and np.nan_to_num) to protect against microphone clipping, input spikes, and memory overflows.

3. Shared Vector Mapping & Fusion Logic
Canonical Mapping: Normalizes output distributions from both modalities into 6 shared canonical classes:

Shared Vector=[Anger,Disgust/Fear,Joy/Happiness,Sadness,Surprise,Neutral]
Incongruity Index: Computes the angular discrepancy between normalized vision and text probability vectors:

Discrepancy=1.0− 
∥ 
V

 ∥∥ 
T

 ∥
V

 ⋅ 
T

 
​
 
Sarcasm/Contradiction Override: When valence opposition is detected (e.g., Happy face + Sad/Angry text), the system overrides standard weighted averaging and raises a Sarcasm / Masked Emotion Alert.

Project Structure
Plaintext
CSE427 Project/
│
├── best_emotion_resnet18.pth         # Saved weights for 79.33% ResNet-18 model
├── haarcascade_frontalface_default.xml # OpenCV Face Detector
├── fusion_engine.py                   # Vector alignment & contradiction logic module
├── main_app.py                       # Master script (Camera, Audio Threads, UI Dashboard)
├── train_vision.py                   # PyTorch vision model training pipeline
└── README.md                         # Project documentation
Setup & Running Guide
1. Environment Installation
Ensure PyTorch with CUDA support and the required dependencies are installed:

PowerShell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install faster-whisper transformers sounddevice scipy opencv-python numpy
2. File Verification
Ensure best_emotion_resnet18.pth and fusion_engine.py are present in the working directory.

3. Launching System
Run the main application script:

PowerShell
python main_app.py
Press q inside the webcam window to exit gracefully.

Live Demonstration Guide
Scenario	Facial Action	Spoken Phrase	Expected Final Verdict	Alert Badge
Normal Agreement	Smile clearly at camera	"I am so happy with this project"	Aligned: Joy/Happiness	GREEN (Normal)
Sarcasm / Contradiction	Smile widely at camera	"This is completely terrible and ruined my day"	Sarcasm / Masked Sadness	RED (Contradiction)
Neutral Conversational	Neutral expression	"The weather today is cloudy"	Neutral (Mild)	GREEN (Normal)