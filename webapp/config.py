import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- ESP32-CAM ---
# Printed to Serial by the firmware on boot. Override with an env var if you
# don't want to edit this file: set ESP32_STREAM_URL before running app.py.
ESP32_STREAM_URL = os.environ.get("ESP32_STREAM_URL", "http://192.168.1.50/stream")

# --- Classification model ---
# Whole-frame classifier (no object detection/localization) - drop the files
# downloaded from Signal_Classifier_Training.ipynb here.
MODEL_PATH = os.path.join(BASE_DIR, "models", "signal_classifier.pt")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.json")
# A 2-class softmax's top score is mathematically always >= 0.5 (the two
# probabilities sum to 1), so 0.5 would filter nothing - this needs to be
# meaningfully above that to actually reject "barely leaning one way" frames.
CONF_THRESHOLD = 0.6
STABLE_FRAMES = 3          # consecutive frames that must agree before a prediction is trusted
REPEAT_INTERVAL_SEC = 10   # re-announce the current state if it hasn't changed in this long

# --- Class -> spoken phrase ---
# Class names come from the training data folder names (data/Green, data/Red).
# This watches the VEHICLE traffic light, not a pedestrian walk signal:
# vehicles stopped on red -> safe to cross. Vehicles moving on green -> not safe.
CLASS_PHRASES = {
    "Red": "You can cross the Road",
    "Green": "Stop! Don't cross the road",
}

# --- Streaming ---
JPEG_QUALITY = 80
TARGET_FPS = 25
