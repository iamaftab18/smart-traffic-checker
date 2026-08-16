import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- ESP32-CAM ---
# Printed to Serial by the firmware on boot. Override with an env var if you
# don't want to edit this file: set ESP32_STREAM_URL before running app.py.
ESP32_STREAM_URL = os.environ.get("ESP32_STREAM_URL", "http://192.168.1.50/stream")

# --- Detection model ---
# Drop the best.pt downloaded from the training notebook here.
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
CONF_THRESHOLD = 0.5
STABLE_FRAMES = 3          # consecutive frames that must agree before a detection is trusted
REPEAT_INTERVAL_SEC = 10   # re-announce the current state if it hasn't changed in this long

# --- Class -> spoken phrase ---
# This watches the VEHICLE traffic light, not a pedestrian walk signal:
# vehicles stopped on red -> safe to cross. Vehicles moving on green -> not safe.
CLASS_PHRASES = {
    "Red Light": "You can cross the Road",
    "Green Light": "Stop! Don't cross the road",
}

# --- Streaming ---
JPEG_QUALITY = 80
TARGET_FPS = 25
