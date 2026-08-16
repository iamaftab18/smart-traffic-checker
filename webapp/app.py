import logging
import os
import threading
import time
from urllib.parse import urlparse

import cv2
from flask import Flask, Response, jsonify, render_template, request

import config
from camera_stream import ESP32CamStream, build_stream_url
from detector import SignalDetector, draw_boxes
from tts_engine import TTSWorker, list_output_devices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists(config.MODEL_PATH):
    raise SystemExit(
        f"Model not found at {config.MODEL_PATH}\n"
        "Download 'best.pt' from the training notebook and place it in webapp/models/."
    )

app = Flask(__name__)

cam_stream = ESP32CamStream(config.ESP32_STREAM_URL)
detector = SignalDetector(config.MODEL_PATH, config.CONF_THRESHOLD, config.STABLE_FRAMES)
tts = TTSWorker()

state_lock = threading.Lock()
state = {
    "running": False,
    "connected": False,
    "stream_url": None,
    "current_class": None,
    "last_spoken_class": None,
    "last_spoken_text": None,
    "last_spoken_time": 0.0,
}

worker_thread = None
stop_event = threading.Event()


def detection_loop():
    while not stop_event.is_set():
        frame = cam_stream.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        stable_class = detector.process(frame)

        with state_lock:
            state["current_class"] = stable_class
            state["connected"] = cam_stream.connected

        if stable_class in config.CLASS_PHRASES:
            now = time.time()
            with state_lock:
                changed = stable_class != state["last_spoken_class"]
                stale = (now - state["last_spoken_time"]) >= config.REPEAT_INTERVAL_SEC
            if changed or stale:
                phrase = config.CLASS_PHRASES[stable_class]
                tts.speak(phrase)
                with state_lock:
                    state["last_spoken_class"] = stable_class
                    state["last_spoken_text"] = phrase
                    state["last_spoken_time"] = now

        time.sleep(0.01)


@app.route("/")
def index():
    default_host = urlparse(config.ESP32_STREAM_URL).netloc or config.ESP32_STREAM_URL
    return render_template("index.html", default_esp32_host=default_host)


@app.route("/api/audio_devices")
def audio_devices():
    return jsonify(list_output_devices())


@app.route("/api/start", methods=["POST"])
def start_system():
    global worker_thread
    data = request.get_json(force=True, silent=True) or {}
    raw_index = data.get("device_index")
    try:
        device_index = int(raw_index) if raw_index is not None else None
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid device_index"}), 400

    stream_url = build_stream_url(data.get("esp32_ip"), default=config.ESP32_STREAM_URL)
    if not stream_url:
        return jsonify({"ok": False, "error": "ESP32-CAM IP address is required"}), 400

    with state_lock:
        if state["running"]:
            return jsonify({"ok": True, "message": "Already running"})
        state["running"] = True
        state["stream_url"] = stream_url
        state["current_class"] = None
        state["last_spoken_class"] = None
        state["last_spoken_text"] = None
        state["last_spoken_time"] = 0.0

    tts.set_device(device_index)
    detector.reset()
    cam_stream.set_url(stream_url)
    cam_stream.start()

    stop_event.clear()
    worker_thread = threading.Thread(target=detection_loop, daemon=True)
    worker_thread.start()

    return jsonify({"ok": True, "stream_url": stream_url})


@app.route("/api/stop", methods=["POST"])
def stop_system():
    stop_event.set()
    cam_stream.stop()
    with state_lock:
        state["running"] = False
        state["connected"] = False
        state["stream_url"] = None
    return jsonify({"ok": True})


@app.route("/api/status")
def status():
    with state_lock:
        return jsonify(dict(state))


def mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = cam_stream.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        annotated = draw_boxes(frame, detector.last_boxes)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if not ok:
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(1 / config.TARGET_FPS)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
