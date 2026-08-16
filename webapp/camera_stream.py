import logging
import threading
import time

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


def build_stream_url(raw, default=None):
    """Turns user-entered text (bare IP, host:port, or full URL) into a stream
    URL the ESP32 firmware actually serves. Falls back to `default` when raw
    is empty, so a blank frontend field doesn't have to mean a broken request."""
    raw = (raw or "").strip()
    if not raw:
        return default
    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "http://" + raw
    if not raw.rstrip("/").endswith("/stream"):
        raw = raw.rstrip("/") + "/stream"
    return raw


class ESP32CamStream:
    """Pulls an MJPEG stream from an ESP32-CAM over HTTP and exposes the latest
    decoded frame. Runs in its own thread so a slow/dropped connection never
    blocks the Flask request threads; auto-reconnects on failure."""

    def __init__(self, url, reconnect_delay=2.0):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.connected = False

    def set_url(self, url):
        self.url = url

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.connected = False

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _run(self):
        while self._running:
            try:
                self._read_stream()
            except Exception:
                logger.exception("ESP32-CAM stream error, retrying in %.1fs", self.reconnect_delay)
            self.connected = False
            if self._running:
                time.sleep(self.reconnect_delay)

    def _read_stream(self):
        resp = requests.get(self.url, stream=True, timeout=5)
        resp.raise_for_status()
        self.connected = True
        buf = b""
        for chunk in resp.iter_content(chunk_size=2048):
            if not self._running:
                break
            if not chunk:
                continue
            buf += chunk
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = buf[start:end + 2]
                buf = buf[end + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
            elif len(buf) > 300_000:
                # Stream desynced without a valid frame boundary showing up - drop it
                # instead of letting the buffer grow without bound.
                buf = b""
