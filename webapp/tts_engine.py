import logging
import os
import queue
import tempfile
import threading

import pyttsx3
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

SPEAK_TIMEOUT_SEC = 15


def list_output_devices():
    """Returns [{'index': int, 'name': str}] for every playback-capable device
    the OS currently exposes - including any connected Bluetooth speaker/headset,
    which shows up here once it's paired and connected in Windows Bluetooth settings."""
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_output_channels", 0) > 0:
            devices.append({"index": idx, "name": dev["name"]})
    return devices


class TTSWorker:
    """Synthesizes speech offline (pyttsx3, no internet dependency) and plays it
    through a chosen output device (sounddevice). All engine calls happen on one
    dedicated thread processing a queue - pyttsx3 isn't safe to drive from
    multiple threads at once, so this keeps announcements off the detection loop
    without risking that."""

    def __init__(self):
        self._queue = queue.Queue()
        self._device_index = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_device(self, device_index):
        self._device_index = device_index

    def speak(self, text):
        self._queue.put(text)

    def _run(self):
        while True:
            text = self._queue.get()
            self._speak_with_timeout(text)

    def _speak_with_timeout(self, text):
        # A stuck pyttsx3/SAPI5 call would otherwise block this loop forever,
        # silently killing every announcement after it. Running it on a
        # throwaway thread means a hang just gets abandoned instead of wedging
        # the whole announcer.
        done = threading.Event()

        def worker():
            try:
                self._speak_now(text)
            except Exception:
                logger.exception("TTS playback failed")
            finally:
                done.set()

        threading.Thread(target=worker, daemon=True).start()
        if not done.wait(SPEAK_TIMEOUT_SEC):
            logger.error(
                "TTS call for %r did not finish within %ss - abandoning it so later announcements aren't blocked",
                text, SPEAK_TIMEOUT_SEC,
            )

    def _speak_now(self, text):
        # pyttsx3's SAPI5 driver only reliably renders once per engine instance -
        # reusing one across calls makes every announcement after the first go
        # silent with no error. A fresh engine per utterance avoids that.
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            engine.save_to_file(text, wav_path)
            engine.runAndWait()
            engine.stop()
            data, samplerate = sf.read(wav_path, dtype="float32")
            sd.play(data, samplerate, device=self._device_index)
            sd.wait()
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
