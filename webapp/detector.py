from collections import deque

from ultralytics import YOLO

BOX_COLORS = {
    "Red Light": (0, 0, 255),     # BGR
    "Green Light": (0, 200, 0),
}
DEFAULT_COLOR = (0, 200, 255)


class SignalDetector:
    """Wraps the trained YOLOv8 model and debounces raw per-frame detections into
    a stable class, so a single noisy/flickered frame can't trigger an announcement."""

    def __init__(self, model_path, conf_threshold=0.5, stable_frames=3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.stable_frames = stable_frames
        self._recent = deque(maxlen=stable_frames)
        self.last_boxes = []       # [{"cls": str, "conf": float, "xyxy": (x1,y1,x2,y2)}]
        self.stable_class = None

    def reset(self):
        self._recent.clear()
        self.last_boxes = []
        self.stable_class = None

    def process(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]

        boxes = []
        detected = None
        best_conf = 0.0
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_name = results.names[int(box.cls[0])]
            xyxy = tuple(int(v) for v in box.xyxy[0])
            boxes.append({"cls": cls_name, "conf": conf, "xyxy": xyxy})
            if conf > best_conf:
                best_conf = conf
                detected = cls_name

        self.last_boxes = boxes
        self._recent.append(detected)
        if len(self._recent) == self.stable_frames and len(set(self._recent)) == 1:
            self.stable_class = self._recent[0]

        return self.stable_class


def draw_boxes(frame, boxes):
    import cv2

    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        color = BOX_COLORS.get(b["cls"], DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{b["cls"]} {b["conf"]:.2f}'
        cv2.putText(out, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
