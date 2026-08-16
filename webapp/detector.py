import json
from collections import deque

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LABEL_COLORS = {"Red": (0, 0, 255), "Green": (0, 200, 0)}   # BGR
DEFAULT_COLOR = (0, 200, 255)


class ImageClassifier(nn.Module):
    """Same architecture as torchnn.py / Signal_Classifier_Training.ipynb - the
    state_dict loaded below must come from a model built with this exact
    definition, or load_state_dict will fail on a shape mismatch."""

    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 26 * 26, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class SignalDetector:
    """Classifies a whole camera frame as one of the trained signal classes -
    no bounding boxes, since this is pure classification rather than object
    detection. Debounces raw per-frame predictions into a stable class so a
    single noisy frame can't trigger an announcement."""

    def __init__(self, model_path, classes_path, conf_threshold=0.6, stable_frames=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(classes_path) as f:
            self.classes = json.load(f)

        self.model = ImageClassifier(len(self.classes)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.conf_threshold = conf_threshold
        self.stable_frames = stable_frames
        self._recent = deque(maxlen=stable_frames)
        self.last_prediction = None    # {"cls": str, "conf": float}
        self.stable_class = None

    def reset(self):
        self._recent.clear()
        self.last_prediction = None
        self.stable_class = None

    def process(self, frame_bgr):
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

        cls_name = self.classes[idx.item()]
        conf_value = conf.item()
        detected = cls_name if conf_value >= self.conf_threshold else None

        self.last_prediction = {"cls": cls_name, "conf": conf_value}
        self._recent.append(detected)
        if len(self._recent) == self.stable_frames and len(set(self._recent)) == 1:
            self.stable_class = self._recent[0]

        return self.stable_class


def draw_label(frame, prediction):
    if not prediction:
        return frame

    out = frame.copy()
    color = LABEL_COLORS.get(prediction["cls"], DEFAULT_COLOR)
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 6)

    label = f'{prediction["cls"]} {prediction["conf"]:.0%}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (10, 10), (10 + tw + 16, 10 + th + 16), color, -1)
    cv2.putText(out, label, (18, 10 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out
