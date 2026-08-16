import json
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

# Paths
DATA_DIR = Path("data")                                   # expects data/<class_name>/*.jpg
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "silkworm_disease_detection.pt"
CLASSES_PATH = MODEL_DIR / "classes.json"

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])


# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            # Input: 3 channels (RGB), 224x224
            nn.Conv2d(3, 32, (3, 3)),   # Output: 32x222x222
            nn.ReLU(),
            nn.MaxPool2d(2),            # Output: 32x111x111
            nn.Conv2d(32, 64, (3, 3)),  # Output: 64x109x109
            nn.ReLU(),
            nn.MaxPool2d(2),            # Output: 64x54x54
            nn.Conv2d(64, 128, (3, 3)), # Output: 128x52x52
            nn.ReLU(),
            nn.MaxPool2d(2),            # Output: 128x26x26
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 26 * 26, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(device):
    """Load a trained model + its class list from disk. Returns (clf, classes)."""
    with open(CLASSES_PATH) as f:
        classes = json.load(f)
    clf = ImageClassifier(len(classes)).to(device)
    # map_location lets a model trained on a Colab GPU load on a CPU-only Raspberry Pi
    clf.load_state_dict(load(MODEL_PATH, map_location=device))
    clf.eval()
    return clf, classes


def predict_image(image_path, clf, classes, device):
    clf.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = clf(img_tensor)
        prediction = torch.argmax(output).item()

    return classes[prediction]


def train(epochs=10, batch_size=32, lr=1e-3, val_split=0.2):
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    classes = full_dataset.classes
    print(f"Found classes: {classes} ({len(full_dataset)} images)")

    val_size = max(1, int(val_split * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = get_device()
    print(f"Training on: {device}")

    clf = ImageClassifier(len(classes)).to(device)
    opt = Adam(clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        clf.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            yhat = clf(X)
            loss = loss_fn(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        clf.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                yhat = clf(X)
                correct += (torch.argmax(yhat, dim=1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0.0

        print(f"Epoch {epoch + 1}/{epochs} - loss: {running_loss / len(train_loader):.4f} - val_acc: {val_acc:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    save(clf.state_dict(), MODEL_PATH)
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Classes saved to {CLASSES_PATH}")

    return clf, classes, device


if __name__ == "__main__":
    device = get_device()

    if MODEL_PATH.exists() and CLASSES_PATH.exists():
        clf, classes = load_model(device)
        print(f"Loaded existing model. Classes: {classes}")

        example_image = Path("example.jpg")
        if example_image.exists():
            prediction = predict_image(example_image, clf, classes, device)
            print(f"Predicted class: {prediction}")
        else:
            print("Model loaded. Call predict_image(path, clf, classes, device) to run inference.")
    else:
        print("No saved model found. Starting training...")
        train()
