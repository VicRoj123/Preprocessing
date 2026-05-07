import os
from pathlib import Path

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# SETTINGS

MODEL_PATH = "C:\\Users\\Cavem\\Detilt_Model\\deskew_model.pth"
INPUT_DIR = "C:\\Users\\Cavem\\Detilt_Model\\real_pages"
OUTPUT_DIR = "C:\\Users\\Cavem\\Detilt_Model\\prio_skew"

IMAGE_SIZE = 224
MAX_ABS_PREDICTION = 10.0   # ignore predictions beyond this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

 
# MODEL
 
class DeskewAngleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


# HELPERS
 
def list_image_files(folder):
     # Returns all valid image paths in a folder.
    folder = Path(folder)
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
            files.append(str(p))
    return sorted(files)


def rotate_cv_expand(image, angle_deg, border_value=(255, 255, 255)):
    """
    Rotate image while expanding canvas so nothing gets clipped.
    Positive angle = counterclockwise.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])

    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated


class InferenceTransform:       # same as training transform but without random rotation
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __call__(self, pil_img):
        return self.transform(pil_img)


@torch.no_grad()
def predict_angle(model, pil_img, transform, device):
    model.eval()
    x = transform(pil_img).unsqueeze(0).to(device)
    pred = model(x).squeeze().item()
    return float(pred)


def load_model(model_path, device):
    model = DeskewAngleRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def process_folder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = list_image_files(INPUT_DIR)
    if not image_paths:
        print("No images found in:", INPUT_DIR)
        return

    model = load_model(MODEL_PATH, DEVICE)
    transform = InferenceTransform(IMAGE_SIZE)

    print(f"Found {len(image_paths)} image(s).\n")

    for path in image_paths:
        filename = os.path.basename(path)

        # PIL image for prediction
        pil_img = Image.open(path).convert("RGB")
        predicted_angle = predict_angle(model, pil_img, transform, DEVICE)

        # safety clamp
        if abs(predicted_angle) > MAX_ABS_PREDICTION:
            used_angle = 0.0
        else:
            used_angle = predicted_angle

        # OpenCV image for full-resolution rotation/saving
        image_cv = cv2.imread(path)
        if image_cv is None:
            print(f"Could not read: {filename}")
            continue

        corrected = rotate_cv_expand(image_cv, -used_angle)

        name_only = os.path.splitext(filename)[0]
        new_filename = f"{name_only}_deskewed.tif"
        out_path = os.path.join(OUTPUT_DIR, new_filename)
        
        success = cv2.imwrite(out_path, corrected)

        if not success:
            print(f"Failed to save: {filename}")
            continue

        print(
            f"{filename}: predicted angle = {predicted_angle:.3f}° | "
            f"applied correction = {-used_angle:.3f}°"
        )


if __name__ == "__main__":
    process_folder()