import os
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

 
# SETTINGS // change these paths and parameters as needed
 
TRAIN_IMAGE_DIR = "C:\\Users\\Cavem\\ML_TrainingImages"  # folder of mostly upright training pages
MODEL_SAVE_PATH = "C:\\Users\\Cavem\\Detilt_Model\\deskew_model.pth" #where to save the trained model .pth file

IMAGE_SIZE = 224                         # input size expected by ResNet
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

MAX_ROTATION_DEG = 5.0                   # synthetic rotation range during training
TRAIN_SPLIT = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

 
# HELPERS
 
def list_image_files(folder):
    # Returns all valid image paths in a folder.
    folder = Path(folder)
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
            files.append(str(p))
    return sorted(files)


def rotate_pil_expand(img, angle_deg, fill=(255, 255, 255)):
    
    #Rotate PIL image with expanded canvas.
    #Positive angle = counterclockwise.
    
    return img.rotate(angle_deg, resample=Image.BICUBIC, expand=True, fillcolor=fill)


def rotate_cv_expand(image, angle_deg, border_value=(255, 255, 255)):
    
    #Rotate OpenCV image with expanded canvas.
    #Positive angle = counterclockwise.
    
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


 
# DATASET
 
class SyntheticDeskewDataset(Dataset):
    """
    Takes mostly upright document images and creates training examples by:
      1. loading image
      2. applying random synthetic rotation in [-MAX_ROTATION_DEG, +MAX_ROTATION_DEG]
      3. returning tensor + target angle

    Target angle = the rotation that was applied.
    To deskew later, rotate image by -predicted_angle.
    """
    def __init__(self, image_paths, image_size=224, max_rotation_deg=5.0):
        self.image_paths = image_paths
        self.image_size = image_size
        self.max_rotation_deg = max_rotation_deg

        self.to_tensor = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # use 3 channels for ResNet
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
        rotated = rotate_pil_expand(img, angle)

        x = self.to_tensor(rotated)
        y = torch.tensor([angle], dtype=torch.float32)

        return x, y

 
# MODEL
 
class DeskewAngleRegressor(nn.Module):
    """
    ResNet18 backbone with a single regression output:
      predicted angle in degrees
    """
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


# TRAIN / EVAL
 
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, angles in loader:
        images = images.to(device)
        angles = angles.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, angles)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):     # returns avg_loss, avg_mae (MAE = mean absolute error in degrees)
    model.eval()
    running_loss = 0.0
    mae_sum = 0.0

    for images, angles in loader:
        images = images.to(device)
        angles = angles.to(device)

        preds = model(images)
        loss = criterion(preds, angles)

        running_loss += loss.item() * images.size(0)
        mae_sum += torch.abs(preds - angles).sum().item()

    avg_loss = running_loss / len(loader.dataset)
    avg_mae = mae_sum / len(loader.dataset)
    return avg_loss, avg_mae


def train_model():
    image_paths = list_image_files(TRAIN_IMAGE_DIR)

    if not image_paths:
        print(f"No training images found in: {TRAIN_IMAGE_DIR}")
        return

    print(f"Found {len(image_paths)} training image(s).")

    dataset = SyntheticDeskewDataset(
        image_paths=image_paths,
        image_size=IMAGE_SIZE,
        max_rotation_deg=MAX_ROTATION_DEG
    )

    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        print("Not enough images to split into train/validation.")
        return

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = DeskewAngleRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss()  # Huber-like, usually stable for regression

    best_val_mae = float("inf")
    
    

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_mae = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mae={val_mae:.4f} deg"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "model_state": model.state_dict(),
                "best_mae": best_val_mae
            }, MODEL_SAVE_PATH)
            print(f"Saved best model to: {MODEL_SAVE_PATH}")

    print("Training complete.")
    print(f"Best validation MAE: {best_val_mae:.4f} degrees")


 
# INFERENCE / DESKEW
 
class InferenceTransform:
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


def load_trained_model(model_path, device):
    model = DeskewAngleRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def deskew_folder_with_model(
    model_path,
    input_dir,
    output_dir,
    max_abs_prediction=10.0,
):
    os.makedirs(output_dir, exist_ok=True)

    files = list_image_files(input_dir)
    if not files:
        print(f"No images found in: {input_dir}")
        return

    model = load_trained_model(model_path, DEVICE)
    transform = InferenceTransform(IMAGE_SIZE)

    for path in files:
        filename = os.path.basename(path)

        pil_img = Image.open(path).convert("RGB")
        predicted_angle = predict_angle(model, pil_img, transform, DEVICE)

        # Safety clamp
        if abs(predicted_angle) > max_abs_prediction:
            used_angle = 0.0
        else:
            used_angle = predicted_angle

        # OpenCV rotate for saving corrected full-size image
        image_cv = cv2.imread(path)
        corrected = rotate_cv_expand(image_cv, -used_angle)

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, corrected)

        print(
            f"{filename}: predicted angle = {predicted_angle:.3f} deg | "
            f"applied correction = {-used_angle:.3f} deg"
        )


 
# MAIN
 
if __name__ == "__main__":
    # train model
    train_model()
