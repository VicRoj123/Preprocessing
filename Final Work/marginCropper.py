import cv2 as cv
import os

"""
Margin Cropper

Removes artificial borders introduced by deskewing.
Designed to integrate into OCR preprocessing pipeline.

Features:
- Crops a percentage from all sides of the image
- Supports single image or batch processing
- Outputs to Detilt_Model/cropped/
"""

# =========================
# CONFIG
# =========================

# Input options
INPUT_MODE = "batch"   # "single" or "batch"

INPUT_PATH = "C:\\Users\\Cavem\\Detilt_Model\\deskewed_pages\\00721_deskewed.tif"
INPUT_FOLDER = "C:\\Users\\Cavem\\Detilt_Model\\prio_skew"

# Crop percentage (1% default)
CROP_PERCENT = 0.01

# Root directory (ensures consistent output location)
DETILT_MODEL_DIR = "C:\\Users\\Cavem\\Detilt_Model"


# =========================
# CORE FUNCTION
# =========================

def crop_margin_percent(image, percent=0.01):
    """
    Crops a percentage from all four sides of an image.

    Args:
        image: Input image (numpy array)
        percent: Percentage to crop from each side (0.01 = 1%)

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]

    dx = int(w * percent)
    dy = int(h * percent)

    # Prevent over-cropping edge cases
    if dx <= 0 or dy <= 0:
        return image

    cropped = image[dy:h - dy, dx:w - dx]
    return cropped


# =========================
# SAVE FUNCTION
# =========================

def save_cropped_image(input_path, cropped_img):
    """
    Saves cropped image into Detilt_Model/cropped/
    """
    output_dir = os.path.join(DETILT_MODEL_DIR, "cropped")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_cropped.tif")

    cv.imwrite(output_path, cropped_img)
    print(f"Saved: {output_path}")


# =========================
# PROCESS SINGLE IMAGE
# =========================

def process_single_image(input_path):
    img = cv.imread(input_path)

    if img is None:
        print(f"❌ Failed to load: {input_path}")
        return

    # (Replace this with your actual deskew output if needed)
    deskewed = img

    cropped = crop_margin_percent(deskewed, CROP_PERCENT)
    save_cropped_image(input_path, cropped)


# =========================
# PROCESS BATCH
# =========================

def process_folder(input_folder):
    if not os.path.exists(input_folder):
        print(f"❌ Folder not found: {input_folder}")
        return

    for file in os.listdir(input_folder):
        if file.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            process_single_image(path)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    if INPUT_MODE == "single":
        process_single_image(INPUT_PATH)

    elif INPUT_MODE == "batch":
        process_folder(INPUT_FOLDER)

    else:
        print("❌ Invalid INPUT_MODE. Use 'single' or 'batch'.")