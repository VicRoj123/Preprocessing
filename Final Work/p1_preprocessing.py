import cv2 as cv
import os

"""
    Initial and core page preprocessing. Cleans, denoises,
    enhances, and binarizes scans for better results with OCR.
    The first step in the pipeline.
"""

# Define input path
input_path = "C:\\Users\\Cavem\\Downloads\\Programs\\webstersdictionary\\test_images\\00171.tif"

# Load image
img = cv.imread(input_path)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Morphological opening (erosion + dilation) to remove small noise
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
opened = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)

# More aggressive denoising
denoised = cv.fastNlMeansDenoising(opened, h=25, templateWindowSize=7, searchWindowSize=21)

# Contrast enhancement w/ CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(denoised)

# Binarization via adaptive thresholding
binary = cv.adaptiveThreshold(
    enhanced, 
    255, 
    cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv.THRESH_BINARY, 
    blockSize=15,
    C=9
)

# Morphological closing to tidy up breaks and gaps
kernel_clean = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))
cleaned = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel_clean)

# Save the preprocessed result
filename = os.path.splitext(os.path.basename(input_path))[0]
output_path = f"preprocessed/{filename}_preprocessed.tif"
cv.imwrite(output_path, cleaned)