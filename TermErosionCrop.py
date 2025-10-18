import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load and grayscale 
img = cv.imread("C:\\Users\\Cavem\\WebsterMLProcessed\\column_1.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Binary threshold (Otsu) 
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
binary = cv.bitwise_not(binary)   # text becomes white on black background

# Mild morphological closing to connect bold letters 
kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel_close, iterations=1)

# Remove tiny components (thin or faint text) 
num_labels, labels, stats, _ = cv.connectedComponentsWithStats(closed, connectivity=8)
min_area = 400   # increase to make stricter, decrease if terms vanish
mask = np.zeros(closed.shape, dtype=np.uint8)

for i in range(1, num_labels):       # skip background
    if stats[i, cv.CC_STAT_AREA] > min_area:
        mask[labels == i] = 255

#  Optional horizontal merge (helps if bold terms are broken) 
kernel_h = cv.getStructuringElement(cv.MORPH_RECT, (40, 3))
bold_bands = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_h, iterations=1)

# Horizontal projection to locate bold zones 
projection = np.sum(bold_bands, axis=1)
threshold_val = np.max(projection) * 0.55   # raise toward 0.6 if too many lines
rows = np.where(projection > threshold_val)[0]

#  Group consecutive rows into single term lines 
groups = []
for y in rows:
    if not groups or y - groups[-1][-1] > 10:
        groups.append([y])
    else:
        groups[-1].append(y)

term_lines = [int(np.mean(g)) for g in groups]

#  Visualization 
color_img = img.copy()
for y in term_lines:
    cv.line(color_img, (0, y), (color_img.shape[1], y), (0, 0, 255), 2)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap='gray')
plt.title("Bold Term Mask")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(color_img, cv.COLOR_BGR2RGB))
plt.title("Detected Term Lines")
plt.axis('off')
plt.show()

print(f"Detected {len(term_lines)} potential term headings")
