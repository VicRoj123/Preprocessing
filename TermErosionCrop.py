import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("C:\\Users\\Cavem\\WebsterMLProcessed\\column_1.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Re-threshold (ensure binary)
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
binary = cv.bitwise_not(binary)  # white text on black background

# Morphological close to connect bold letters
kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel_close, iterations=1)

# Morphological open to remove thin noise lines
kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel_open, iterations=2)

# Connected components to filter small noise
num_labels, labels, stats, _ = cv.connectedComponentsWithStats(opened, connectivity=8)
min_area =500  # pixels; tune based on image resolution
mask = np.zeros(opened.shape, dtype=np.uint8)

for i in range(1, num_labels):  # skip background
    if stats[i, cv.CC_STAT_AREA] > min_area:
        mask[labels == i] = 255

# Further bold the text regions to ensure connectivity
kernel_h = cv.getStructuringElement(cv.MORPH_RECT, (40, 3))

# Apply morphological close to the mask
bold_bands = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_h, iterations=2)

# Project to find horizontal bands
projection = np.sum(bold_bands, axis=1)
# Identify peaks in the projection
threshold_val = np.max(projection) * 0.4
# Find rows above the threshold
rows = np.where(projection > threshold_val)[0]

# Group rows into bands
groups = []
for y in rows:
    if not groups or y - groups[-1][-1] > 10:
        groups.append([y])
    else:
        groups[-1].append(y)

term_lines = [int(np.mean(g)) for g in groups]

# Visualize detected terminal lines
color_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
for y in term_lines:
    cv.line(color_img, (0, y), (color_img.shape[1], y), (0, 0, 255), 2)

plt.figure(figsize=(10, 12))
plt.imshow(color_img)
plt.axis('off')
plt.show()
