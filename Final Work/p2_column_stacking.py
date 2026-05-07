import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans #pip install scikit-learn


"""
    Prepares the pages for OCR and cropping. Removes headers,
    detects column dividers, and stacks the three columns of each
    scan vertically for simpler processing. The second step in
    the pipeline.
"""

# Define input path
input_path = fr'preprocessed/00171_preprocessed.tif'

# Load image
img = cv.imread(input_path)

# Convert to grayscale for processing
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ========== HEADER REMOVAL ==========
# Invert so text is white on black for horizontal projection
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
binary = cv.bitwise_not(binary)

# Using horizontal projection to detect header/first line
# Sum pixels horizontally, then normalize
proj = np.sum(binary, axis=1)
proj_norm = proj / np.max(proj)

# Detect all rows with significant text density
text_rows = np.where(proj_norm > 0.15)[0]

if len(text_rows) == 0:
    raise ValueError("No text detected at top of page.")

#Find longest continuous block starting from the top, stored in header_block/block[0]
blocks = []
current = [text_rows[0]]
for r in text_rows[1:]:
    if r == current[-1] + 1:
        current.append(r)
    else:
        blocks.append(current)
        current = [r]
blocks.append(current)
header_block = blocks[0]

y0 = header_block[0]
y1 = header_block[-1]

#'''
#Add padding to account more of the header (better look)
pad = 10
y0 = max(0, y0 - pad)
y1 = min(binary.shape[0], y1 + pad)
#'''

''' Testing purpose, this is the detected header line
header_crop = img[y0:y1, :]
showImage(header_crop)
'''

#y1 is the bottom of the detected header line (already computed)
remove_start = y1 + 5 # remove header + small padding, increase padding if the bottom of the header still shows

#Catch, ensures we don't go out of bounds
remove_start = min(remove_start, img.shape[0] - 1)

#Crop original image from below the header to the bottom
header_removed = img[remove_start:, :]

#Display
#showImage(header_removed) #<-- new testing image, not grayscaled

#---End of Header Removal---
#---Start Column Detection/Removal---

#grayscale then edge detection with canny
gray = cv.cvtColor(header_removed, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 250, 400)

#two houghlines to detect the 2 verticle lines, then combine them
Stronglines = cv.HoughLinesP(canny, 1, np.pi/360, threshold=150, minLineLength=200, maxLineGap=10)
Weaklines = cv.HoughLinesP(canny, 1, np.pi/360, threshold=10, minLineLength=100, maxLineGap=5)

allLines = []

if Stronglines is not None:
    allLines.extend(Stronglines[:, 0])
if Weaklines is not None:
    allLines.extend(Weaklines[:, 0])

'''
#draw and visualize all lines
for x1, y1, x2, y2 in allLines:
    if abs(y2 - y1) > abs(x2 - x1):  # Vertical line
        cv.line(header_removed, (x1, y1), (x2, y2), (255,0 , 0), 2) #red line
'''

### Finding the 2 column lines (using k-means clustering)
#'''
#all midpoints of vertical lines, sorted to make sure that houghlines are in order
vert_centers = [
    int((x1 + x2) / 2)
    for x1, y1, x2, y2 in allLines
    if abs(y2 - y1) > abs(x2 - x1)
]
vert_centers = np.array(vert_centers).reshape(-1, 1)

#using k-means (need to pip install scikit-learn)
kmeans = KMeans(n_clusters=2, n_init="auto").fit(vert_centers)
cluster_centers = sorted(kmeans.cluster_centers_.flatten())

left_x, right_x = cluster_centers

# Draw detected columns
h = header_removed.shape[0]

#convert back to int, else float error
left_x = int(left_x)
right_x = int(right_x)

# cv.line(header_removed, (left_x, 0),  (left_x,  h), (0,255,0), 2)
# cv.line(header_removed, (right_x, 0), (right_x, h), (0,255,0), 2)
###
#'''

#Visual: Convert BGR to RGB for displaying with matplotlib
img_rgb = cv.cvtColor(header_removed, cv.COLOR_BGR2RGB)
img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

#---End of Column Detection/Removal---
#---Start of Columns Merge---

#split via columns, parameters [a:b, c:d], rows from a to b (top to bottom), columns from c to d, left to right
left = img_bgr[:, :left_x]
mid = img_bgr[:, left_x:right_x]
right = img_bgr[:, right_x:]

#Add padding (for output image)
#Make both columns the same width, first detect largest width, then add padding if smaller than width, then merge
w = max(left.shape[1], mid.shape[1], right.shape[1]) 

if left.shape[1] < w:
    left = cv.copyMakeBorder(left, 0, 0, 0, w - left.shape[1],  cv.BORDER_CONSTANT, value=(255,255,255))

if mid.shape[1] < w:
    mid = cv.copyMakeBorder(mid, 0, 0, 0, w - mid.shape[1], cv.BORDER_CONSTANT, value=(255,255,255))

if right.shape[1] < w:
    right = cv.copyMakeBorder(right, 0, 0, 0, w - right.shape[1], cv.BORDER_CONSTANT, value=(255,255,255))

merged = np.vstack([left, mid, right])

filename = os.path.splitext(os.path.basename(input_path))[0]
filename = filename.replace("_preprocessed", "")
output_path = f"stacked/{filename}_stacked.tif"
cv.imwrite(output_path, merged)

metadata = {
    'left_x': left_x,
    'right_x': right_x,
    'remove_start': int(remove_start)
}

#meta_path = f"stacked/{filename}_meta.json"
#with open(meta_path, 'w') as f:
 #   json.dump(metadata, f)

# Draw verification lines on the original image
verify_img = img.copy()
# Horizontal line at remove_start showing where header removal ends
cv.line(verify_img, (0, remove_start), (verify_img.shape[1], remove_start), (0, 255, 0), 3)
# Vertical lines at column dividers
cv.line(verify_img, (left_x, 0), (left_x, verify_img.shape[0]), (255, 0, 0), 3)
cv.line(verify_img, (right_x, 0), (right_x, verify_img.shape[0]), (0, 0, 255), 3)
cv.imwrite(f"stacked/{filename}_stacking_info.png", verify_img)