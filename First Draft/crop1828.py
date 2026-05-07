#IMPORTANT!!!Make sure to edit image_path AND output_path before using this code
#################################################################################
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans #pip install scikit-learn

image_path = fr'C:\Users\Kenri\Downloads\A20 - ocr - python\1828 Work\1828_Pages\00500.tif' ###replace with your file path
output_path = fr'C:\Users\Kenri\Downloads\A20 - ocr - python\1828 Work\1828_cropped_00500.tif' ###replace with your file output path

#helper function(s):

#takes a image and shows it after compiling
def showImage(showing):
    plt.figure(figsize=(10, 6))
    plt.imshow(cv.cvtColor(showing, cv.COLOR_BGR2RGB))
    plt.title("show image")
    plt.axis('off')
    plt.show()

# ---START OF CODE---
# ---Header Removal---

#Basic Loading: load and covert to grayscale (for ease of processing), then to white text on black background
img = cv.imread(image_path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
binary = cv.bitwise_not(binary)   # text becomes white on black background

#Using horizontal projection to detect header/first line
#Sum pixels horizontally, then normalize
proj = np.sum(binary, axis=1)
proj_norm = proj / np.max(proj)

#Detect all rows with significant text density
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
remove_start = y1 + 30  # remove header + small padding, increase padding if the bottom of the header still shows

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

cv.line(header_removed, (left_x, 0),  (left_x,  h), (0,255,0), 2)
cv.line(header_removed, (right_x, 0), (right_x, h), (0,255,0), 2)
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

cv.imwrite(output_path, merged)
#showImage(img_bgr)