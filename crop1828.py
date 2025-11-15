import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image_path = fr'C:\Users\Kenri\Downloads\A20 - ocr - python\1828 Work\1828_Pages\00500.tif' ###replace with your file path
#output_path = fr'C:\Users\Kenri\Downloads\A20 - ocr - python\1828 Work\1828_cropped_00500.tif' ###replace with your file output path

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
showImage(header_removed)

#---End of Header Removal---

