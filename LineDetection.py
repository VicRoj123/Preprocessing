import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

img = cv.imread("C:\\Users\\Cavem\\Downloads\\Webster_1806\\AA00114919_00001\\00160.tif")

#converts imag to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Use Canny edge detection, tuned for highlight edges,  param = (image, minThresh, maxThresh), lower thresh = more edges(allows for weaker lines to show up), higher thresh = less edges
canny = cv.Canny(gray, 250, 400)

# Use HoughLines to detect lines, tuned for vertical lines
Stronglines = cv.HoughLinesP(canny, 1, np.pi/360, threshold=150, minLineLength=200, maxLineGap=15)
Weaklines = cv.HoughLinesP(canny, 1, np.pi/360, threshold=10, minLineLength=100, maxLineGap=10)

allLines = []
# Combine strong and weak lines
if Stronglines is not None:
    allLines.extend(Stronglines[:, 0])
if Weaklines is not None:
    allLines.extend(Weaklines[:, 0])

# Draw detected vertical lines on the original image
for x1, y1, x2, y2 in allLines:
    if abs(y2 - y1) > abs(x2 - x1):  # Vertical line
        cv.line(img, (x1, y1), (x2, y2), (255,0 , 0), 2)

# Convert BGR to RGB for displaying with matplotlib
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Save the output image
img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
#cv.imwrite("C:\\Users\\Cavem\\Downloads\\00894_output.tif", img_bgr)


plt.figure(figsize=(12, 12))
plt.imshow(img_rgb, cmap='gray')

plt.axis('off')
plt.show()
