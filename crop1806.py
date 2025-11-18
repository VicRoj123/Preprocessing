import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image_path = fr'C:\Users\Kenri\Downloads\A20 - ocr - python\1806 Work\1806_Pages\00385.tif' ###replace with your file path
output_path = fr'C:\Users\Kenri\Downloads\A20 - ocr - python\1806 Work\1806_cropped_385.tif' ###replace with your file output path

#START BY CROPPING OUT THE HEADER
# Load and grayscale 
img = cv.imread(image_path)
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

'''
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
'''

#if the header line is detected, display the image after cropping, works well
if term_lines:
    lowest_y = max(term_lines)
    #print(f"Lowest header line at y = {lowest_y}")
    trimmed = img[lowest_y:, :]  
    '''
    # Crop below that line
    plt.figure(figsize=(10, 6))
    plt.imshow(cv.cvtColor(trimmed, cv.COLOR_BGR2RGB))
    plt.title("Trimmed Image (removed header)")
    plt.axis('off')
    plt.show()
    '''
else:
    print("No term lines detected.")

#finished cropping out header, starting column detection ---

gray = cv.cvtColor(trimmed, cv.COLOR_BGR2GRAY)

# Use Canny edge detection, tuned for highlight edges,  param = (image, minThresh, maxThresh), lower thresh = more edges(allows for weaker lines to show up), higher thresh = less edges
canny = cv.Canny(gray, 250, 400)

# Use HoughLines to detect lines, tuned for vertical lines
#original line gap, 15s-10w
Stronglines = cv.HoughLinesP(canny, 1, np.pi/360, threshold=150, minLineLength=200, maxLineGap=10)
Weaklines = cv.HoughLinesP(canny, 1, np.pi/360, threshold=10, minLineLength=100, maxLineGap=5)

##
allLines = []
# Combine strong and weak lines
if Stronglines is not None:
    allLines.extend(Stronglines[:, 0])
if Weaklines is not None:
    allLines.extend(Weaklines[:, 0])

#'''show
# Draw detected vertical lines on the original image
for x1, y1, x2, y2 in allLines:
    if abs(y2 - y1) > abs(x2 - x1):  # Vertical line
        cv.line(trimmed, (x1, y1), (x2, y2), (255,0 , 0), 2) #red line
#'''

split_x = int(np.median([int((x1 + x2) / 2) for x1, y1, x2, y2 in allLines if abs(y2 - y1) > abs(x2 - x1)]))

# Convert BGR to RGB for displaying with matplotlib
img_rgb = cv.cvtColor(trimmed, cv.COLOR_BGR2RGB)

# Save the output image
img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
#cv.imwrite(output_path, img_bgr)

'''show
plt.figure(figsize=(12, 12))
plt.imshow(img_bgr, cmap='gray')
plt.axis('off')
plt.show()
'''

'''
#finding midpoint ---
mid_img = img_bgr.copy()
img_center = mid_img.shape[1] // 2
# Choose line whose x position is closest to image center
line_centers = [(int((x1 + x2) / 2), (x1, y1, x2, y2)) for x1, y1, x2, y2 in allLines]
split_x, best_line = min(line_centers, key=lambda l: abs(l[0] - img_center))
print(f"Detected main vertical divider at x = {split_x}")

#show
# Draw just that one
divider_img = img_bgr.copy()
cv.line(divider_img, (split_x, 0), (split_x, img_bgr.shape[0]), (0, 255, 0), 2) #blue green
'''

'''show
plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(divider_img, cv.COLOR_BGR2RGB))
plt.title("Main Column Divider Detected")
plt.axis('off')
plt.show()
'''

#'''
#output cropped image as a single column
left = img_bgr[:, :split_x]
right = img_bgr[:, split_x:]

# Add padding (for output image)
# Make both columns the same width
w = max(left.shape[1], right.shape[1])  # maximum width
h_left, h_right = left.shape[0], right.shape[0]

# Pad the left image if needed
if left.shape[1] < w:
    pad_w = w - left.shape[1]
    left = cv.copyMakeBorder(left, 0, 0, 0, pad_w, cv.BORDER_CONSTANT, value=(255,255,255))

# Pad the right image if needed
if right.shape[1] < w:
    pad_w = w - right.shape[1]
    right = cv.copyMakeBorder(right, 0, 0, 0, pad_w, cv.BORDER_CONSTANT, value=(255,255,255))

# Now safe to merge vertically
merged = np.vstack([left, right])

cv.imwrite(output_path, merged)
#'''
