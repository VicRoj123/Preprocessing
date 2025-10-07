import cv2 as cv

# Load the image (your processed page with lines)
img = cv.imread("C:\\Users\\Cavem\\Downloads\\00894.tif")

# Approximate vertical divider positions
# You can fine-tune these based on your detected line X positions
column_dividers = [1365, 2300]  #update after checking your output visually, will check for a more automated way later

# Sort and add edges of the page
column_dividers = sorted(column_dividers)
boundaries = [0] + column_dividers + [img.shape[1]]

# Split and save each column
for i in range(len(boundaries) - 1):
    x_start = boundaries[i]
    x_end = boundaries[i + 1]
    column = img[:, x_start:x_end]

    # Save each column as its own TIF file
    output_path = fr"C:\Users\Cavem\WebsterMLProcessed\column_{i+1}.png"
    cv.imwrite(output_path, column)
    print(f"Saved: {output_path}")
