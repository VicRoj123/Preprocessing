import cv2 as cv
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans # pip install scikit-learn
from PIL import Image
import pytesseract

# ====================== Change the two paths before running! ======================
input_dir = Path(r"C:\Users\Kenri\Downloads\A20 - ocr - python\1828 Work\1828Hlinetest")
output_dir = Path(r"C:\Users\Kenri\Downloads\A20 - ocr - python\1828 Work\1828_Hcropped")

output_dir.mkdir(parents=True, exist_ok=True)

#Takes the image boxed and crops it to the output directory
def save_cropped_lines(img_bgr, line_boxes, output_folder, prefix="line"):
    """
    Crop each line from an image and save it as a separate TIFF in output_folder.
    
    Parameters:
    - img_bgr: OpenCV BGR image
    - line_boxes: list of bounding boxes [(x, y, w, h), ...]
    - output_folder: Path to save the cropped lines
    - prefix: string prefix for filenames
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for i, (x, y, w, h) in enumerate(line_boxes, start=1):
        cropped = img_bgr[y:y+h, x:x+w]
        filename = output_folder / f"{prefix}_{i:03d}.tif"
        cv.imwrite(str(filename), cropped)

#Using Tesseract for Horizontal Line Detection, outputs the tif file with the detected lines highlighted (adds 10 seconds to runtime)
def highlight_lines_tesseract(img_bgr, lang='eng'):
    """
    Highlight horizontal text lines in a dictionary page using Tesseract.
    
    Parameters:
    - img_bgr: OpenCV BGR image
    - lang: language code for Tesseract (default 'eng')
    
    Returns:
    - img_marked: BGR image with bounding boxes drawn around each line
    - lines: list of bounding boxes [(x, y, w, h), ...] for each line
    """
    img_marked = img_bgr.copy()
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    
    #get line-level bounding boxes
    data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT, lang=lang)
    
    n_boxes = len(data['level'])
    lines = []
    
    for i in range(n_boxes):
        if data['level'][i] == 4:  # level=4 corresponds to lines
            x, y, w, h = (data['left'][i], data['top'][i],
                          data['width'][i], data['height'][i])
            #draw green rectangle
            cv.rectangle(img_marked, (x, y), (x + w, y + h), (0, 255, 0), 1)
            lines.append((x, y, w, h))
    
    return img_marked, lines

#MAIN FUNCTION USED
def process_tif(image_path: Path, output_root: Path):
    img = cv.imread(str(image_path))
    #check exception when opening the file
    if img is None:
        print(f"Could not read {image_path.name}")
        return

    #create output folder for the image
    out_dir = output_root / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- HEADER REMOVAL ---
    #apply preprocessing method to remove noise
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    binary = cv.bitwise_not(binary)

    #use projection to find the header (the first line with sufficient text)
    proj = np.sum(binary, axis=1)
    proj_norm = proj / np.max(proj)
    text_rows = np.where(proj_norm > 0.15)[0] #<-- adjust proj_norm if needed
    #catch exception for empty page
    if len(text_rows) == 0:
        print(f"No header detected in {image_path.name}")
        return

    #
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
    y0, y1 = header_block[0], header_block[-1]

    #remove additional horizontal layers to account for the size of the text
    pad = 10
    y0 = max(0, y0 - pad)
    y1 = min(binary.shape[0], y1 + pad)
    remove_start = min(y1 + 30, img.shape[0] - 1)
    header_removed = img[remove_start:, :]

    # --- COLUMN DETECTION ---
    #preprocess, then use strongline weakline method, adjust the parameters if needed
    gray = cv.cvtColor(header_removed, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 250, 400)

    Stronglines = cv.HoughLinesP(
        canny, 1, np.pi / 360, threshold=150, minLineLength=200, maxLineGap=10
    )
    Weaklines = cv.HoughLinesP(
        canny, 1, np.pi / 360, threshold=10, minLineLength=100, maxLineGap=5
    )

    allLines = []
    if Stronglines is not None:
        allLines.extend(Stronglines[:, 0])
    if Weaklines is not None:
        allLines.extend(Weaklines[:, 0])

    vert_centers = [
        int((x1 + x2) / 2)
        for x1, y1, x2, y2 in allLines
        if abs(y2 - y1) > abs(x2 - x1)
    ]

    if len(vert_centers) < 2:
        print(f"Not enough vertical lines in {image_path.name}")
        return

    #use k-means to detect the two columns
    vert_centers = np.array(vert_centers).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init="auto").fit(vert_centers)
    left_x, right_x = sorted(kmeans.cluster_centers_.flatten().astype(int))

    #column split
    left = header_removed[:, :left_x]
    mid = header_removed[:, left_x:right_x]
    right = header_removed[:, right_x:]

    #change BGR to RGB for pillow
    left = cv.cvtColor(left, cv.COLOR_BGR2RGB)
    mid = cv.cvtColor(mid, cv.COLOR_BGR2RGB)
    right = cv.cvtColor(right, cv.COLOR_BGR2RGB)

    #call tesseract function to detect horizontal rows
    leftMark, leftLines  = highlight_lines_tesseract(left)
    midMark, midLines   = highlight_lines_tesseract(mid)
    rightMark, rightLines = highlight_lines_tesseract(right)
    
    #used to visualize the rows (good to run before actually cropping)
    Image.fromarray(leftMark).save(out_dir / "left.tif")
    Image.fromarray(midMark).save(out_dir / "mid.tif")
    Image.fromarray(rightMark).save(out_dir / "right.tif")

    #crops the tif image based on the previous method, for testing, comment out the next 3 lines
    #save_cropped_lines(left, leftLines, output_dir, prefix="left")
    #save_cropped_lines(mid, midLines, output_dir, prefix="mid")
    #save_cropped_lines(right, rightLines, output_dir, prefix="right")
    #The above 3 lines should be used for production, currently only works with 1 tif pages

    print(f"Saved columns to {output_dir}")

# ======================MAIN======================: For every loop --> process_tif --> tesseractfunc --> trimfunc
for tif_path in sorted(input_dir.glob("*.tif")):
    print(f"Processing {tif_path.name}")
    process_tif(tif_path, output_dir)

print("Batch processing complete.")
