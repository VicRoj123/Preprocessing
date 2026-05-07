# Noah Webster Dictionary Digitization – Preprocessing Pipeline

## Overview

The Noah Webster Dictionary Digitization project is a computer vision and OCR preprocessing pipeline designed to assist in the large-scale extraction and preservation of historical dictionary entries from scanned Noah Webster dictionary pages. The project focuses on cleaning, correcting, segmenting, and preparing scanned pages for downstream OCR and lexical extraction.

The preprocessing system was developed to handle historical scans that contain:

* Skewed or rotated pages
* Multi-column layouts
* Uneven lighting and aged paper artifacts
* Noise and scan degradation
* Cross-column and cross-page dictionary entries
* Dense formatting and bolded headwords

The pipeline combines traditional computer vision techniques with machine learning-based deskewing methods to improve OCR readability and extraction quality.

---


# Technologies Used

## Languages

* Python

## Computer Vision / ML Libraries

* OpenCV
* PyTorch
* NumPy
* PIL
* pytesseract
* matplotlib

## GUI / Utilities

* Tkinter
* JSON
* os

---

# Pipeline Architecture

The project is organized into multiple preprocessing stages.

## Stage 1 – Deskewing

* Uses a trained machine learning model to estimate page rotation
* Corrects skew automatically
* Removes artificial border padding introduced during rotation

## Stage 2 – Preprocessing

Initial page cleanup:

* Grayscale conversion
* Denoising
* Adaptive thresholding
* Morphological cleanup

## Stage 3 – Column Separation

* Detects dictionary columns
* Separates pages into individual column regions
* Maintains mapping information for later reconstruction


