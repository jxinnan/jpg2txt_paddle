# jpg2txt - Theatrical Play Script OCR Pipeline

An automated Python OCR pipeline that converts two-page scanned spreads into structured plain text and like-for-like A5 PDFs using OpenCV image preprocessing and Tesseract OCR.

## Features

- **Split & Deskew**: Splits landscape 2-up scans into individual A5 portrait pages with spine gutter shadow removal.
- **Page Dewarping (`page-dewarp`)**: Applies cubic sheet modeling as the first preprocessing step to flatten curled pages, correct perspective distortion, and straighten text lines.
- **OpenCV Preprocessing**: Applies 5x5 Gaussian blur filtering, Otsu's thresholding (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`), and post-Otsu morphological dilation (`cv2.dilate` with $2 \times 2$ structuring element to thin character strokes and separate touching letters).
- **Intermediate Storage**:
  - `output/dewarped_pages/`: 64 dewarped grayscale 300 DPI A5 images.
  - `output/preprocessed_pages/`: 64 dewarped + binarized + dilated 300 DPI A5 images.
- **Version Snapshots**:
  - `output_v1_otsu/`: Version 1 outputs (Otsu thresholding only).
  - `output_v2_dewarp/`: Version 2 outputs (Dewarping + Otsu thresholding + Erosion test).
  - `output/`: Current version (Dewarping + Otsu thresholding + Dilation).
- **Tesseract OCR**: High-accuracy text recognition with 300 DPI upscaling and theatrical script layout processing.
- **Strict Verbatim Rule**: Strictly preserves text seen in scans without filling in missing or cut-off words.
- **Output 1 (Structured .txt)**: Clean plain text of the entire play with running headers, page numbers, and divider lines stripped.
- **Output 2 (Like-for-Like A5 PDF)**: 
  - `undercover.pdf` / `undercover_searchable.pdf`: 64 individual A5 pages ($419.52 \times 595.20\text{ pt}$) matching the original document with an invisible searchable OCR text layer.
  - `undercover_typeset.pdf`: Clean digital typeset A5 PDF reconstructed with script styling.

## Quick Start with uv

```bash
# 1. Setup environment and Tesseract
uv sync
bash scripts/setup_tesseract.sh

# 2. Run full OCR pipeline
uv run python -m jpg2txt.cli --all

# Pipeline options:
# uv run python -m jpg2txt.cli --dewarp-only       # Only split and dewarp images
# uv run python -m jpg2txt.cli --preprocess-only   # Split, dewarp, threshold, and erode images
# uv run python -m jpg2txt.cli --ocr-only          # Run OCR on preprocessed pages
# uv run python -m jpg2txt.cli --pdf-only          # Generate PDFs from preprocessed pages
# uv run python -m jpg2txt.cli --erosion-ksize 2   # Set erosion structuring element size
# uv run python -m jpg2txt.cli --no-erosion        # Disable post-Otsu morphological erosion
```

## Generated Outputs

Outputs are saved in the `output/` directory:
- `output/undercover.txt`: Output 1 (Structured script text without headers)
- `output/undercover.pdf`: Output 2 (Like-for-like A5 searchable PDF)
- `output/undercover_searchable.pdf`: 64-page A5 PDF with searchable text layer
- `output/undercover_typeset.pdf`: Reconstructed typeset companion edition
- `output/pages/`: 64 individual cropped A5 raw page images
- `output/dewarped_pages/`: 64 dewarped grayscale 300 DPI A5 images
- `output/preprocessed_pages/`: 64 preprocessed binarized + eroded 300 DPI A5 page images

