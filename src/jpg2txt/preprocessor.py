"""Image preprocessing using OpenCV with Gaussian filtering and Otsu's thresholding."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from jpg2txt.image_splitter import PageInfo


class ImagePreprocessor:
    """Preprocesses scanned document pages to remove noise and optimize for OCR."""

    def __init__(
        self,
        output_dir: str | Path = "output/preprocessed_pages",
        dilate_ksize: int = 2,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dilate_ksize = dilate_ksize

    @staticmethod
    def preprocess_cv2_image(
        img: np.ndarray,
        gaussian_ksize: int = 5,
        dilate_ksize: int = 2,
        border_cleanup_px: int = 25,
    ) -> np.ndarray:
        """Apply Gaussian blur noise reduction, Otsu's binarization, and dilation to an image.
        
        Following OpenCV document thresholding & morphological processing standards:
        1. Grayscale conversion
        2. 5x5 Gaussian blur to suppress scanner noise and paper grain
        3. Otsu's thresholding to compute optimal global bimodal threshold
        4. Dilation using a structuring element (expands white background / thins character strokes)
        5. Clean outer borders to eliminate scanner glass/gutter edge slivers
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Gaussian blur for high-frequency noise suppression
        blur = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), 0)

        # Otsu's binarization
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Small amount of dilation after Otsu's thresholding (thins dark text strokes)
        if dilate_ksize > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))
            binary = cv2.dilate(binary, kernel, iterations=1)

        # Blank extreme outer scanner margins to avoid edge frame artifacts
        if border_cleanup_px > 0:
            h, w = binary.shape
            b = min(border_cleanup_px, h // 20, w // 20)
            binary[:b, :] = 255
            binary[-b:, :] = 255
            binary[:, :b] = 255
            binary[:, -b:] = 255

        return binary

    def process_file(
        self,
        input_image_path: Union[str, Path],
        output_image_path: Union[str, Path] | None = None,
        dpi: int = 300,
        dilate_ksize: int | None = None,
    ) -> Path:
        """Preprocess a single page image file and save the result with DPI metadata."""
        in_path = Path(input_image_path)
        if output_image_path is None:
            out_path = self.output_dir / in_path.name
        else:
            out_path = Path(output_image_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        img_bgr = cv2.imread(str(in_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image at {in_path}")

        k = self.dilate_ksize if dilate_ksize is None else dilate_ksize
        binary_cv2 = self.preprocess_cv2_image(img_bgr, dilate_ksize=k)

        # Save via Pillow to ensure 300 DPI metadata is written for Tesseract
        pil_img = Image.fromarray(binary_cv2)
        pil_img.save(out_path, dpi=(dpi, dpi))
        return out_path

    def process_all_pages(
        self,
        pages_info: List[PageInfo],
        progress_callback=None,
    ) -> List[PageInfo]:
        """Preprocess all split A5 pages and return updated PageInfo objects."""
        preprocessed_pages: List[PageInfo] = []
        total = len(pages_info)

        for idx, page in enumerate(pages_info):
            src_file = page.image_path
            if src_file is None or not src_file.exists():
                continue

            target_file = self.output_dir / src_file.name
            self.process_file(src_file, target_file)

            preprocessed_pages.append(
                PageInfo(
                    page_number=page.page_number,
                    image_index=page.image_index,
                    is_left=page.is_left,
                    source_path=page.source_path,
                    image_path=target_file,
                )
            )

            if progress_callback:
                progress_callback(idx + 1, total)

        return preprocessed_pages

