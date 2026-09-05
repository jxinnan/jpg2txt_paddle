"""Image splitting and preprocessing for 2-up scanned play script spreads."""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image


@dataclass
class PageInfo:
    """Information and image data for a single A5 script page."""
    page_number: int
    image_index: int
    is_left: bool
    source_path: Path
    image_path: Path | None = None
    image: Image.Image | None = None


class ImageSplitter:
    """Splits 2-up landscape scans into individual A5 portrait pages."""

    # Standard A5 proportions at 300 DPI
    A5_WIDTH: int = 1748
    A5_HEIGHT: int = 2480

    def __init__(self, input_dir: str | Path, output_pages_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_pages_dir = Path(output_pages_dir)
        self.output_pages_dir.mkdir(parents=True, exist_ok=True)

    def get_sorted_images(self) -> List[Tuple[int, Path]]:
        """Find and return all input image files sorted by numeric prefix."""
        files = list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.jpeg"))
        parsed: List[Tuple[int, Path]] = []
        for f in files:
            m = re.search(r"(\d+)-", f.name)
            if m:
                parsed.append((int(m.group(1)), f))
            else:
                # Fallback to sorting by name if pattern doesn't match
                parsed.append((9999, f))
        parsed.sort(key=lambda x: (x[0], x[1].name))
        return parsed

    @staticmethod
    def detect_gutter_x(img: Image.Image) -> int:
        """Dynamically detect the darkest vertical column in the center spine region."""
        gray = img.convert("L")
        w, h = gray.size
        # Resample image vertically down to 1 pixel height to get column averages
        col_averages_img = gray.resize((w, 1), Image.Resampling.BOX)
        pixels = col_averages_img.tobytes()
        
        # Center spine search window: between x=420 and x=495 for 904-width images
        search_start = max(0, int(w * 0.46))
        search_end = min(w, int(w * 0.55))
        
        darkest_x = min(range(search_start, search_end), key=lambda x: pixels[x])
        return darkest_x

    def create_a5_page_image(self, cropped: Image.Image) -> Image.Image:
        """Place cropped page content centered on an A5 canvas at 300 DPI."""
        cw, ch = cropped.size
        target_w, target_h = self.A5_WIDTH, self.A5_HEIGHT

        # Scale cropped page to fit within target A5 dimensions with slight margin
        margin_w = int(target_w * 0.04)
        margin_h = int(target_h * 0.04)
        avail_w = target_w - 2 * margin_w
        avail_h = target_h - 2 * margin_h

        scale = min(avail_w / cw, avail_h / ch)
        new_w = int(cw * scale)
        new_h = int(ch * scale)

        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create clean white A5 background
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))

        return canvas

    def process_all_images(self, save_pages: bool = True) -> List[PageInfo]:
        """Split all scanned spread images into 64 individual A5 pages."""
        sorted_images = self.get_sorted_images()
        pages: List[PageInfo] = []

        for idx, (img_num, img_path) in enumerate(sorted_images):
            # Calculate page numbers: image 1 has pages 108 and 109
            page_left_num = 108 + (img_num - 1) * 2
            page_right_num = page_left_num + 1

            with Image.open(img_path) as raw_img:
                w, h = raw_img.size
                gutter_x = self.detect_gutter_x(raw_img)

                # Crop left page: from x=0 to gutter, avoiding gutter shadow
                left_crop_box = (0, 0, max(gutter_x - 10, 435), h)
                left_cropped = raw_img.crop(left_crop_box)
                left_a5 = self.create_a5_page_image(left_cropped)

                # Crop right page: from gutter shadow edge to right border
                right_crop_box = (min(gutter_x + 15, 510), 0, w, h)
                right_cropped = raw_img.crop(right_crop_box)
                right_a5 = self.create_a5_page_image(right_cropped)

            left_page_path = self.output_pages_dir / f"page_{page_left_num:03d}.png"
            right_page_path = self.output_pages_dir / f"page_{page_right_num:03d}.png"

            if save_pages:
                left_a5.save(left_page_path, dpi=(300, 300))
                right_a5.save(right_page_path, dpi=(300, 300))

            pages.append(
                PageInfo(
                    page_number=page_left_num,
                    image_index=img_num,
                    is_left=True,
                    source_path=img_path,
                    image_path=left_page_path,
                    image=left_a5,
                )
            )
            pages.append(
                PageInfo(
                    page_number=page_right_num,
                    image_index=img_num,
                    is_left=False,
                    source_path=img_path,
                    image_path=right_page_path,
                    image=right_a5,
                )
            )

        return pages

