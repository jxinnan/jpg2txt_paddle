"""Page dewarping using page-dewarp (cubic sheet model)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Union

from PIL import Image

from jpg2txt.image_splitter import PageInfo

logger = logging.getLogger(__name__)


class PageDewarper:
    """Corrects page curvature, perspective tilt, and spine warping using page-dewarp."""

    A5_WIDTH_300DPI = 1748
    A5_HEIGHT_300DPI = 2480

    def __init__(self, output_dir: Union[str, Path] = "output/dewarped_pages") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def dewarp_page(
        self,
        input_image_path: Union[str, Path],
        target_path: Union[str, Path] | None = None,
        dpi: int = 300,
    ) -> Path:
        """Dewarp a single page image and center the result on a 300 DPI A5 canvas.
        
        Falls back gracefully to the original image if dewarping cannot converge.
        """
        in_path = Path(input_image_path)
        if target_path is None:
            out_path = self.output_dir / in_path.name
        else:
            out_path = Path(target_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from page_dewarp.image import WarpedImage
            from page_dewarp.options import Config

            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = Config(OUTPUT_DIR=tmpdir, NO_BINARY=1, DEBUG_LEVEL=0)
                warped = WarpedImage(in_path, config=cfg)

                generated_files = list(Path(tmpdir).glob("*.png"))
                if warped.written and generated_files:
                    dewarped_file = generated_files[0]
                    dewarped_img = Image.open(dewarped_file).convert("L")

                    # Center and scale onto 300 DPI A5 canvas
                    a5_canvas = Image.new("L", (self.A5_WIDTH_300DPI, self.A5_HEIGHT_300DPI), color=255)
                    dw, dh = dewarped_img.size

                    # Allow maximum fit up to 94% of canvas dimension to maintain comfortable margins
                    scale = min(
                        (self.A5_WIDTH_300DPI * 0.94) / dw,
                        (self.A5_HEIGHT_300DPI * 0.94) / dh,
                        1.0,
                    )
                    new_w = max(1, int(dw * scale))
                    new_h = max(1, int(dh * scale))
                    dewarped_scaled = dewarped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    pos_x = (self.A5_WIDTH_300DPI - new_w) // 2
                    pos_y = (self.A5_HEIGHT_300DPI - new_h) // 2
                    a5_canvas.paste(dewarped_scaled, (pos_x, pos_y))
                    a5_canvas.save(out_path, dpi=(dpi, dpi))
                    return out_path
                else:
                    logger.warning("page-dewarp produced no output for %s; using original", in_path.name)
        except Exception as exc:
            logger.warning("page-dewarp failed on %s (%s); falling back to original", in_path.name, exc)

        # Fallback: copy / re-save original image centered on A5
        with Image.open(in_path) as orig_img:
            orig_gray = orig_img.convert("L")
            orig_gray.save(out_path, dpi=(dpi, dpi))
        return out_path

    def dewarp_all_pages(
        self,
        pages_info: List[PageInfo],
        progress_callback=None,
        force: bool = False,
    ) -> List[PageInfo]:
        """Dewarp all split A5 pages and return updated PageInfo objects."""
        dewarped_pages: List[PageInfo] = []
        total = len(pages_info)

        for idx, page in enumerate(pages_info):
            src_file = page.image_path
            if src_file is None or not src_file.exists():
                continue

            target_file = self.output_dir / src_file.name
            if not target_file.exists() or force:
                self.dewarp_page(src_file, target_file)

            dewarped_pages.append(
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

        return dewarped_pages
