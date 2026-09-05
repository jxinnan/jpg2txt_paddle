"""OCR Engine using PaddlePaddle/PaddleOCR-VL-1.6 with Hugging Face Transformers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Union

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face authentication token if available."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        try:
            return token_file.read_text().strip()
        except OSError:
            pass
    return None


class PaddleOCREngine:
    """Wrapper around PaddlePaddle/PaddleOCR-VL-1.6 for document OCR and Markdown generation."""

    def __init__(
        self,
        model_id: str = "PaddlePaddle/PaddleOCR-VL-1.6",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_pixels: int = 2048 * 28 * 28,
        token: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if self.device == "cuda" else torch.float32
        )
        self.max_pixels = max_pixels
        self.token = token or get_hf_token()

        print(f"Loading PaddleOCR-VL model '{self.model_id}' on {self.device} ({self.torch_dtype})...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=self.token,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            token=self.token,
        ).to(self.device).eval()
        print("PaddleOCR-VL model loaded successfully!")

    def image_to_markdown(
        self,
        image_input: Union[str, Path, Image.Image],
        task_prompt: str = "OCR:",
        max_new_tokens: int = 4096,
    ) -> str:
        """Run OCR on a single image and return structured Markdown text."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        else:
            img = image_input.convert("RGB")

        orig_w, orig_h = img.size
        # Upscale smaller images for better text recognition if needed
        if orig_w < 1500 and orig_h < 1500:
            resample_filter = getattr(Image, "Resampling", Image).LANCZOS
            img = img.resize((orig_w * 2, orig_h * 2), resample_filter)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": task_prompt},
                ],
            }
        ]

        shortest_edge = getattr(
            self.processor.image_processor, "min_pixels", 28 * 28
        )
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "images_kwargs": {
                    "size": {
                        "shortest_edge": shortest_edge,
                        "longest_edge": self.max_pixels,
                    }
                }
            },
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        decoded = self.processor.decode(outputs[0][input_len:-1], skip_special_tokens=True)
        return decoded.strip()

    def process_image_folder(
        self,
        folder_path: Union[str, Path],
        output_dir: Union[str, Path],
        combined_filename: str = "combined.md",
        task_prompt: str = "OCR:",
        max_new_tokens: int = 2048,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Path]:
        """Process all images in a folder and save individual & combined Markdown files."""
        folder = Path(folder_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".bmp")]
        )

        total = len(image_files)
        saved_md_files: List[Path] = []
        combined_sections: List[str] = []

        for idx, img_path in enumerate(image_files):
            stem = img_path.stem
            out_md_path = out_dir / f"{stem}.md"

            if not overwrite and out_md_path.exists() and out_md_path.stat().st_size > 0:
                md_text = out_md_path.read_text(encoding="utf-8")
            else:
                md_text = self.image_to_markdown(
                    img_path,
                    task_prompt=task_prompt,
                    max_new_tokens=max_new_tokens,
                )
                out_md_path.write_text(md_text, encoding="utf-8")

            saved_md_files.append(out_md_path)
            combined_sections.append(f"<!-- Page: {stem} -->\n\n{md_text}\n")

            if progress_callback:
                progress_callback(idx + 1, total, stem)

        if combined_filename:
            combined_path = out_dir / combined_filename
            combined_path.write_text("\n\n---\n\n".join(combined_sections), encoding="utf-8")

        return saved_md_files

