"""Command Line Interface for PaddleOCR-VL-1.6 multi-stage page processing and comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from jpg2txt.ocr_engine import PaddleOCREngine


def compute_stage_stats(md_dir: Path) -> Dict[str, float]:
    """Calculate aggregate statistics across markdown files in a directory."""
    md_files = sorted(md_dir.glob("page_*.md"))
    total_words = 0
    total_lines = 0
    total_chars = 0

    for f in md_files:
        text = f.read_text(encoding="utf-8")
        total_words += len(text.split())
        total_lines += len([l for l in text.splitlines() if l.strip()])
        total_chars += len(text)

    num_pages = len(md_files)
    return {
        "pages": num_pages,
        "words": total_words,
        "lines": total_lines,
        "chars": total_chars,
        "avg_words_per_page": round(total_words / num_pages, 1) if num_pages else 0,
        "avg_lines_per_page": round(total_lines / num_pages, 1) if num_pages else 0,
    }


def generate_comparison_report(
    output_base: Path,
    stages: Dict[str, Path],
    sample_pages: List[str] | None = None,
) -> Path:
    """Generate a Markdown comparison report evaluating outputs from the three stages."""
    report_path = output_base / "comparison_report.md"
    stats: Dict[str, Dict[str, float]] = {}

    for stage_name, stage_dir in stages.items():
        stats[stage_name] = compute_stage_stats(stage_dir)

    # Determine representative sample pages if not provided
    if sample_pages is None:
        first_stage = next(iter(stages.values()))
        available = sorted([f.stem for f in first_stage.glob("page_*.md")])
        if len(available) >= 3:
            # Pick an early page (e.g. index 1 or 3), a middle page, and a later page
            idx1 = min(3, len(available) - 1) if len(available) > 4 else 0
            idx2 = len(available) // 2
            idx3 = max(0, len(available) - 2)
            sample_pages = [available[idx1], available[idx2], available[idx3]]
        elif available:
            sample_pages = available
        else:
            sample_pages = ["page_108", "page_120", "page_140"]

    lines: List[str] = [
        "# PaddleOCR-VL-1.6 OCR Output Comparison Report",
        "",
        "This report compares the OCR outputs across the three levels of page processing:",
        "1. **Raw Pages (`pages`)**: Scanned spreads split directly into A5 pages.",
        "2. **Dewarped Pages (`dewarped_pages`)**: Split pages straightened via page-dewarp (cubic sheet model).",
        "3. **Preprocessed Pages (`preprocessed_pages`)**: Dewarped pages filtered with Gaussian blur and Otsu binarization.",
        "",
        "## Summary Statistics Table",
        "",
        "| Processing Stage | Pages Processed | Total Words | Total Lines | Total Characters | Avg Words/Page | Avg Lines/Page |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for stage_name in ("pages", "dewarped", "preprocessed"):
        st = stats.get(stage_name, {})
        lines.append(
            f"| **{stage_name.capitalize()}** | {int(st.get('pages', 0))} | "
            f"{int(st.get('words', 0)):,} | {int(st.get('lines', 0)):,} | "
            f"{int(st.get('chars', 0)):,} | {st.get('avg_words_per_page', 0)} | "
            f"{st.get('avg_lines_per_page', 0)} |"
        )

    lines.extend([
        "",
        "## Observations & Analysis",
        "",
        "- **Vision-Language Model Sensitivity to Binarization**: Modern VLMs (like PaddleOCR-VL-1.6) are trained extensively on raw RGB document images and grayscale book scans. Excessive binary thresholding (Otsu) can sometimes degrade subtle stroke textures or character antialiasing compared to raw or dewarped scans.",
        "- **Impact of Dewarping**: Page dewarping straightens curved text lines near the book gutter, which significantly improves line segmentation and multi-column parsing accuracy.",
        "",
        "## Sample Page Comparisons",
        "",
    ])

    for page_name in sample_pages:
        lines.append(f"### Sample Page: `{page_name}`\n")
        lines.append("| Raw (`pages`) | Dewarped (`dewarped_pages`) | Preprocessed (`preprocessed_pages`) |")
        lines.append("| :--- | :--- | :--- |")

        page_texts: Dict[str, str] = {}
        for stage_name, stage_dir in stages.items():
            f = stage_dir / f"{page_name}.md"
            if f.exists():
                snippet = f.read_text(encoding="utf-8").strip()
                # Preview first 15 lines of snippet, escape pipe
                preview_lines = snippet.splitlines()[:15]
                preview = "<br>".join([p.replace("|", "\\|") for p in preview_lines])
                page_texts[stage_name] = preview
            else:
                page_texts[stage_name] = "*Not found*"

        lines.append(
            f"| {page_texts.get('pages', '')} | "
            f"{page_texts.get('dewarped', '')} | "
            f"{page_texts.get('preprocessed', '')} |"
        )
        lines.append("\n")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nComparison report generated: {report_path}")
    return report_path


def run_pipeline(
    input_dir: str | Path = "input/godeatgod",
    output_dir: str | Path = "output",
    book_name: str = "godeatgod",
    start_page: int = 179,
    stage: str = "all",
    task_prompt: str = "OCR:",
    max_new_tokens: int = 2048,
    overwrite: bool = False,
    prepare_images: bool = True,
) -> None:
    """Run full image preparation and PaddleOCR-VL-1.6 across page variants."""
    base_out = Path(output_dir)
    inp_path = Path(input_dir)

    pages_dir = base_out / "pages"
    dewarped_dir = base_out / "dewarped_pages"
    preprocessed_dir = base_out / "preprocessed_pages"

    stage_configs = {
        "pages": {
            "input_dir": pages_dir,
            "output_dir": base_out / "ocr_pages",
            "combined_file": f"{book_name}_pages.md",
        },
        "dewarped": {
            "input_dir": dewarped_dir,
            "output_dir": base_out / "ocr_dewarped",
            "combined_file": f"{book_name}_dewarped.md",
        },
        "preprocessed": {
            "input_dir": preprocessed_dir,
            "output_dir": base_out / "ocr_preprocessed",
            "combined_file": f"{book_name}_preprocessed.md",
        },
    }

    # Step 1: Image splitting / A5 normalization if needed
    if prepare_images and inp_path.exists():
        from jpg2txt.image_splitter import ImageSplitter
        from jpg2txt.dewarper import PageDewarper
        from jpg2txt.preprocessor import ImagePreprocessor

        # Check if pages need generation
        pages_exist = pages_dir.exists() and any(pages_dir.glob("page_*.png"))
        if not pages_exist or overwrite:
            print(f"\n[Stage 1/3] Normalizing input images from {inp_path} -> {pages_dir}...")
            splitter = ImageSplitter(inp_path, pages_dir)
            splitter.process_all_images(save_pages=True, start_page=start_page)
            print(f"Normalized {len(list(pages_dir.glob('page_*.png')))} pages.")

        # Check if dewarping needed
        dewarped_exist = dewarped_dir.exists() and any(dewarped_dir.glob("page_*.png"))
        if not dewarped_exist or overwrite:
            print(f"\n[Stage 2/3] Dewarping pages from {pages_dir} -> {dewarped_dir}...")
            dewarper = PageDewarper(dewarped_dir)
            dewarper.dewarp_folder(pages_dir, force=overwrite)
            print(f"Dewarped {len(list(dewarped_dir.glob('page_*.png')))} pages.")

        # Check if preprocessing needed
        preprocessed_exist = preprocessed_dir.exists() and any(preprocessed_dir.glob("page_*.png"))
        if not preprocessed_exist or overwrite:
            print(f"\n[Stage 3/3] Preprocessing pages from {dewarped_dir} -> {preprocessed_dir}...")
            preprocessor = ImagePreprocessor(preprocessed_dir)
            preprocessor.process_folder(dewarped_dir, force=overwrite)
            print(f"Preprocessed {len(list(preprocessed_dir.glob('page_*.png')))} pages.")

    selected_stages = list(stage_configs.keys()) if stage == "all" else [stage]

    # Validate input directories exist
    for s in selected_stages:
        inp = stage_configs[s]["input_dir"]
        if not inp.exists():
            raise FileNotFoundError(f"Input directory for stage '{s}' does not exist: {inp}")

    print("==================================================")
    print("PaddleOCR-VL-1.6 Multi-Stage Processing Pipeline")
    print(f"Book Identifier:       {book_name}")
    print(f"Output Base Directory: {base_out.resolve()}")
    print(f"Selected Stage(s):     {selected_stages}")
    print(f"Task Prompt:           {task_prompt}")
    print(f"Max New Tokens:        {max_new_tokens}")
    print(f"Overwrite Existing:    {overwrite}")
    print("==================================================")

    # Initialize PaddleOCR engine once to reuse model in GPU VRAM
    engine = PaddleOCREngine()

    processed_stage_dirs: Dict[str, Path] = {}

    for s in selected_stages:
        cfg = stage_configs[s]
        stage_inp_dir = cfg["input_dir"]
        stage_out_dir = cfg["output_dir"]
        comb_file = cfg["combined_file"]

        image_files = sorted(
            [f for f in stage_inp_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        print(f"\nRunning OCR on stage '{s}' ({len(image_files)} images from {stage_inp_dir})...")

        with tqdm(total=len(image_files), desc=f"OCR ({s})") as pbar:
            def pbar_cb(curr, total, name):
                pbar.set_postfix_str(name)
                pbar.update(1)

            engine.process_image_folder(
                folder_path=stage_inp_dir,
                output_dir=stage_out_dir,
                combined_filename=comb_file,
                task_prompt=task_prompt,
                max_new_tokens=max_new_tokens,
                overwrite=overwrite,
                progress_callback=pbar_cb,
            )

        comb_path = stage_out_dir / comb_file
        if comb_path.exists():
            # Also copy combined to base_out for convenience
            (base_out / comb_file).write_text(comb_path.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"Completed stage '{s}': Individual .md and combined '{comb_file}' saved in {stage_out_dir}")
        processed_stage_dirs[s] = stage_out_dir

    # If all stages processed (or already available), generate comparison report
    all_stages = {s: stage_configs[s]["output_dir"] for s in stage_configs}
    if all(d.exists() and any(d.glob("page_*.md")) for d in all_stages.values()):
        generate_comparison_report(base_out, all_stages)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-1.6 OCR pipeline with comparison across processing stages",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default="input/godeatgod",
        help="Input directory with scanned images (default: input/godeatgod)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Base directory containing output folders (default: output)",
    )
    parser.add_argument(
        "--book",
        "-b",
        default="godeatgod",
        help="Book title identifier for filenames (default: godeatgod)",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=179,
        help="Starting page number for sequential naming (default: 179)",
    )
    parser.add_argument(
        "--stage",
        "-s",
        choices=["all", "pages", "dewarped", "preprocessed"],
        default="all",
        help="Processing stage to run OCR on (default: all)",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip image splitting/dewarping/preprocessing preparation",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="OCR:",
        help="Task prompt for PaddleOCR-VL (default: 'OCR:')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum generated tokens per page (default: 2048)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite previously generated images and markdown files",
    )

    args = parser.parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        book_name=args.book,
        start_page=args.start_page,
        stage=args.stage,
        task_prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        overwrite=args.force,
        prepare_images=not args.skip_prep,
    )


if __name__ == "__main__":
    main()

