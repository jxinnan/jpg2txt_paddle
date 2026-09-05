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
    sample_pages: List[str] = ("page_108", "page_120", "page_140"),
) -> Path:
    """Generate a Markdown comparison report evaluating outputs from the three stages."""
    report_path = output_base / "comparison_report.md"
    stats: Dict[str, Dict[str, float]] = {}

    for stage_name, stage_dir in stages.items():
        stats[stage_name] = compute_stage_stats(stage_dir)

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
    output_dir: str | Path = "output",
    stage: str = "all",
    task_prompt: str = "OCR:",
    max_new_tokens: int = 2048,
    overwrite: bool = False,
) -> None:
    """Run PaddleOCR-VL-1.6 across one or all input page variants."""
    base_out = Path(output_dir)

    stage_configs = {
        "pages": {
            "input_dir": base_out / "pages",
            "output_dir": base_out / "ocr_pages",
            "combined_file": "undercover_pages.md",
        },
        "dewarped": {
            "input_dir": base_out / "dewarped_pages",
            "output_dir": base_out / "ocr_dewarped",
            "combined_file": "undercover_dewarped.md",
        },
        "preprocessed": {
            "input_dir": base_out / "preprocessed_pages",
            "output_dir": base_out / "ocr_preprocessed",
            "combined_file": "undercover_preprocessed.md",
        },
    }

    selected_stages = list(stage_configs.keys()) if stage == "all" else [stage]

    # Validate input directories exist
    for s in selected_stages:
        inp = stage_configs[s]["input_dir"]
        if not inp.exists():
            raise FileNotFoundError(f"Input directory for stage '{s}' does not exist: {inp}")

    print("==================================================")
    print("PaddleOCR-VL-1.6 Multi-Stage Processing Pipeline")
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
        inp_dir = cfg["input_dir"]
        out_dir = cfg["output_dir"]
        comb_file = cfg["combined_file"]

        image_files = sorted(
            [f for f in inp_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        print(f"\nProcessing stage '{s}' ({len(image_files)} images from {inp_dir})...")

        with tqdm(total=len(image_files), desc=f"OCR ({s})") as pbar:
            def pbar_cb(curr, total, name):
                pbar.set_postfix_str(name)
                pbar.update(1)

            engine.process_image_folder(
                folder_path=inp_dir,
                output_dir=out_dir,
                combined_filename=comb_file,
                task_prompt=task_prompt,
                max_new_tokens=max_new_tokens,
                overwrite=overwrite,
                progress_callback=pbar_cb,
            )

        print(f"Completed stage '{s}': Individual .md and combined '{comb_file}' saved in {out_dir}")
        processed_stage_dirs[s] = out_dir

    # If all stages processed (or already available), generate comparison report
    all_stages = {s: stage_configs[s]["output_dir"] for s in stage_configs}
    if all(d.exists() and any(d.glob("page_*.md")) for d in all_stages.values()):
        generate_comparison_report(base_out, all_stages)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-1.6 OCR pipeline with comparison across processing stages",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Base directory containing output folders (default: output)",
    )
    parser.add_argument(
        "--stage",
        "-s",
        choices=["all", "pages", "dewarped", "preprocessed"],
        default="all",
        help="Processing stage to run (default: all)",
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
        help="Overwrite previously generated markdown files",
    )

    args = parser.parse_args()
    run_pipeline(
        output_dir=args.output_dir,
        stage=args.stage,
        task_prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        overwrite=args.force,
    )


if __name__ == "__main__":
    main()

