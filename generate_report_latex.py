#!/usr/bin/env python3
"""
Generate the FYP dissertation report by compiling LaTeX to PDF.

Usage:
    python generate_report_latex.py

This script:
1. Copies all required images to report/images/
2. Runs pdflatex + bibtex + pdflatex x2 to compile the report
3. Outputs report/main.pdf
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PROJECT_DIR = Path(__file__).parent
REPORT_DIR = PROJECT_DIR / "report"
IMAGES_DIR = REPORT_DIR / "images"

# MiKTeX path (user-specific)
MIKTEX_BIN = Path(os.path.expanduser(
    "~/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
))

# Fallback: check PATH
PDFLATEX = shutil.which("pdflatex") or str(MIKTEX_BIN / "pdflatex.exe")
BIBTEX = shutil.which("bibtex") or str(MIKTEX_BIN / "bibtex.exe")

# Image sources
FEEDBACK_DIR = PROJECT_DIR / "professor feedback 260326"
PLOTS_DIR = PROJECT_DIR / "output" / "plots"


def copy_images():
    """Copy all required images to report/images/."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # COMSOL images
    for name in [
        "comsol_3d_unit_cell.jpeg",
        "comsol_topdown_unit_cell.jpeg",
        "comsol_sparam_on_state.jpeg",
        "comsol_sparam_off_state.jpeg",
    ]:
        src = FEEDBACK_DIR / name
        if src.exists():
            shutil.copy2(src, IMAGES_DIR / name)

    # Dashboard screenshots
    for f in FEEDBACK_DIR.glob("dashboard_*.png"):
        shutil.copy2(f, IMAGES_DIR / f.name)

    # Mermaid diagrams
    for f in PLOTS_DIR.glob("mermaid_*.png"):
        shutil.copy2(f, IMAGES_DIR / f.name)

    # Generated plots
    r2_plot = IMAGES_DIR / "ml_r2_comparison.png"
    if not r2_plot.exists():
        src = PLOTS_DIR / "ml_r2_comparison.png"
        if src.exists():
            shutil.copy2(src, r2_plot)

    print(f"Images copied to {IMAGES_DIR}")
    print(f"  Total: {len(list(IMAGES_DIR.iterdir()))} files")


def run_latex(cmd, label=""):
    """Run a LaTeX command and check for errors."""
    print(f"\n{'='*60}")
    print(f"Running: {label or ' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        cwd=str(REPORT_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"WARNING: {label} returned code {result.returncode}")
        # Print last 30 lines of output for debugging
        lines = result.stdout.split("\n")
        for line in lines[-30:]:
            if line.strip():
                print(f"  {line}")
        if result.stderr:
            print(f"STDERR: {result.stderr[-500:]}")
    else:
        print(f"  OK")

    return result.returncode


def compile_pdf():
    """Run the full LaTeX compilation pipeline."""
    tex_file = "main.tex"

    # First pass: generate .aux file
    rc = run_latex(
        [PDFLATEX, "-interaction=nonstopmode", tex_file],
        "pdflatex (pass 1)"
    )

    # BibTeX: process references
    run_latex(
        [BIBTEX, "main"],
        "bibtex"
    )

    # Second pass: incorporate references
    run_latex(
        [PDFLATEX, "-interaction=nonstopmode", tex_file],
        "pdflatex (pass 2)"
    )

    # Third pass: resolve cross-references
    run_latex(
        [PDFLATEX, "-interaction=nonstopmode", tex_file],
        "pdflatex (pass 3)"
    )

    pdf_path = REPORT_DIR / "main.pdf"
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"SUCCESS: {pdf_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"{'='*60}")
        return True
    else:
        print(f"\nERROR: PDF not generated at {pdf_path}")
        return False


def main():
    print("FYP Dissertation Report Generator")
    print(f"Project: {PROJECT_DIR}")
    print(f"Report:  {REPORT_DIR}")
    print(f"pdflatex: {PDFLATEX}")
    print(f"bibtex:   {BIBTEX}")

    # Check pdflatex exists
    if not Path(PDFLATEX).exists() and not shutil.which("pdflatex"):
        print("\nERROR: pdflatex not found. Install MiKTeX from https://miktex.org")
        sys.exit(1)

    # Step 1: Copy images
    copy_images()

    # Step 2: Compile PDF
    success = compile_pdf()

    if success:
        print("\nDone! Open report/main.pdf to view the dissertation.")
    else:
        print("\nCompilation failed. Check the log above for errors.")
        print("Tip: MiKTeX may prompt to install missing packages — click 'Install'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
