#!/usr/bin/env python3
"""Extract text from the local Ark Nova rulebook PDF.

Usage:
    .venv/bin/python tools/extract_rulebook_text.py
"""

from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "ark-nova-rulebook.pdf"
OUTPUT_PATH = ROOT / "docs" / "ark_nova_rulebook_extracted.txt"


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Rulebook not found: {PDF_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(PDF_PATH))
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            f.write(f"\n\n===== PAGE {page_index} =====\n")
            f.write(text)

    print(f"Extracted {len(reader.pages)} pages -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
