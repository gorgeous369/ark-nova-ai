#!/usr/bin/env python3
"""Download and extract Ark Nova card images from ssimeonoff.github.io."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
from urllib.parse import urljoin
from urllib.request import Request, urlopen


PAGE_URL = "https://ssimeonoff.github.io/ark-nova"
USER_AGENT = "ark-nova-ai/0.1 (+sprite extractor)"

CARD_RE = re.compile(
    r'<li onclick="getClickedCard\(\);"\s+class="([^"]+)">\s*'
    r'<div class="number">#(\d+)</div>\s*'
    r'<div class="name">([^<]+)</div>',
    flags=re.S,
)
CSS_HREF_RE = re.compile(r'<link[^>]+href="([^"]*mystyles-ark\.css)"')
SHEET_RE = re.compile(
    r"\.cards(\d+)\s*\{\s*background-image:\s*url\([\"']?([^\"')]+)[\"']?\);\s*\}",
    flags=re.S,
)
COL_OFFSET_RE = re.compile(r"\.c(\d+)\s*\{\s*background-position-x:\s*(-?\d+)px;\s*\}", flags=re.S)
ROW_OFFSET_RE = re.compile(r"\.r(\d+)\s*\{\s*background-position-y:\s*(-?\d+)px;\s*\}", flags=re.S)
FILTER_DIV_RE = re.compile(
    r"\.filterDiv\s*\{.*?width:\s*(\d+)px;.*?height:\s*(\d+)px;",
    flags=re.S,
)
ROW_CLASS_RE = re.compile(r"r\d+")
COL_CLASS_RE = re.compile(r"c\d+")

CATEGORY_DIRS = {
    "animal": "animals",
    "sponsor": "sponsors",
    "project": "projects",
    "endgame": "endgame",
}


@dataclass(frozen=True)
class Layout:
    card_width: int
    card_height: int
    row_offsets: dict[str, int]
    col_offsets: dict[str, int]
    sheet_urls: dict[str, str]


@dataclass(frozen=True)
class CardCrop:
    category: str
    number: str
    name: str
    sheet: str
    row_class: str
    col_class: str
    offset_x: int
    offset_y: int

    @property
    def output_category(self) -> str:
        return CATEGORY_DIRS[self.category]

    @property
    def filename(self) -> str:
        return f"{self.number}_{slugify(self.name)}.jpg"

    @property
    def relative_path(self) -> Path:
        return Path(self.output_category) / self.filename


def fetch_text(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "ignore")


def fetch_bytes(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")
    return slug or "card"


def parse_layout(page_url: str, html_text: str, css_text: str) -> Layout:
    css_match = FILTER_DIV_RE.search(css_text)
    if not css_match:
        raise RuntimeError("Could not find .filterDiv width/height in CSS.")
    card_width = int(css_match.group(1))
    card_height = int(css_match.group(2))

    col_offsets = {"c1": 0}
    for col_index, offset_text in COL_OFFSET_RE.findall(css_text):
        col_offsets[f"c{col_index}"] = abs(int(offset_text))

    row_offsets = {"r1": 0}
    for row_index, offset_text in ROW_OFFSET_RE.findall(css_text):
        row_offsets[f"r{row_index}"] = abs(int(offset_text))

    sheet_urls: dict[str, str] = {}
    for sheet_index, relative_url in SHEET_RE.findall(css_text):
        sheet_urls[f"cards{sheet_index}"] = urljoin(page_url, relative_url)

    if not sheet_urls:
        raise RuntimeError("No sheet URLs found in CSS.")

    return Layout(
        card_width=card_width,
        card_height=card_height,
        row_offsets=row_offsets,
        col_offsets=col_offsets,
        sheet_urls=sheet_urls,
    )


def parse_css_url(page_url: str, html_text: str) -> str:
    match = CSS_HREF_RE.search(html_text)
    if not match:
        raise RuntimeError("Could not find mystyles-ark.css in HTML.")
    return urljoin(page_url, match.group(1))


def parse_card_crops(html_text: str, layout: Layout) -> list[CardCrop]:
    cards: list[CardCrop] = []
    for classes_raw, number, name in CARD_RE.findall(html_text):
        tokens = classes_raw.split()
        sheet = next((token for token in tokens if token.startswith("cards")), "")
        category = next((token for token in tokens if token in CATEGORY_DIRS), "")
        row_class = next((token for token in tokens if ROW_CLASS_RE.fullmatch(token)), "r1")
        col_class = next((token for token in tokens if COL_CLASS_RE.fullmatch(token)), "c1")

        if not sheet or not category:
            continue
        if sheet not in layout.sheet_urls:
            raise RuntimeError(f"Unknown sheet class: {sheet}")
        if row_class not in layout.row_offsets:
            raise RuntimeError(f"Unknown row class: {row_class}")
        if col_class not in layout.col_offsets:
            raise RuntimeError(f"Unknown column class: {col_class}")

        cards.append(
            CardCrop(
                category=category,
                number=number,
                name=name.strip(),
                sheet=sheet,
                row_class=row_class,
                col_class=col_class,
                offset_x=layout.col_offsets[col_class],
                offset_y=layout.row_offsets[row_class],
            )
        )

    if not cards:
        raise RuntimeError("No cards found in HTML.")
    return cards


def ensure_sheet(url: str, destination: Path, timeout: float, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(fetch_bytes(url, timeout=timeout))


def crop_card(
    sheet_path: Path,
    card: CardCrop,
    output_path: Path,
    card_width: int,
    card_height: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # `sips` falls back to centered crop when both offsets are zero.
    offset_y = card.offset_y
    offset_x = card.offset_x
    if offset_y == 0 and offset_x == 0:
        offset_y = 1

    command = [
        "sips",
        "-c",
        str(card_height),
        str(card_width),
        "--cropOffset",
        str(offset_y),
        str(offset_x),
        str(sheet_path),
        "--out",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def build_manifest(
    cards: list[CardCrop],
    output_dir: Path,
    sheet_dir: Path,
    layout: Layout,
) -> dict[str, object]:
    counts = Counter(card.output_category for card in cards)
    return {
        "source_page": PAGE_URL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "card_size": {
            "width": layout.card_width,
            "height": layout.card_height,
        },
        "counts": dict(sorted(counts.items())),
        "sheets": {
            sheet: {
                "url": url,
                "local_path": str((sheet_dir / f"{sheet}.jpg").relative_to(output_dir)),
            }
            for sheet, url in sorted(layout.sheet_urls.items())
        },
        "cards": [
            {
                "category": card.category,
                "output_category": card.output_category,
                "number": card.number,
                "name": card.name,
                "sheet": card.sheet,
                "row_class": card.row_class,
                "col_class": card.col_class,
                "offset_x": card.offset_x,
                "offset_y": card.offset_y,
                "local_path": str(card.relative_path),
            }
            for card in cards
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract Ark Nova card images from ssimeonoff.github.io."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cards/images/cards"),
        help="Directory where extracted card JPGs are stored (default: data/cards/images/cards)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download sheets and overwrite extracted card images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")

    print(f"Fetching page: {PAGE_URL}")
    html_text = fetch_text(PAGE_URL, timeout=args.timeout)
    css_url = parse_css_url(PAGE_URL, html_text)
    print(f"Fetching CSS : {css_url}")
    css_text = fetch_text(css_url, timeout=args.timeout)

    layout = parse_layout(page_url=PAGE_URL, html_text=html_text, css_text=css_text)
    cards = parse_card_crops(html_text=html_text, layout=layout)

    output_dir = args.output_dir
    sheet_dir = output_dir / "sheets"
    output_dir.mkdir(parents=True, exist_ok=True)

    for sheet, url in sorted(layout.sheet_urls.items()):
        destination = sheet_dir / f"{sheet}.jpg"
        ensure_sheet(url=url, destination=destination, timeout=args.timeout, overwrite=args.overwrite)

    for card in cards:
        crop_card(
            sheet_path=sheet_dir / f"{card.sheet}.jpg",
            card=card,
            output_path=output_dir / card.relative_path,
            card_width=layout.card_width,
            card_height=layout.card_height,
            overwrite=args.overwrite,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(build_manifest(cards=cards, output_dir=output_dir, sheet_dir=sheet_dir, layout=layout), indent=2)
        + "\n",
        encoding="utf-8",
    )

    counts = Counter(card.output_category for card in cards)
    print(f"Output dir : {output_dir}")
    print(f"Card size  : {layout.card_width}x{layout.card_height}")
    print(f"Total cards: {len(cards)}")
    for category in sorted(counts):
        print(f"{category:10s}: {counts[category]}")
    print(f"Manifest   : {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        raise
