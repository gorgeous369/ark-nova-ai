#!/usr/bin/env python3
"""Download all Ark Nova card JPGs listed by arknova.cards frontend data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable, List, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://arknova.cards"
APP_JS_URL = f"{BASE_URL}/dist/app.js"
USER_AGENT = "ark-nova-ai/0.1 (+local downloader)"

CARD_RE = re.compile(r'"id":(\d+),"category":"(animal|sponsor)"')


@dataclass(frozen=True)
class CardImageRef:
    card_id: int
    category: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/dist/images/{self.category}/{self.card_id}.jpg"

    @property
    def relative_path(self) -> Path:
        return Path(self.category) / f"{self.card_id}.jpg"


def fetch_text(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "ignore")


def fetch_bytes(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def extract_cards_from_app_js(app_js: str) -> List[CardImageRef]:
    seen: Set[Tuple[int, str]] = set()
    refs: List[CardImageRef] = []
    for card_id_text, category in CARD_RE.findall(app_js):
        key = (int(card_id_text), category)
        if key in seen:
            continue
        seen.add(key)
        refs.append(CardImageRef(card_id=key[0], category=key[1]))
    refs.sort(key=lambda item: (item.category, item.card_id))
    return refs


def _download_one(
    ref: CardImageRef,
    output_dir: Path,
    timeout: float,
    overwrite: bool,
) -> str:
    target = output_dir / ref.relative_path
    if target.exists() and not overwrite:
        return "skipped"

    target.parent.mkdir(parents=True, exist_ok=True)
    blob = fetch_bytes(ref.url, timeout=timeout)
    target.write_bytes(blob)
    return "downloaded"


def download_all(
    refs: Iterable[CardImageRef],
    output_dir: Path,
    timeout: float,
    workers: int,
    overwrite: bool,
) -> Tuple[int, int, int]:
    downloaded = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, ref, output_dir, timeout, overwrite): ref
            for ref in refs
        }
        for future in as_completed(futures):
            ref = futures[future]
            try:
                status = future.result()
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                failed += 1
                print(f"[FAIL] {ref.url} -> {exc}", file=sys.stderr)
                continue
            if status == "downloaded":
                downloaded += 1
            else:
                skipped += 1

    return downloaded, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all card JPGs from arknova.cards")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cards/images/arknova_cards"),
        help="Directory where JPG files are stored (default: data/cards/images/arknova_cards)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel download workers (default: 16)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Network timeout in seconds (default: 20)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")

    print(f"Fetching index: {APP_JS_URL}")
    app_js = fetch_text(APP_JS_URL, timeout=args.timeout)
    refs = extract_cards_from_app_js(app_js)
    if not refs:
        raise RuntimeError("No card IDs found in app.js")

    print(f"Found {len(refs)} card image references")
    downloaded, skipped, failed = download_all(
        refs=refs,
        output_dir=args.output_dir,
        timeout=args.timeout,
        workers=args.workers,
        overwrite=args.overwrite,
    )

    print(f"Output dir : {args.output_dir}")
    print(f"Downloaded : {downloaded}")
    print(f"Skipped    : {skipped}")
    print(f"Failed     : {failed}")

    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
