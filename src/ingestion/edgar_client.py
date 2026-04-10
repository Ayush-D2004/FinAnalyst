"""
edgar_client.py
──────────────
Thin wrapper around sec-edgar-downloader that fetches raw 10-K / 10-Q
HTML filings from the SEC EDGAR API and returns the local file path.

sec-edgar-downloader v5+ changes:
  - Files are saved under  <download_folder>/sec-edgar-filings/<TICKER>/<form>/
  - Each accession dir contains only `full-submission.txt` (SGML bundle),
    not a standalone .htm file.  We extract the primary HTML document from
    that bundle and cache it as `primary-document.htm`.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Optional

from sec_edgar_downloader import Downloader

from src import config


class EdgarClient:
    """Download SEC filings on demand and cache them locally."""

    def __init__(self, cache_dir: Path = config.EDGAR_CACHE_DIR):
        self.cache_dir = cache_dir
        # The Downloader requires a company name and email as per SEC fair-use policy
        self.downloader = Downloader(
            company_name="FinAnalyst-RAG",
            email_address="user@finanalyst.local",
            download_folder=str(cache_dir),
        )
        # v5+ saves under <cache_dir>/sec-edgar-filings/
        self._filings_root = cache_dir / "sec-edgar-filings"

    def get_10k_path(
        self,
        ticker: str,
        year: int,
        force_download: bool = False,
    ) -> Optional[Path]:
        """
        Returns the local path to the primary HTML document of the 10-K filing.
        Downloads the filing first if it is not already cached.
        """
        ticker = ticker.upper().strip()
        ticker_dir = self._filings_root / ticker / "10-K"

        if not force_download and ticker_dir.exists():
            match = self._find_htm_in_dir(ticker_dir, year)
            if match:
                return match

        # ── Download ──────────────────────────────────────────────────────────
        print(f"Downloading {ticker} 10-K ({year}) from SEC EDGAR …")
        try:
            self.downloader.get(
                "10-K",
                ticker,
                limit=1,
                after=f"{year - 1}-12-31",
                before=f"{year + 1}-01-01",
            )
        except Exception as exc:
            print(f"[EdgarClient] Download failed for {ticker} {year}: {exc}")
            return None

        return self._find_htm_in_dir(ticker_dir, year)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_htm_in_dir(self, ticker_dir: Path, year: int) -> Optional[Path]:
        """
        Scans the cached directory tree for the primary HTML document.
        Handles both standalone .htm files and SGML full-submission.txt bundles.
        """
        if not ticker_dir.exists():
            return None

        # 1) Look for already-extracted standalone HTM files
        patterns = [
            str(ticker_dir / "**" / "primary-document.htm"),
            str(ticker_dir / "**" / "*.htm"),
        ]
        for pattern in patterns:
            files = sorted(glob.glob(pattern, recursive=True))
            if files:
                year_matches = [f for f in files if str(year) in f]
                return Path(year_matches[0] if year_matches else files[-1])

        # 2) Extract HTML from full-submission.txt bundles (v5+ format)
        bundles = sorted(glob.glob(
            str(ticker_dir / "**" / "full-submission.txt"), recursive=True
        ))
        for bundle_path in bundles:
            extracted = self._extract_htm_from_bundle(Path(bundle_path))
            if extracted:
                return extracted

        return None

    def _extract_htm_from_bundle(self, bundle_path: Path) -> Optional[Path]:
        """
        Extracts the first <DOCUMENT> of type 10-K from an SGML bundle and
        writes it as `primary-document.htm` in the same directory.
        Returns the path if successful, else None.
        """
        out_path = bundle_path.parent / "primary-document.htm"
        if out_path.exists():
            return out_path

        try:
            content = bundle_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"[EdgarClient] Could not read bundle {bundle_path}: {exc}")
            return None

        # Find the first DOCUMENT block that is actually the 10-K HTML
        doc_pattern = re.compile(
            r"<DOCUMENT>\s*<TYPE>10-K.*?<TEXT>(.*?)</TEXT>",
            re.DOTALL | re.IGNORECASE,
        )
        match = doc_pattern.search(content)
        if not match:
            # Fallback: grab the first <TEXT> block (likely the primary doc)
            text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL | re.IGNORECASE)
            match = text_pattern.search(content)

        if not match:
            print(f"[EdgarClient] No text block found in bundle {bundle_path}")
            return None

        html_content = match.group(1).strip()
        try:
            out_path.write_text(html_content, encoding="utf-8")
            print(f"[EdgarClient] Extracted HTML -> {out_path}")
            return out_path
        except Exception as exc:
            print(f"[EdgarClient] Failed to write extracted HTML: {exc}")
            return None
