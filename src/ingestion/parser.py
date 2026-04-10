"""
parser.py
─────────
Production-grade SEC 10-K parser using alphanome-ai/sec-parser.

sec-parser builds a semantic tree from the raw EDGAR HTML, giving us
clean node-level access to text, headings, and section boundaries –
avoiding brittle regex heuristics entirely.

Fallback: Plain PDF / text files are handled by a simple PyMuPDF pass
when the file is not an EDGAR HTML document.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF – used only as a PDF fallback
import sec_parser as sp

from src import config

# ── Type alias ────────────────────────────────────────────────────────────────
ParsedSection = Dict[str, str]   # {"section": "...", "text": "..."}


class EdgarParser:
    """
    Parse SEC 10-K filings into a flat list of section dicts.
    Each dict maps {'section': '<SEC Item title>', 'text': '<raw text>'}.
    """

    # Normalised names we want to track; everything else is bucketed under
    # the nearest parent heading.
    _ITEM_RE = re.compile(
        r"^(item\s+\d+[a-z]?\.?\s+.*|part\s+[ivx]+\.?\s+.*)",
        re.IGNORECASE,
    )

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, file_path: str | Path) -> List[ParsedSection]:
        """Auto-detect format and dispatch to the correct parser."""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext in {".htm", ".html"}:
            return self._parse_edgar_html(path)
        elif ext == ".pdf":
            return self._parse_pdf(path)
        else:
            return self._parse_plaintext(path)

    # ── EDGAR HTML → sec-parser ───────────────────────────────────────────────

    def _parse_edgar_html(self, path: Path) -> List[ParsedSection]:
        html_text = path.read_text(encoding="utf-8", errors="replace")
        
        # sp.Edgar10QParser is used as a full-featured parser for all filings.
        # It returns a list of semantic elements (Title, Text, Table, etc.).
        elements = sp.Edgar10QParser().parse(html_text)

        sections: List[ParsedSection] = []
        current_title = "General"
        current_text_parts: List[str] = []

        for element in elements:
            # In sec-parser v0.58, we check titles by type. 
            # TopSectionTitle handles 'Item X' and TitleElement handles subheadings.
            node_text = element.text.strip() if hasattr(element, "text") else ""
            if not node_text:
                continue

            # Detect a new SEC Item heading or subheading
            if isinstance(element, (sp.TitleElement, sp.TopSectionTitle)):
                # Flush previous section
                if current_text_parts:
                    sections.append({
                        "section": current_title,
                        "text": "\n".join(current_text_parts),
                    })
                current_title = self._normalise_title(node_text)
                current_text_parts = []
            else:
                # Any non-title node contributes to the current section
                current_text_parts.append(node_text)

        # Flush the final section
        if current_text_parts:
            sections.append({
                "section": current_title,
                "text": "\n".join(current_text_parts),
            })

        # Remove empty / micro sections
        sections = [s for s in sections if len(s["text"].split()) > 20]
        return sections

    # ── PDF fallback (non-EDGAR uploads) ─────────────────────────────────────

    def _parse_pdf(self, path: Path) -> List[ParsedSection]:
        """
        Best-effort extraction from generic PDFs (user uploads).
        We still attempt to split on Item headings; otherwise the whole
        document becomes one 'General' section which the chunker handles.
        """
        sections: List[ParsedSection] = []
        current_title = "General"
        current_parts: List[str] = []

        doc = fitz.open(str(path))
        for page in doc:
            for block in page.get_text("blocks"):
                txt = block[4].strip()
                if not txt:
                    continue
                first_line = txt.split("\n")[0].strip()

                if len(first_line) < 120 and self._ITEM_RE.match(first_line):
                    if current_parts:
                        sections.append({"section": current_title,
                                         "text": "\n".join(current_parts)})
                    current_title = self._normalise_title(first_line)
                    current_parts = []
                else:
                    current_parts.append(txt)

        doc.close()
        if current_parts:
            sections.append({"section": current_title,
                              "text": "\n".join(current_parts)})

        sections = [s for s in sections if len(s["text"].split()) > 20]
        return sections if sections else [{"section": "General",
                                           "text": " ".join(
                                               p for s in sections
                                               for p in [s["text"]])}]

    # ── Plaintext fallback ────────────────────────────────────────────────────

    def _parse_plaintext(self, path: Path) -> List[ParsedSection]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return [{"section": "General", "text": content}]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_title(raw: str) -> str:
        """Trim and title-case the section label for consistent storage."""
        cleaned = re.sub(r"\s+", " ", raw).strip()
        return cleaned[:120]  # cap at 120 chars to avoid runaway headings
