"""
chunker.py
──────────
Section-aware, token-bounded chunker that preserves the
SEC Item hierarchy produced by the EdgarParser.

Each chunk is self-contained:
  "Section: Item 1A. Risk Factors\nContext: <chunk text>"

The section prefix ensures the bi-encoder sees structural metadata
during embedding, even when chunks are retrieved out-of-order.
"""
from __future__ import annotations

import uuid
from typing import List, Dict

from transformers import AutoTokenizer

from src import config


class SectionAwareChunker:
    """
    Convert a list of ParsedSection dicts (from EdgarParser) into
    a flat list of chunk dicts ready for SQLiteStore + embedding.
    """

    def __init__(
        self,
        tokenizer_name: str = config.BASE_ENCODER_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.CHUNK_OVERLAP,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_document(
        self,
        doc_id: str,
        parsed_sections: List[Dict[str, str]],
    ) -> List[Dict]:
        """
        Args:
            doc_id:           UUID of the parent document row in SQLite.
            parsed_sections:  Output of EdgarParser.parse().

        Returns:
            List of chunk dicts compatible with SQLiteStore.add_chunks().
        """
        chunks_out: List[Dict] = []

        for section in parsed_sections:
            section_name = section["section"]
            text = section["text"]

            # The embedded prefix anchors every chunk to its SEC Item.
            prefix = f"Section: {section_name}\nContext: "
            prefix_ids = self.tokenizer(
                prefix, add_special_tokens=False
            )["input_ids"]

            body_ids = self.tokenizer(
                text, add_special_tokens=False
            )["input_ids"]

            # How many body tokens fit after the prefix + safety buffer
            effective = self.chunk_size - len(prefix_ids) - 4
            if effective <= 0:
                continue   # pathological section title – skip

            start = 0
            while start < len(body_ids):
                end = start + effective
                slice_ids = body_ids[start:end]

                chunk_text = prefix + self.tokenizer.decode(
                    slice_ids, skip_special_tokens=True
                )

                chunks_out.append({
                    "chunk_id":    str(uuid.uuid4()),
                    "doc_id":      doc_id,
                    "section_name": section_name,
                    "chunk_text":  chunk_text,
                    "token_count": len(slice_ids) + len(prefix_ids),
                })

                if end >= len(body_ids):
                    break
                start += effective - self.overlap   # stride with overlap

        return chunks_out
