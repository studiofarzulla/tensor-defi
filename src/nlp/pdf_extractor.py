#!/usr/bin/env python3
"""
PDF Text Extraction for TENSOR-DEFI

Extracts clean text from cryptocurrency whitepapers using PyMuPDF.
Handles multi-column layouts, removes boilerplate, chunks for NLP.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedDocument:
    """Extracted and processed document."""
    symbol: str
    source_path: str
    raw_text: str
    clean_text: str
    chunks: list[str]
    page_count: int
    word_count: int


class PDFExtractor:
    """Extracts and preprocesses text from PDF whitepapers."""

    # Boilerplate patterns to remove
    BOILERPLATE_PATTERNS = [
        r'^\s*table of contents\s*$',
        r'^\s*\d+\s*$',  # Page numbers
        r'^\s*(copyright|©|all rights reserved).*$',
        r'^\s*references?\s*$',
        r'^\s*appendix\s*[a-z]?\s*$',
        r'^\s*acknowledgements?\s*$',
        r'^\s*disclaimer\s*$',
        r'https?://[^\s]+',  # URLs
    ]

    # Minimum chunk size (chars)
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 1000

    def __init__(self, whitepaper_dir: Path, output_dir: Path):
        self.whitepaper_dir = Path(whitepaper_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract raw text from PDF."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []

            for page in doc:
                text = page.get_text("text")
                text_parts.append(text)

            doc.close()
            return "\n".join(text_parts)

        except Exception as e:
            logger.error(f"PDF extraction failed: {pdf_path} - {e}")
            return None

    def extract_markdown(self, md_path: Path) -> Optional[str]:
        """Extract text from markdown fallback."""
        try:
            return md_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Markdown extraction failed: {md_path} - {e}")
            return None

    def extract_html(self, html_path: Path) -> Optional[str]:
        """Extract text from HTML, stripping tags."""
        try:
            html = html_path.read_text(encoding='utf-8')
            # Strip HTML tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            # Decode common entities
            text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
            text = text.replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&quot;', '"').replace('&#39;', "'")
            return text
        except Exception as e:
            logger.error(f"HTML extraction failed: {html_path} - {e}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove boilerplate
        lines = text.split('\n')
        clean_lines = []

        for line in lines:
            line_lower = line.lower().strip()
            is_boilerplate = any(
                re.match(pattern, line_lower, re.IGNORECASE)
                for pattern in self.BOILERPLATE_PATTERNS
            )
            if not is_boilerplate and len(line.strip()) > 3:
                clean_lines.append(line.strip())

        text = ' '.join(clean_lines)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)

        return text.strip()

    def chunk_text(self, text: str) -> list[str]:
        """Split text into claim-sized chunks for NLP."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_length + len(sentence) > self.MAX_CHUNK_SIZE and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.MIN_CHUNK_SIZE:
                    chunks.append(chunk_text)
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1

        # Final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.MIN_CHUNK_SIZE:
                chunks.append(chunk_text)

        return chunks

    def process_document(self, symbol: str, path: Path) -> Optional[ExtractedDocument]:
        """Process a single document (PDF, HTML, or markdown)."""
        # Extract raw text based on file type
        suffix = path.suffix.lower()
        if suffix == '.pdf':
            raw_text = self.extract_pdf(path)
            page_count = len(fitz.open(path)) if raw_text else 0
        elif suffix == '.html':
            raw_text = self.extract_html(path)
            page_count = 1
        else:
            raw_text = self.extract_markdown(path)
            page_count = 1

        if not raw_text:
            return None

        # Clean and chunk
        clean_text = self.clean_text(raw_text)
        chunks = self.chunk_text(clean_text)

        doc = ExtractedDocument(
            symbol=symbol,
            source_path=str(path),
            raw_text=raw_text,
            clean_text=clean_text,
            chunks=chunks,
            page_count=page_count,
            word_count=len(clean_text.split())
        )

        logger.info(f"{symbol}: {doc.word_count} words, {len(chunks)} chunks")
        return doc

    def process_all(self, metadata_path: Path) -> list[ExtractedDocument]:
        """Process all whitepapers using metadata."""
        with open(metadata_path) as f:
            metadata = json.load(f)

        documents = []
        for entry in metadata:
            path = Path(entry['local_path'])
            if not path.exists():
                logger.warning(f"{entry['symbol']}: File not found")
                continue

            doc = self.process_document(entry['symbol'], path)
            if doc:
                documents.append(doc)

                # Update page count in metadata
                entry['pages'] = doc.page_count

        # Save processed chunks
        self._save_chunks(documents)

        # Update metadata with page counts
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return documents

    def _save_chunks(self, documents: list[ExtractedDocument]):
        """Save extracted chunks for NLP processing."""
        output = {
            doc.symbol: {
                'chunks': doc.chunks,
                'word_count': doc.word_count,
                'page_count': doc.page_count
            }
            for doc in documents
        }

        output_path = self.output_dir / "extracted_chunks.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved chunks: {output_path}")

        # Summary
        total_chunks = sum(len(doc.chunks) for doc in documents)
        total_words = sum(doc.word_count for doc in documents)

        print(f"\n{'='*50}")
        print(f"TEXT EXTRACTION SUMMARY")
        print(f"{'='*50}")
        print(f"Documents processed: {len(documents)}")
        print(f"Total chunks:        {total_chunks}")
        print(f"Total words:         {total_words:,}")
        print(f"Avg chunks/doc:      {total_chunks/len(documents):.1f}")
        print(f"{'='*50}")


def main():
    """Run text extraction."""
    base_path = Path(__file__).parent.parent.parent
    extractor = PDFExtractor(
        whitepaper_dir=base_path / "data" / "whitepapers",
        output_dir=base_path / "outputs" / "nlp"
    )
    metadata_path = base_path / "data" / "metadata" / "whitepaper_metadata.json"

    if metadata_path.exists():
        extractor.process_all(metadata_path)
    else:
        logger.error("Run whitepaper_collector.py first")


if __name__ == "__main__":
    main()
