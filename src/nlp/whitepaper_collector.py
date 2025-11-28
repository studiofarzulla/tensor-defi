"""
Whitepaper Collector

Handles acquisition and text extraction from cryptocurrency whitepapers.
Supports PDF files and web-based documents.

For PoC: Manual curation of top 20 project whitepapers.
Future: Integration with Golden.com API or web scraping.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from datetime import datetime

# PyMuPDF for PDF extraction
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Run: pip install PyMuPDF")


@dataclass
class WhitepaperMetadata:
    """Metadata for a collected whitepaper."""
    project_symbol: str
    project_name: str
    source_path: str
    source_type: str  # 'pdf', 'url', 'text'
    collection_date: datetime = field(default_factory=datetime.now)
    version: Optional[str] = None
    publication_date: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: str = "en"
    extra: Dict = field(default_factory=dict)


@dataclass
class CollectedWhitepaper:
    """A collected whitepaper with extracted text."""
    metadata: WhitepaperMetadata
    raw_text: str
    pages: List[str] = field(default_factory=list)  # Text per page
    extraction_quality: float = 1.0  # 0-1 quality estimate
    
    @property
    def is_valid(self) -> bool:
        """Check if extraction produced valid content."""
        return len(self.raw_text) > 500  # Minimum viable whitepaper


class WhitepaperCollector:
    """Collect and extract text from cryptocurrency whitepapers."""
    
    # Known whitepaper URLs for manual collection
    WHITEPAPER_SOURCES = {
        "BTC": "https://bitcoin.org/bitcoin.pdf",
        "ETH": "https://ethereum.org/whitepaper",
        "SOL": "https://solana.com/solana-whitepaper.pdf",
        "ADA": "https://why.cardano.org/",
        "AVAX": "https://www.avalabs.org/whitepapers",
        "DOT": "https://polkadot.network/whitepaper/",
        "LINK": "https://chain.link/whitepaper",
        "ATOM": "https://cosmos.network/whitepaper",
        "ALGO": "https://www.algorand.com/technology/white-papers",
        "FIL": "https://filecoin.io/filecoin.pdf",
        "XMR": "https://www.getmonero.org/resources/research-lab/",
        "UNI": "https://uniswap.org/whitepaper.pdf",
        "NEAR": "https://near.org/papers/the-official-near-white-paper/",
    }
    
    def __init__(self, data_dir: str = "data/whitepapers"):
        """
        Initialize collector.
        
        Args:
            data_dir: Directory to store downloaded whitepapers
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_PYMUPDF:
            print("Warning: PDF extraction disabled without PyMuPDF")
    
    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> CollectedWhitepaper:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            CollectedWhitepaper with extracted text
        """
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF required for PDF extraction. Run: pip install PyMuPDF")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Infer project info from filename
        filename = pdf_path.stem.lower()
        project_symbol = self._infer_symbol(filename)
        project_name = self._infer_name(filename)
        
        # Extract text from PDF
        pages = []
        full_text = ""
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                pages.append(page_text)
                full_text += page_text + "\n\n"
            
            page_count = len(doc)
            doc.close()
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract PDF: {e}")
        
        # Estimate extraction quality
        quality = self._estimate_quality(full_text)
        
        # Create metadata
        metadata = WhitepaperMetadata(
            project_symbol=project_symbol,
            project_name=project_name,
            source_path=str(pdf_path),
            source_type="pdf",
            page_count=page_count,
            word_count=len(full_text.split()),
        )
        
        return CollectedWhitepaper(
            metadata=metadata,
            raw_text=full_text,
            pages=pages,
            extraction_quality=quality,
        )
    
    def extract_from_text(
        self,
        text: str,
        project_symbol: str,
        project_name: str,
    ) -> CollectedWhitepaper:
        """
        Create whitepaper from raw text (for testing or manual input).
        
        Args:
            text: Raw whitepaper text
            project_symbol: Crypto ticker
            project_name: Full project name
        
        Returns:
            CollectedWhitepaper
        """
        metadata = WhitepaperMetadata(
            project_symbol=project_symbol,
            project_name=project_name,
            source_path="manual_input",
            source_type="text",
            word_count=len(text.split()),
        )
        
        return CollectedWhitepaper(
            metadata=metadata,
            raw_text=text,
            pages=[text],
            extraction_quality=1.0,
        )
    
    def collect_from_directory(
        self,
        directory: Optional[Union[str, Path]] = None
    ) -> List[CollectedWhitepaper]:
        """
        Collect all whitepapers from a directory.
        
        Args:
            directory: Directory containing PDF files (default: self.data_dir)
        
        Returns:
            List of CollectedWhitepaper objects
        """
        directory = Path(directory) if directory else self.data_dir
        
        whitepapers = []
        pdf_files = list(directory.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_path in pdf_files:
            try:
                wp = self.extract_from_pdf(pdf_path)
                if wp.is_valid:
                    whitepapers.append(wp)
                    print(f"  ✓ {wp.metadata.project_symbol}: {wp.metadata.word_count} words")
                else:
                    print(f"  ✗ {pdf_path.name}: Extraction failed (too short)")
            except Exception as e:
                print(f"  ✗ {pdf_path.name}: {e}")
        
        return whitepapers
    
    def save_extracted_text(
        self,
        whitepaper: CollectedWhitepaper,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save extracted text to file for caching.
        
        Args:
            whitepaper: CollectedWhitepaper to save
            output_dir: Output directory (default: self.data_dir/extracted)
        
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir) if output_dir else self.data_dir / "extracted"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{whitepaper.metadata.project_symbol.lower()}_whitepaper.txt"
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write metadata header
            f.write(f"# {whitepaper.metadata.project_name} ({whitepaper.metadata.project_symbol})\n")
            f.write(f"# Source: {whitepaper.metadata.source_path}\n")
            f.write(f"# Extracted: {whitepaper.metadata.collection_date.isoformat()}\n")
            f.write(f"# Words: {whitepaper.metadata.word_count}\n")
            f.write(f"# Quality: {whitepaper.extraction_quality:.2f}\n")
            f.write("#" * 60 + "\n\n")
            f.write(whitepaper.raw_text)
        
        return output_path
    
    def load_extracted_text(
        self,
        project_symbol: str,
        extracted_dir: Optional[Union[str, Path]] = None
    ) -> Optional[CollectedWhitepaper]:
        """
        Load previously extracted text from cache.
        
        Args:
            project_symbol: Crypto ticker to load
            extracted_dir: Directory with extracted files
        
        Returns:
            CollectedWhitepaper or None if not found
        """
        extracted_dir = Path(extracted_dir) if extracted_dir else self.data_dir / "extracted"
        filename = f"{project_symbol.lower()}_whitepaper.txt"
        filepath = extracted_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse metadata from header
        lines = content.split('\n')
        header_lines = []
        text_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('#'):
                header_lines.append(line)
            else:
                text_start = i
                break
        
        raw_text = '\n'.join(lines[text_start:]).strip()
        
        # Extract metadata from header
        project_name = project_symbol  # Default
        source_path = "cached"
        word_count = len(raw_text.split())
        
        for line in header_lines:
            if line.startswith('# ') and '(' in line:
                # First line: "# Bitcoin (BTC)"
                match = re.match(r'# (.+) \((\w+)\)', line)
                if match:
                    project_name = match.group(1)
            elif 'Source:' in line:
                source_path = line.split('Source:')[1].strip()
        
        metadata = WhitepaperMetadata(
            project_symbol=project_symbol,
            project_name=project_name,
            source_path=source_path,
            source_type="cached",
            word_count=word_count,
        )
        
        return CollectedWhitepaper(
            metadata=metadata,
            raw_text=raw_text,
            pages=[raw_text],
            extraction_quality=1.0,
        )
    
    def _infer_symbol(self, filename: str) -> str:
        """Infer project symbol from filename."""
        filename = filename.lower()
        
        # Known mappings
        symbol_map = {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "solana": "SOL",
            "cardano": "ADA",
            "avalanche": "AVAX",
            "polkadot": "DOT",
            "chainlink": "LINK",
            "cosmos": "ATOM",
            "algorand": "ALGO",
            "filecoin": "FIL",
            "monero": "XMR",
            "uniswap": "UNI",
            "near": "NEAR",
            "litecoin": "LTC",
            "ripple": "XRP",
            "tron": "TRX",
            "polygon": "MATIC",
            "bnb": "BNB",
            "stellar": "XLM",
            "dogecoin": "DOGE",
        }
        
        for name, symbol in symbol_map.items():
            if name in filename:
                return symbol
        
        # Try to extract from filename directly
        match = re.search(r'\b([A-Z]{2,5})\b', filename.upper())
        if match:
            return match.group(1)
        
        return "UNKNOWN"
    
    def _infer_name(self, filename: str) -> str:
        """Infer project name from filename."""
        filename = filename.lower()
        
        name_map = {
            "bitcoin": "Bitcoin",
            "ethereum": "Ethereum",
            "solana": "Solana",
            "cardano": "Cardano",
            "avalanche": "Avalanche",
            "polkadot": "Polkadot",
            "chainlink": "Chainlink",
            "cosmos": "Cosmos",
            "algorand": "Algorand",
            "filecoin": "Filecoin",
            "monero": "Monero",
            "uniswap": "Uniswap",
            "near": "NEAR Protocol",
            "litecoin": "Litecoin",
            "ripple": "Ripple",
            "tron": "TRON",
            "polygon": "Polygon",
            "bnb": "BNB Chain",
            "stellar": "Stellar",
            "dogecoin": "Dogecoin",
        }
        
        for key, name in name_map.items():
            if key in filename:
                return name
        
        # Capitalize filename as fallback
        return filename.replace('_', ' ').replace('-', ' ').title()
    
    def _estimate_quality(self, text: str) -> float:
        """
        Estimate extraction quality based on text characteristics.
        
        Returns value between 0 and 1.
        """
        if not text:
            return 0.0
        
        # Check for common extraction issues
        issues = 0
        
        # Too many non-ASCII characters (garbled text)
        non_ascii_ratio = len(re.findall(r'[^\x00-\x7F]', text)) / len(text)
        if non_ascii_ratio > 0.1:
            issues += 1
        
        # Too many consecutive whitespace (layout extraction)
        if re.search(r'\s{10,}', text):
            issues += 1
        
        # Very short average word length (broken words)
        words = text.split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < 3:
                issues += 1
        
        # No sentence structure (missing periods)
        sentence_count = len(re.findall(r'[.!?]', text))
        if sentence_count < len(words) / 50:
            issues += 1
        
        # Calculate quality score
        quality = max(0.0, 1.0 - (issues * 0.25))
        return quality


if __name__ == "__main__":
    print("=== Whitepaper Collector Test ===\n")
    
    # Test with sample text
    collector = WhitepaperCollector()
    
    sample_text = """
    Bitcoin: A Peer-to-Peer Electronic Cash System
    
    Abstract. A purely peer-to-peer version of electronic cash would allow online
    payments to be sent directly from one party to another without going through a
    financial institution. Digital signatures provide part of the solution, but the main
    benefits are lost if a trusted third party is still required to prevent double-spending.
    We propose a solution to the double-spending problem using a peer-to-peer network.
    
    1. Introduction
    
    Commerce on the Internet has come to rely almost exclusively on financial institutions
    serving as trusted third parties to process electronic payments. While the system works
    well enough for most transactions, it still suffers from the inherent weaknesses of the
    trust based model.
    """
    
    wp = collector.extract_from_text(sample_text, "BTC", "Bitcoin")
    
    print(f"Project: {wp.metadata.project_name} ({wp.metadata.project_symbol})")
    print(f"Source: {wp.metadata.source_type}")
    print(f"Words: {wp.metadata.word_count}")
    print(f"Valid: {wp.is_valid}")
    print(f"Quality: {wp.extraction_quality:.2f}")
    
    # Save and reload test
    print("\n=== Save/Load Test ===")
    saved_path = collector.save_extracted_text(wp)
    print(f"Saved to: {saved_path}")
    
    loaded = collector.load_extracted_text("BTC")
    if loaded:
        print(f"Loaded: {loaded.metadata.project_name}")
        print(f"Words match: {loaded.metadata.word_count == wp.metadata.word_count}")
