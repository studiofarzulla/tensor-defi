"""
Text Processor for Crypto Whitepapers

Handles document chunking, cleaning, and section detection for
whitepaper analysis. Designed to work with semantic chunking
for optimal embedding generation.

"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class SectionType(Enum):
    """Common whitepaper section types."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    PROBLEM = "problem"
    SOLUTION = "solution"
    ARCHITECTURE = "architecture"
    TOKENOMICS = "tokenomics"
    GOVERNANCE = "governance"
    ROADMAP = "roadmap"
    TEAM = "team"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    OTHER = "other"


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    start_idx: int
    end_idx: int
    section_type: SectionType = SectionType.OTHER
    page_number: Optional[int] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class ProcessedDocument:
    """A fully processed whitepaper document."""
    project_symbol: str
    project_name: str
    raw_text: str
    chunks: List[TextChunk]
    sections: Dict[SectionType, List[TextChunk]]
    metadata: Dict = field(default_factory=dict)
    
    @property
    def total_words(self) -> int:
        return len(self.raw_text.split())
    
    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


class TextProcessor:
    """Process whitepaper text for NLP analysis."""
    
    # Section header patterns (case-insensitive)
    SECTION_PATTERNS = {
        SectionType.ABSTRACT: r'\b(abstract|executive\s+summary|overview)\b',
        SectionType.INTRODUCTION: r'\b(introduction|background|motivation)\b',
        SectionType.PROBLEM: r'\b(problem|challenge|issue|current\s+state)\b',
        SectionType.SOLUTION: r'\b(solution|approach|our\s+approach|proposal)\b',
        SectionType.ARCHITECTURE: r'\b(architecture|design|technical|protocol|consensus|implementation)\b',
        SectionType.TOKENOMICS: r'\b(token|economics|tokenomics|distribution|supply|emission)\b',
        SectionType.GOVERNANCE: r'\b(governance|voting|dao|community)\b',
        SectionType.ROADMAP: r'\b(roadmap|timeline|milestones|future)\b',
        SectionType.TEAM: r'\b(team|founders|advisors|contributors)\b',
        SectionType.CONCLUSION: r'\b(conclusion|summary|closing)\b',
        SectionType.REFERENCES: r'\b(references|bibliography|citations)\b',
    }
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        """
        Initialize text processor.
        
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text from PDF extraction.
        
        Handles common PDF extraction artifacts:
        - Excessive whitespace
        - Page numbers and headers
        - Broken hyphenation
        - Unicode issues
        """
        # Normalize unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Fix broken hyphenation (word- \nbreak -> wordbreak)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'\x00', '', text)  # Null bytes
        text = re.sub(r'[\x0c\x0b]', '\n', text)  # Form feeds
        
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def detect_section(self, text: str) -> SectionType:
        """
        Detect the section type based on text content.
        
        Uses header patterns to identify common whitepaper sections.
        """
        text_lower = text.lower()
        
        # Check first 200 chars for section headers
        header_region = text_lower[:200]
        
        for section_type, pattern in self.SECTION_PATTERNS.items():
            if re.search(pattern, header_region, re.IGNORECASE):
                return section_type
        
        return SectionType.OTHER
    
    def chunk_by_sentences(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks at sentence boundaries.
        
        More semantic than fixed-size chunking.
        """
        # Simple sentence splitting (handles common cases)
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        start_idx=current_start,
                        end_idx=current_start + len(current_chunk),
                        section_type=self.detect_section(current_chunk),
                    ))
                    current_start += len(current_chunk) - self.chunk_overlap
                    # Keep overlap from previous chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                start_idx=current_start,
                end_idx=current_start + len(current_chunk),
                section_type=self.detect_section(current_chunk),
            ))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks at paragraph boundaries.
        
        Respects document structure better than sentence chunking.
        """
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        start_idx=current_start,
                        end_idx=current_start + len(current_chunk),
                        section_type=self.detect_section(current_chunk),
                    ))
                    current_start += len(current_chunk)
                    current_chunk = para
                else:
                    # Paragraph itself is too large, need to split
                    sub_chunks = self.chunk_by_sentences(para)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                start_idx=current_start,
                end_idx=current_start + len(current_chunk),
                section_type=self.detect_section(current_chunk),
            ))
        
        return chunks
    
    def chunk_by_sections(self, text: str) -> Dict[SectionType, List[TextChunk]]:
        """
        Split text into sections, then chunk within each section.
        
        Best for preserving document structure.
        """
        # Find section boundaries
        section_boundaries = []
        
        for section_type, pattern in self.SECTION_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                section_boundaries.append((match.start(), section_type))
        
        # Sort by position
        section_boundaries.sort(key=lambda x: x[0])
        
        # Extract sections
        sections: Dict[SectionType, List[TextChunk]] = {st: [] for st in SectionType}
        
        for i, (start_pos, section_type) in enumerate(section_boundaries):
            # End is start of next section or end of document
            end_pos = section_boundaries[i + 1][0] if i + 1 < len(section_boundaries) else len(text)
            section_text = text[start_pos:end_pos]
            
            # Chunk within section
            section_chunks = self.chunk_by_paragraphs(section_text)
            for chunk in section_chunks:
                chunk.section_type = section_type
            sections[section_type].extend(section_chunks)
        
        return sections
    
    def process_document(
        self,
        text: str,
        project_symbol: str,
        project_name: str,
        chunk_method: str = "paragraphs"
    ) -> ProcessedDocument:
        """
        Fully process a whitepaper document.
        
        Args:
            text: Raw text content
            project_symbol: Crypto ticker (e.g., 'BTC')
            project_name: Full project name (e.g., 'Bitcoin')
            chunk_method: 'sentences', 'paragraphs', or 'sections'
        
        Returns:
            ProcessedDocument with chunks and sections
        """
        # Clean text
        clean = self.clean_text(text)
        
        # Chunk based on method
        if chunk_method == "sentences":
            chunks = self.chunk_by_sentences(clean)
        elif chunk_method == "paragraphs":
            chunks = self.chunk_by_paragraphs(clean)
        else:
            # Default to paragraphs if invalid method
            chunks = self.chunk_by_paragraphs(clean)
        
        # Organize chunks by section
        sections = {st: [] for st in SectionType}
        for chunk in chunks:
            sections[chunk.section_type].append(chunk)
        
        return ProcessedDocument(
            project_symbol=project_symbol,
            project_name=project_name,
            raw_text=clean,
            chunks=chunks,
            sections=sections,
            metadata={
                "chunk_method": chunk_method,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        )
    
    def extract_key_claims(self, text: str, max_claims: int = 20) -> List[str]:
        """
        Extract key claim sentences from text.
        
        Looks for sentences that make assertions about what the project does.
        Useful for focused functional claim extraction.
        """
        claims = []
        
        # Patterns that indicate functional claims
        claim_patterns = [
            r'(?:we|our|this)\s+(?:provide|enable|allow|offer|deliver|build|create)',
            r'(?:designed|built|created)\s+(?:to|for)',
            r'(?:enables?|allows?|provides?|offers?)\s+\w+',
            r'(?:solves?|addresses?|tackles?)\s+(?:the\s+)?(?:problem|issue|challenge)',
            r'(?:first|only|most|fastest|cheapest|secure|scalable)',
            r'(?:unlike|compared\s+to|better\s+than)',
        ]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 500:
                continue
            
            for pattern in claim_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append(sentence)
                    break
            
            if len(claims) >= max_claims:
                break
        
        return claims


if __name__ == "__main__":
    # Test with sample whitepaper-like text
    sample_text = """
    Abstract
    
    Bitcoin is a peer-to-peer electronic cash system that enables online payments 
    to be sent directly from one party to another without going through a financial 
    institution. Digital signatures provide part of the solution, but the main 
    benefits are lost if a trusted third party is still required to prevent double-spending.
    
    Introduction
    
    Commerce on the Internet has come to rely almost exclusively on financial 
    institutions serving as trusted third parties to process electronic payments. 
    While the system works well enough for most transactions, it still suffers from 
    the inherent weaknesses of the trust based model.
    
    Our Solution
    
    We propose a solution to the double-spending problem using a peer-to-peer network. 
    The network timestamps transactions by hashing them into an ongoing chain of 
    hash-based proof-of-work, forming a record that cannot be changed without redoing 
    the proof-of-work.
    
    Technical Architecture
    
    A block is a collection of transaction data. Each block contains a cryptographic 
    hash of the previous block, a timestamp, and transaction data. This creates an 
    immutable chain where modifying any block would require recalculating all subsequent 
    blocks.
    """
    
    processor = TextProcessor(chunk_size=300, chunk_overlap=50)
    doc = processor.process_document(sample_text, "BTC", "Bitcoin")
    
    print(f"=== Processed Document ===")
    print(f"Project: {doc.project_name} ({doc.project_symbol})")
    print(f"Total words: {doc.total_words}")
    print(f"Number of chunks: {doc.num_chunks}")
    
    print(f"\n=== Chunks ===")
    for i, chunk in enumerate(doc.chunks):
        print(f"\nChunk {i+1} ({chunk.section_type.value}):")
        print(f"  Words: {chunk.word_count}")
        print(f"  Preview: {chunk.text[:100]}...")
    
    print(f"\n=== Sections ===")
    for section_type, chunks in doc.sections.items():
        if chunks:
            print(f"  {section_type.value}: {len(chunks)} chunks")
    
    print(f"\n=== Key Claims ===")
    claims = processor.extract_key_claims(sample_text)
    for claim in claims:
        print(f"  • {claim[:80]}...")
