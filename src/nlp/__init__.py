"""
NLP Pipeline for Crypto Whitepaper Analysis

Extracts functional claims from cryptocurrency project communications
and generates embeddings for tensor construction.

Modules:
- taxonomy: Functional category definitions
- text_processor: Document chunking and cleaning
- claim_extractor: Zero-shot and fine-tuned classification
- embedding_generator: Sentence embeddings via Ollama
- whitepaper_collector: PDF acquisition and text extraction
"""

from .taxonomy import (
    FUNCTIONAL_TAXONOMY,
    FunctionalCategory,
    TaxonomyEntry,
    TOP_20_PROJECTS,
    get_all_keywords,
    get_zero_shot_labels,
    get_candidate_labels_flat,
    category_to_vector,
    vector_to_categories,
)

from .text_processor import (
    TextProcessor,
    TextChunk,
    ProcessedDocument,
    SectionType,
)

from .whitepaper_collector import (
    WhitepaperCollector,
    CollectedWhitepaper,
    WhitepaperMetadata,
)

from .embedding_generator import (
    EmbeddingGenerator,
    EmbeddingResult,
    DocumentEmbeddings,
    create_mock_embeddings,
)

from .claim_extractor import (
    ClaimExtractor,
    ClaimClassification,
    DocumentClaims,
    create_mock_classifications,
)

__all__ = [
    # Taxonomy
    'FUNCTIONAL_TAXONOMY',
    'FunctionalCategory',
    'TaxonomyEntry',
    'TOP_20_PROJECTS',
    'get_all_keywords',
    'get_zero_shot_labels',
    'get_candidate_labels_flat',
    'category_to_vector',
    'vector_to_categories',
    # Text Processing
    'TextProcessor',
    'TextChunk',
    'ProcessedDocument',
    'SectionType',
    # Whitepaper Collection
    'WhitepaperCollector',
    'CollectedWhitepaper',
    'WhitepaperMetadata',
    # Embeddings
    'EmbeddingGenerator',
    'EmbeddingResult',
    'DocumentEmbeddings',
    'create_mock_embeddings',
    # Claim Extraction
    'ClaimExtractor',
    'ClaimClassification',
    'DocumentClaims',
    'create_mock_classifications',
]
