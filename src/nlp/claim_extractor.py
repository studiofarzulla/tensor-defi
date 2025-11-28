"""
Claim Extractor for Crypto Whitepapers

Extracts functional claims from whitepaper text using:
1. Zero-shot classification with BART MNLI
2. Keyword matching as fallback
3. Multi-label classification for functional taxonomy

This is the core NLP component for functional claim extraction.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from .taxonomy import (
    FunctionalCategory,
    FUNCTIONAL_TAXONOMY,
    get_candidate_labels_flat,
    get_zero_shot_labels,
    category_to_vector,
)

# HuggingFace transformers for zero-shot classification
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class ClaimClassification:
    """Result of classifying a text chunk."""
    text: str
    categories: List[FunctionalCategory]
    scores: Dict[FunctionalCategory, float]
    top_category: Optional[FunctionalCategory]
    top_score: float
    confidence: float  # Overall classification confidence


@dataclass
class DocumentClaims:
    """All classified claims for a document."""
    project_symbol: str
    project_name: str
    claims: List[ClaimClassification]
    functional_profile: Dict[FunctionalCategory, float]  # Aggregated scores
    
    @property
    def num_claims(self) -> int:
        return len(self.claims)
    
    @property
    def primary_functions(self) -> List[FunctionalCategory]:
        """Get top functional categories for this project."""
        sorted_cats = sorted(
            self.functional_profile.items(),
            key=lambda x: x[1],
            reverse=True
        )
        # Return categories with score > 0.3
        return [cat for cat, score in sorted_cats if score > 0.3]
    
    def to_vector(self) -> np.ndarray:
        """Convert functional profile to feature vector."""
        all_cats = list(FunctionalCategory)
        return np.array([
            self.functional_profile.get(cat, 0.0)
            for cat in all_cats
        ], dtype=np.float32)


class ClaimExtractor:
    """Extract and classify functional claims from whitepaper text."""
    
    def __init__(
        self,
        model: str = "facebook/bart-large-mnli",
        device: int = -1,  # -1 for CPU, 0+ for GPU
        use_keywords_fallback: bool = True,
    ):
        """
        Initialize claim extractor.
        
        Args:
            model: HuggingFace model for zero-shot classification
            device: Device ID (-1 for CPU)
            use_keywords_fallback: Use keyword matching when transformers unavailable
        """
        self.model_name = model
        self.device = device
        self.use_keywords_fallback = use_keywords_fallback
        
        # Initialize zero-shot pipeline if available
        if HAS_TRANSFORMERS:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model,
                    device=device
                )
                self.has_classifier = True
                print(f"✓ Loaded zero-shot classifier: {model}")
            except Exception as e:
                print(f"✗ Failed to load classifier: {e}")
                self.classifier = None
                self.has_classifier = False
        else:
            print("Warning: transformers not installed. Using keyword fallback.")
            self.classifier = None
            self.has_classifier = False
        
        # Build keyword index for fallback
        self._build_keyword_index()
    
    def _build_keyword_index(self) -> None:
        """Build keyword-to-category mapping for fallback classification."""
        self.keyword_index: Dict[str, FunctionalCategory] = {}
        
        for category, entry in FUNCTIONAL_TAXONOMY.items():
            for keyword in entry.keywords:
                # Normalize keyword
                keyword_lower = keyword.lower().strip()
                self.keyword_index[keyword_lower] = category
    
    def classify_text(
        self,
        text: str,
        threshold: float = 0.3,
        multi_label: bool = True
    ) -> ClaimClassification:
        """
        Classify a text chunk into functional categories.
        
        Args:
            text: Text to classify
            threshold: Minimum score to include category
            multi_label: Allow multiple category labels
        
        Returns:
            ClaimClassification with categories and scores
        """
        if self.has_classifier:
            return self._classify_zero_shot(text, threshold, multi_label)
        elif self.use_keywords_fallback:
            return self._classify_keywords(text, threshold)
        else:
            # Return empty classification
            return ClaimClassification(
                text=text,
                categories=[],
                scores={},
                top_category=None,
                top_score=0.0,
                confidence=0.0
            )
    
    def _classify_zero_shot(
        self,
        text: str,
        threshold: float,
        multi_label: bool
    ) -> ClaimClassification:
        """Classify using zero-shot classification."""
        # Get candidate labels from taxonomy
        candidate_labels = [
            entry.description
            for entry in FUNCTIONAL_TAXONOMY.values()
        ]
        
        # Run zero-shot classification
        result = self.classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=multi_label
        )
        
        # Map labels back to categories
        label_to_category = {
            entry.description: category
            for category, entry in FUNCTIONAL_TAXONOMY.items()
        }
        
        scores: Dict[FunctionalCategory, float] = {}
        categories: List[FunctionalCategory] = []
        
        for label, score in zip(result["labels"], result["scores"]):
            category = label_to_category.get(label)
            if category:
                scores[category] = score
                if score >= threshold:
                    categories.append(category)
        
        # Determine top category
        top_category = categories[0] if categories else None
        top_score = scores.get(top_category, 0.0) if top_category else 0.0
        
        # Confidence based on score spread
        if len(result["scores"]) >= 2:
            confidence = result["scores"][0] - result["scores"][1]
        else:
            confidence = result["scores"][0] if result["scores"] else 0.0
        
        return ClaimClassification(
            text=text,
            categories=categories,
            scores=scores,
            top_category=top_category,
            top_score=top_score,
            confidence=confidence
        )
    
    def _classify_keywords(
        self,
        text: str,
        threshold: float
    ) -> ClaimClassification:
        """Classify using keyword matching (fallback)."""
        text_lower = text.lower()
        
        # Count keyword matches per category
        category_counts: Dict[FunctionalCategory, int] = {}
        category_matches: Dict[FunctionalCategory, List[str]] = {}
        
        for keyword, category in self.keyword_index.items():
            if keyword in text_lower:
                category_counts[category] = category_counts.get(category, 0) + 1
                if category not in category_matches:
                    category_matches[category] = []
                category_matches[category].append(keyword)
        
        # Convert counts to scores (normalized)
        total_matches = sum(category_counts.values())
        if total_matches == 0:
            return ClaimClassification(
                text=text,
                categories=[],
                scores={},
                top_category=None,
                top_score=0.0,
                confidence=0.0
            )
        
        scores: Dict[FunctionalCategory, float] = {
            cat: count / total_matches
            for cat, count in category_counts.items()
        }
        
        # Apply threshold
        categories = [cat for cat, score in scores.items() if score >= threshold]
        
        # Sort by score
        categories.sort(key=lambda c: scores[c], reverse=True)
        
        top_category = categories[0] if categories else None
        top_score = scores.get(top_category, 0.0) if top_category else 0.0
        
        return ClaimClassification(
            text=text,
            categories=categories,
            scores=scores,
            top_category=top_category,
            top_score=top_score,
            confidence=top_score  # Use top score as confidence for keyword method
        )
    
    def classify_chunks(
        self,
        chunks: List[str],
        threshold: float = 0.3,
        show_progress: bool = True
    ) -> List[ClaimClassification]:
        """
        Classify multiple text chunks.
        
        Args:
            chunks: List of text chunks
            threshold: Minimum score threshold
            show_progress: Show progress indicator
        
        Returns:
            List of ClaimClassification objects
        """
        results = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Classified {i + 1}/{total} chunks")
            
            result = self.classify_text(chunk, threshold=threshold)
            results.append(result)
        
        if show_progress:
            print(f"  Classified {total}/{total} chunks complete")
        
        return results
    
    def extract_document_claims(
        self,
        chunks: List[str],
        project_symbol: str,
        project_name: str,
        threshold: float = 0.3,
        show_progress: bool = True
    ) -> DocumentClaims:
        """
        Extract and aggregate functional claims from a document.
        
        Args:
            chunks: Document text chunks
            project_symbol: Crypto ticker
            project_name: Full project name
            threshold: Classification threshold
            show_progress: Show progress
        
        Returns:
            DocumentClaims with aggregated functional profile
        """
        if show_progress:
            print(f"Extracting claims for {project_name} ({len(chunks)} chunks)...")
        
        # Classify all chunks
        claims = self.classify_chunks(chunks, threshold=threshold, show_progress=show_progress)
        
        # Aggregate into functional profile
        functional_profile = self._aggregate_claims(claims)
        
        return DocumentClaims(
            project_symbol=project_symbol,
            project_name=project_name,
            claims=claims,
            functional_profile=functional_profile
        )
    
    def _aggregate_claims(
        self,
        claims: List[ClaimClassification]
    ) -> Dict[FunctionalCategory, float]:
        """
        Aggregate individual claim classifications into document-level profile.
        
        Uses weighted average based on confidence.
        """
        # Initialize scores for all categories
        profile: Dict[FunctionalCategory, float] = {
            cat: 0.0 for cat in FunctionalCategory
        }
        
        # Weight accumulator
        total_weight = 0.0
        
        for claim in claims:
            weight = max(0.1, claim.confidence)  # Minimum weight of 0.1
            total_weight += weight
            
            for category, score in claim.scores.items():
                profile[category] += score * weight
        
        # Normalize
        if total_weight > 0:
            for cat in profile:
                profile[cat] /= total_weight
        
        return profile
    
    def compare_profiles(
        self,
        profile1: Dict[FunctionalCategory, float],
        profile2: Dict[FunctionalCategory, float]
    ) -> float:
        """
        Compare two functional profiles using cosine similarity.
        
        Args:
            profile1: First functional profile
            profile2: Second functional profile
        
        Returns:
            Similarity score (0-1)
        """
        cats = list(FunctionalCategory)
        
        vec1 = np.array([profile1.get(c, 0.0) for c in cats])
        vec2 = np.array([profile2.get(c, 0.0) for c in cats])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


def create_mock_classifications(
    texts: List[str]
) -> List[ClaimClassification]:
    """
    Create mock classifications for testing without transformers.
    
    Uses simple heuristics based on text content.
    """
    classifications = []
    
    for text in texts:
        text_lower = text.lower()
        
        # Simple heuristic classification
        scores: Dict[FunctionalCategory, float] = {}
        
        if any(kw in text_lower for kw in ["store", "value", "gold", "savings"]):
            scores[FunctionalCategory.STORE_OF_VALUE] = 0.7
        if any(kw in text_lower for kw in ["payment", "cash", "transfer", "transaction"]):
            scores[FunctionalCategory.MEDIUM_OF_EXCHANGE] = 0.7
        if any(kw in text_lower for kw in ["smart contract", "dapp", "defi"]):
            scores[FunctionalCategory.SMART_CONTRACTS] = 0.7
        if any(kw in text_lower for kw in ["scaling", "layer", "throughput"]):
            scores[FunctionalCategory.INFRASTRUCTURE] = 0.7
        if any(kw in text_lower for kw in ["privacy", "anonymous", "confidential"]):
            scores[FunctionalCategory.PRIVACY] = 0.7
        if any(kw in text_lower for kw in ["governance", "dao", "voting"]):
            scores[FunctionalCategory.GOVERNANCE] = 0.7
        if any(kw in text_lower for kw in ["oracle", "data feed"]):
            scores[FunctionalCategory.DATA_ORACLE] = 0.7
        
        categories = [cat for cat, score in scores.items() if score >= 0.3]
        top_cat = categories[0] if categories else None
        top_score = scores.get(top_cat, 0.0) if top_cat else 0.0
        
        classifications.append(ClaimClassification(
            text=text,
            categories=categories,
            scores=scores,
            top_category=top_cat,
            top_score=top_score,
            confidence=top_score
        ))
    
    return classifications


if __name__ == "__main__":
    print("=== Claim Extractor Test ===\n")
    
    # Sample whitepaper-like text
    sample_texts = [
        "Bitcoin is a peer-to-peer electronic cash system that enables online payments to be sent directly from one party to another without going through a financial institution.",
        "Ethereum provides a decentralized platform for smart contracts and decentralized applications.",
        "Solana achieves high throughput through its innovative proof of history consensus mechanism, enabling 65,000 transactions per second.",
        "Chainlink provides decentralized oracle services that connect smart contracts with real-world data.",
        "Monero uses ring signatures and stealth addresses to provide untraceable, private transactions.",
        "The DAO governance structure allows token holders to vote on protocol upgrades and treasury allocation.",
    ]
    
    # Test with mock classifications first
    print("Testing mock classifications...")
    mock_results = create_mock_classifications(sample_texts)
    
    for i, result in enumerate(mock_results):
        print(f"\nText {i+1}: {sample_texts[i][:50]}...")
        print(f"  Categories: {[c.value for c in result.categories]}")
        print(f"  Top: {result.top_category.value if result.top_category else 'None'} ({result.top_score:.2f})")
    
    # Test with real classifier if available
    print("\n=== Zero-Shot Classifier Test ===")
    try:
        extractor = ClaimExtractor(device=-1)  # CPU
        
        if extractor.has_classifier:
            print("\nClassifying with BART MNLI...")
            for text in sample_texts[:3]:  # Test first 3 only (slow)
                result = extractor.classify_text(text)
                print(f"\nText: {text[:60]}...")
                print(f"  Top Category: {result.top_category.value if result.top_category else 'None'}")
                print(f"  Score: {result.top_score:.3f}")
                print(f"  All scores: {[(c.value, f'{s:.2f}') for c, s in result.scores.items() if s > 0.1]}")
        else:
            print("Using keyword fallback...")
            for text in sample_texts[:3]:
                result = extractor.classify_text(text)
                print(f"\nText: {text[:60]}...")
                print(f"  Top Category: {result.top_category.value if result.top_category else 'None'}")
                
    except Exception as e:
        print(f"✗ Classifier test failed: {e}")
