#!/usr/bin/env python3
"""
Embedding-based classification using Sentence Transformers.

Completely different inductive bias than NLI-based methods:
- NLI: Does text X entail label Y?
- Embedding: How geometrically close is X to Y in semantic space?

This provides independent verification - if both approaches agree,
the classification is more robust.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .taxonomy import FUNCTIONAL_CATEGORIES, LABEL_TO_CATEGORY, get_category_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingClassifier:
    """
    Classify whitepaper chunks via embedding similarity.

    Embeds both category descriptions and chunks, then computes
    cosine similarity to get category scores.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: Optional[str] = None
    ):
        """
        Initialize embedding classifier.

        Args:
            model_name: Sentence Transformer model name
                - "all-mpnet-base-v2" (default, best quality)
                - "all-MiniLM-L6-v2" (faster, slightly lower quality)
                - "BAAI/bge-large-en-v1.5" (state-of-art but larger)
            device: "cuda", "cpu", or None (auto-detect)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.categories = get_category_names()

        # Pre-compute category embeddings
        self.category_embeddings = self._embed_categories()

        logger.info(f"Embedding Classifier: model={model_name}, device={self.model.device}")

    def _build_category_texts(self) -> list[str]:
        """
        Build rich text descriptions for each category.

        Combines label, description, and keywords for better embedding.
        """
        texts = []
        for cat in self.categories:
            info = FUNCTIONAL_CATEGORIES[cat]
            # Combine all info for richer embedding
            text = f"{info['label']}: {info['description']}. Keywords: {', '.join(info['keywords'])}"
            texts.append(text)
        return texts

    def _embed_categories(self) -> np.ndarray:
        """Pre-compute normalized category embeddings."""
        category_texts = self._build_category_texts()
        embeddings = self.model.encode(category_texts, normalize_embeddings=True)
        return embeddings  # (10, embedding_dim)

    def classify_chunks(self, chunks: list[str]) -> np.ndarray:
        """
        Classify multiple chunks at once.

        Args:
            chunks: List of text chunks

        Returns:
            Score matrix of shape (n_chunks, 10)
        """
        if not chunks:
            return np.zeros((0, len(self.categories)))

        # Embed all chunks
        chunk_embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )  # (n_chunks, embedding_dim)

        # Cosine similarity (embeddings are normalized, so just dot product)
        similarities = chunk_embeddings @ self.category_embeddings.T  # (n_chunks, 10)

        # Convert to 0-1 range (cosine is -1 to 1, but usually positive for related text)
        # Shift and scale: (sim + 1) / 2 maps [-1,1] to [0,1]
        # Or just clip negatives since unrelated text shouldn't score high
        scores = np.clip(similarities, 0, 1)

        return scores

    def classify_document(
        self,
        chunks: list[str],
        symbol: str,
        aggregation: str = "mean"
    ) -> np.ndarray:
        """
        Classify all chunks for a document and aggregate.

        Args:
            chunks: List of text chunks
            symbol: Asset symbol (for logging)
            aggregation: "mean", "max", or "softmax"

        Returns:
            Document profile as normalized probability vector (10,)
        """
        if not chunks:
            logger.warning(f"{symbol}: No chunks")
            return np.zeros(len(self.categories))

        # Get scores for all chunks
        scores = self.classify_chunks(chunks)  # (n_chunks, 10)

        # Aggregate across chunks
        if aggregation == "mean":
            profile = scores.mean(axis=0)
        elif aggregation == "max":
            profile = scores.max(axis=0)
        elif aggregation == "softmax":
            # Weighted by overall confidence
            weights = scores.sum(axis=1, keepdims=True)
            weights = weights / weights.sum()
            profile = (scores * weights).sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Normalize to probability distribution
        if profile.sum() > 0:
            profile = profile / profile.sum()

        return profile

    def build_claims_matrix(
        self,
        chunks_path: Path,
        output_dir: Path,
        aggregation: str = "mean"
    ) -> dict:
        """
        Build claims matrix for all assets.

        Args:
            chunks_path: Path to extracted_chunks.json
            output_dir: Output directory
            aggregation: Chunk aggregation method

        Returns:
            Dict with matrix, symbols, categories, etc.
        """
        with open(chunks_path) as f:
            chunks_data = json.load(f)

        symbols = sorted(chunks_data.keys())
        n_entities = len(symbols)
        n_categories = len(self.categories)

        matrix = np.zeros((n_entities, n_categories))

        for i, sym in enumerate(tqdm(symbols, desc="Embedding Classification")):
            chunks = chunks_data[sym]['chunks']
            profile = self.classify_document(chunks, sym, aggregation)
            matrix[i] = profile

        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)

        # Main output
        np.save(output_dir / "claims_matrix_embedding.npy", matrix)

        # Metadata
        metadata = {
            'symbols': symbols,
            'categories': self.categories,
            'shape': list(matrix.shape),
            'model': self.model_name,
            'aggregation': aggregation,
            'method': 'sentence_transformers_cosine_similarity'
        }
        with open(output_dir / "claims_matrix_embedding_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # CSV for inspection
        import pandas as pd
        df = pd.DataFrame(matrix, index=symbols, columns=self.categories)
        df.to_csv(output_dir / "claims_matrix_embedding.csv")

        self._print_summary(symbols, matrix)

        return {
            'matrix': matrix,
            'symbols': symbols,
            'categories': self.categories
        }

    def _print_summary(self, symbols: list[str], matrix: np.ndarray):
        """Print classification summary."""
        print(f"\n{'='*60}")
        print("EMBEDDING CLAIMS MATRIX SUMMARY")
        print(f"{'='*60}")
        print(f"Assets:     {len(symbols)}")
        print(f"Categories: {len(self.categories)}")
        print(f"Model:      {self.model_name}")
        print(f"\nCategory distribution (mean across assets):")

        mean_scores = matrix.mean(axis=0)
        sorted_idx = np.argsort(mean_scores)[::-1]

        for idx in sorted_idx:
            cat = self.categories[idx]
            score = mean_scores[idx]
            bar = '#' * int(score * 40)
            print(f"  {cat:20s} {score:.3f} {bar}")

        print(f"{'='*60}")


def main():
    """Run embedding classification."""
    base_path = Path(__file__).parent.parent.parent
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    output_dir = base_path / "outputs" / "nlp"

    if not chunks_path.exists():
        logger.error("Run pdf_extractor.py first")
        return

    classifier = EmbeddingClassifier()
    results = classifier.build_claims_matrix(chunks_path, output_dir)

    print(f"\nDone! Matrix shape: {results['matrix'].shape}")


if __name__ == "__main__":
    main()
