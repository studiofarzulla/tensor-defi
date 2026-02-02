#!/usr/bin/env python3
"""
Zero-Shot Classification for TENSOR-DEFI

Uses BART-MNLI to classify whitepaper chunks into functional categories.
Builds N×10 claims matrix for alignment testing.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline

from .taxonomy import ZERO_SHOT_LABELS, LABEL_TO_CATEGORY, get_category_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeroShotClassifier:
    """Classifies whitepaper chunks using zero-shot NLI."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: Optional[str] = None):
        """Initialize classifier with BART-MNLI."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading {model_name} on {device}...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device == "cuda" else -1
        )
        self.labels = ZERO_SHOT_LABELS
        logger.info("Classifier ready")

    def classify_chunk(self, text: str) -> dict[str, float]:
        """Classify a single text chunk."""
        result = self.classifier(
            text,
            self.labels,
            multi_label=True  # Allow multiple categories
        )

        # Map scores to category names
        scores = {}
        for label, score in zip(result['labels'], result['scores']):
            # Find category index
            idx = self.labels.index(label)
            category = LABEL_TO_CATEGORY[idx]
            scores[category] = score

        return scores

    def classify_document(self, chunks: list[str], symbol: str) -> np.ndarray:
        """Classify all chunks and aggregate to document profile."""
        if not chunks:
            logger.warning(f"{symbol}: No chunks to classify")
            return np.zeros(len(LABEL_TO_CATEGORY))

        # Classify each chunk
        all_scores = []
        for chunk in tqdm(chunks, desc=f"Classifying {symbol}", leave=False):
            scores = self.classify_chunk(chunk)
            score_vector = [scores[cat] for cat in LABEL_TO_CATEGORY]
            all_scores.append(score_vector)

        # Aggregate: mean across chunks
        profile = np.mean(all_scores, axis=0)

        # Normalize to sum to 1 (probability distribution)
        profile = profile / profile.sum()

        return profile

    def build_claims_matrix(
        self,
        chunks_path: Path,
        output_dir: Path
    ) -> tuple[np.ndarray, list[str]]:
        """Build N×10 claims matrix from extracted chunks."""
        with open(chunks_path) as f:
            chunks_data = json.load(f)

        symbols = sorted(chunks_data.keys())
        n_entities = len(symbols)
        n_categories = len(LABEL_TO_CATEGORY)

        claims_matrix = np.zeros((n_entities, n_categories))

        for i, symbol in enumerate(tqdm(symbols, desc="Building claims matrix")):
            chunks = chunks_data[symbol]['chunks']
            profile = self.classify_document(chunks, symbol)
            claims_matrix[i] = profile

        # Save matrix
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "claims_matrix.npy", claims_matrix)

        # Save metadata
        metadata = {
            'symbols': symbols,
            'categories': get_category_names(),
            'shape': list(claims_matrix.shape)
        }
        with open(output_dir / "claims_matrix_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save as readable CSV
        import pandas as pd
        df = pd.DataFrame(
            claims_matrix,
            index=symbols,
            columns=get_category_names()
        )
        df.to_csv(output_dir / "claims_matrix.csv")

        logger.info(f"Saved claims matrix: {claims_matrix.shape}")
        self._print_summary(symbols, claims_matrix)

        return claims_matrix, symbols

    def _print_summary(self, symbols: list[str], matrix: np.ndarray):
        """Print classification summary."""
        categories = get_category_names()

        print(f"\n{'='*60}")
        print("CLAIMS MATRIX SUMMARY")
        print(f"{'='*60}")
        print(f"Entities:   {len(symbols)}")
        print(f"Categories: {len(categories)}")
        print(f"\nCategory distribution (mean across all entities):")

        mean_scores = matrix.mean(axis=0)
        sorted_idx = np.argsort(mean_scores)[::-1]

        for idx in sorted_idx:
            cat = categories[idx]
            score = mean_scores[idx]
            bar = '█' * int(score * 50)
            print(f"  {cat:20s} {score:.3f} {bar}")

        # Top entity per category
        print(f"\nTop entity per category:")
        for i, cat in enumerate(categories):
            top_idx = np.argmax(matrix[:, i])
            top_symbol = symbols[top_idx]
            top_score = matrix[top_idx, i]
            print(f"  {cat:20s} → {top_symbol} ({top_score:.3f})")

        print(f"{'='*60}")


def main():
    """Run zero-shot classification."""
    base_path = Path(__file__).parent.parent.parent
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    output_dir = base_path / "outputs" / "nlp"

    if not chunks_path.exists():
        logger.error("Run pdf_extractor.py first")
        return

    classifier = ZeroShotClassifier()
    classifier.build_claims_matrix(chunks_path, output_dir)


if __name__ == "__main__":
    main()
