#!/usr/bin/env python3
"""
Ensemble Zero-Shot Classification for TENSOR-DEFI Expansion

Uses 3-model ensemble (BART, DeBERTa, RoBERTa) for improved inter-model agreement.
Implements both soft voting (probability averaging) and hard voting (2/3 majority).

Target: Improve Cohen's kappa from 0.14 to 0.45-0.60.
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


# Ensemble models - all trained on MNLI for zero-shot classification
ENSEMBLE_MODELS = [
    "facebook/bart-large-mnli",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "FacebookAI/roberta-large-mnli"
]

# Minimum agreement threshold for hard voting (2/3 = 0.67)
HARD_VOTE_THRESHOLD = 0.67


class EnsembleClassifier:
    """
    Ensemble zero-shot classifier using multiple MNLI models.

    Voting strategies:
    - Soft: Average probability scores across models
    - Hard: Require 2/3 majority agreement at threshold
    """

    def __init__(
        self,
        models: list[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize ensemble classifier.

        Args:
            models: List of HuggingFace model names (default: ENSEMBLE_MODELS)
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Threshold for "positive" classification in hard voting
        """
        self.models = models or ENSEMBLE_MODELS
        self.confidence_threshold = confidence_threshold

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Initializing {len(self.models)}-model ensemble on {device}")

        self.classifiers = []
        for model_name in self.models:
            logger.info(f"Loading {model_name}...")
            clf = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if device == "cuda" else -1
            )
            self.classifiers.append(clf)

        self.labels = ZERO_SHOT_LABELS
        logger.info(f"Ensemble ready: {len(self.classifiers)} models loaded")

    def classify_chunk_single(self, classifier, text: str) -> dict[str, float]:
        """Classify using a single model."""
        result = classifier(
            text,
            self.labels,
            multi_label=True
        )

        scores = {}
        for label, score in zip(result['labels'], result['scores']):
            idx = self.labels.index(label)
            category = LABEL_TO_CATEGORY[idx]
            scores[category] = score

        return scores

    def classify_chunk_ensemble(
        self,
        text: str,
        return_individual: bool = False
    ) -> dict:
        """
        Classify text using all ensemble models.

        Returns dict with:
        - soft_scores: Averaged probability scores
        - hard_scores: Binary scores based on 2/3 agreement
        - individual_scores: Per-model scores (if return_individual=True)
        """
        all_scores = []

        for clf in self.classifiers:
            scores = self.classify_chunk_single(clf, text)
            all_scores.append(scores)

        # Convert to arrays for vectorized operations
        categories = get_category_names()
        score_matrix = np.array([
            [s[cat] for cat in categories]
            for s in all_scores
        ])  # Shape: (n_models, n_categories)

        # Soft voting: mean across models
        soft_scores = score_matrix.mean(axis=0)

        # Hard voting: count models that pass threshold, require 2/3 majority
        binary_votes = (score_matrix >= self.confidence_threshold).astype(int)
        vote_counts = binary_votes.sum(axis=0)
        hard_scores = (vote_counts >= len(self.classifiers) * HARD_VOTE_THRESHOLD).astype(float)

        # For hard voting, use average score where majority agrees, 0 otherwise
        hard_weighted = np.where(
            hard_scores > 0,
            soft_scores,
            0.0
        )

        result = {
            'soft_scores': {cat: soft_scores[i] for i, cat in enumerate(categories)},
            'hard_scores': {cat: hard_weighted[i] for i, cat in enumerate(categories)},
            'agreement_ratio': {
                cat: vote_counts[i] / len(self.classifiers)
                for i, cat in enumerate(categories)
            }
        }

        if return_individual:
            result['individual_scores'] = all_scores

        return result

    def classify_document(
        self,
        chunks: list[str],
        symbol: str,
        method: str = 'soft'
    ) -> np.ndarray:
        """
        Classify all chunks and aggregate to document profile.

        Args:
            chunks: List of text chunks
            symbol: Asset symbol (for logging)
            method: 'soft' or 'hard' voting

        Returns:
            Document profile as normalized probability vector
        """
        if not chunks:
            logger.warning(f"{symbol}: No chunks to classify")
            return np.zeros(len(LABEL_TO_CATEGORY))

        all_scores = []
        for chunk in tqdm(chunks, desc=f"Classifying {symbol} (ensemble)", leave=False):
            result = self.classify_chunk_ensemble(chunk)

            if method == 'soft':
                scores = result['soft_scores']
            else:
                scores = result['hard_scores']

            score_vector = [scores[cat] for cat in get_category_names()]
            all_scores.append(score_vector)

        # Aggregate: mean across chunks
        profile = np.mean(all_scores, axis=0)

        # Normalize to sum to 1 (probability distribution)
        if profile.sum() > 0:
            profile = profile / profile.sum()

        return profile

    def build_claims_matrices(
        self,
        chunks_path: Path,
        output_dir: Path
    ) -> dict:
        """
        Build claims matrices for both voting methods.

        Returns dict with:
        - soft_matrix: N x 10 matrix using soft voting
        - hard_matrix: N x 10 matrix using hard voting
        - symbols: List of asset symbols
        - agreement_stats: Inter-model agreement statistics
        """
        with open(chunks_path) as f:
            chunks_data = json.load(f)

        symbols = sorted(chunks_data.keys())
        n_entities = len(symbols)
        n_categories = len(LABEL_TO_CATEGORY)

        soft_matrix = np.zeros((n_entities, n_categories))
        hard_matrix = np.zeros((n_entities, n_categories))

        # Track agreement statistics
        all_agreements = []

        for i, symbol in enumerate(tqdm(symbols, desc="Building ensemble claims matrices")):
            chunks = chunks_data[symbol]['chunks']

            # Get both profiles
            soft_profile = self.classify_document(chunks, symbol, method='soft')
            hard_profile = self.classify_document(chunks, symbol, method='hard')

            soft_matrix[i] = soft_profile
            hard_matrix[i] = hard_profile

            # Sample agreement stats from first chunk
            if chunks:
                sample_result = self.classify_chunk_ensemble(chunks[0])
                agreements = list(sample_result['agreement_ratio'].values())
                all_agreements.append(np.mean(agreements))

        # Save matrices
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "claims_matrix_ensemble_soft.npy", soft_matrix)
        np.save(output_dir / "claims_matrix_ensemble_hard.npy", hard_matrix)

        # Compute inter-model agreement (Cohen's kappa proxy)
        mean_agreement = np.mean(all_agreements) if all_agreements else 0.0

        # Save metadata
        metadata = {
            'symbols': symbols,
            'categories': get_category_names(),
            'soft_shape': list(soft_matrix.shape),
            'hard_shape': list(hard_matrix.shape),
            'models': self.models,
            'confidence_threshold': self.confidence_threshold,
            'mean_agreement': float(mean_agreement),
            'voting_methods': ['soft', 'hard']
        }
        with open(output_dir / "claims_matrix_ensemble_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save as readable CSVs
        import pandas as pd
        categories = get_category_names()

        df_soft = pd.DataFrame(soft_matrix, index=symbols, columns=categories)
        df_soft.to_csv(output_dir / "claims_matrix_ensemble_soft.csv")

        df_hard = pd.DataFrame(hard_matrix, index=symbols, columns=categories)
        df_hard.to_csv(output_dir / "claims_matrix_ensemble_hard.csv")

        logger.info(f"Saved ensemble matrices: soft={soft_matrix.shape}, hard={hard_matrix.shape}")
        self._print_summary(symbols, soft_matrix, hard_matrix, mean_agreement)

        return {
            'soft_matrix': soft_matrix,
            'hard_matrix': hard_matrix,
            'symbols': symbols,
            'mean_agreement': mean_agreement
        }

    def _print_summary(
        self,
        symbols: list[str],
        soft_matrix: np.ndarray,
        hard_matrix: np.ndarray,
        mean_agreement: float
    ):
        """Print ensemble classification summary."""
        categories = get_category_names()

        print(f"\n{'='*70}")
        print("ENSEMBLE CLAIMS MATRIX SUMMARY")
        print(f"{'='*70}")
        print(f"Entities:        {len(symbols)}")
        print(f"Categories:      {len(categories)}")
        print(f"Ensemble models: {len(self.models)}")
        print(f"Mean agreement:  {mean_agreement:.3f}")
        print(f"\nCategory distribution (soft voting, mean across entities):")

        mean_soft = soft_matrix.mean(axis=0)
        mean_hard = hard_matrix.mean(axis=0)
        sorted_idx = np.argsort(mean_soft)[::-1]

        for idx in sorted_idx:
            cat = categories[idx]
            soft = mean_soft[idx]
            hard = mean_hard[idx]
            bar = '#' * int(soft * 40)
            print(f"  {cat:20s} soft={soft:.3f} hard={hard:.3f} {bar}")

        # Compare soft vs hard agreement
        print(f"\nSoft vs Hard correlation: {np.corrcoef(mean_soft, mean_hard)[0,1]:.3f}")
        print(f"{'='*70}")


def compute_inter_model_kappa(
    classifiers: list,
    test_texts: list[str],
    labels: list[str],
    threshold: float = 0.5
) -> float:
    """
    Compute average pairwise Cohen's kappa between models.

    This gives a proper measure of inter-model agreement.
    """
    from sklearn.metrics import cohen_kappa_score

    # Get binary predictions from each model
    n_models = len(classifiers)
    n_texts = len(test_texts)
    n_labels = len(labels)

    predictions = np.zeros((n_models, n_texts, n_labels))

    for m, clf in enumerate(classifiers):
        for t, text in enumerate(test_texts):
            result = clf(text, labels, multi_label=True)
            for label, score in zip(result['labels'], result['scores']):
                idx = labels.index(label)
                predictions[m, t, idx] = 1 if score >= threshold else 0

    # Compute pairwise kappa
    kappas = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Flatten predictions for comparison
            pred_i = predictions[i].flatten()
            pred_j = predictions[j].flatten()
            kappa = cohen_kappa_score(pred_i, pred_j)
            kappas.append(kappa)

    return np.mean(kappas)


def main():
    """Run ensemble classification."""
    base_path = Path(__file__).parent.parent.parent
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    output_dir = base_path / "outputs" / "nlp"

    if not chunks_path.exists():
        logger.error("Run pdf_extractor.py first")
        return

    classifier = EnsembleClassifier()
    results = classifier.build_claims_matrices(chunks_path, output_dir)

    print(f"\nFinal mean agreement: {results['mean_agreement']:.3f}")


if __name__ == "__main__":
    main()
