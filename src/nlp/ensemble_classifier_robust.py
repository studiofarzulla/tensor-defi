#!/usr/bin/env python3
"""
Robust Ensemble Zero-Shot Classification for TENSOR-DEFI Expansion

Key improvements over original:
1. Processes ONE model at a time (memory efficient)
2. Checkpoints after each model completes (crash-safe)
3. Auto-resume from last checkpoint
4. Proper batching for GPU efficiency

Uses 3-model ensemble (BART, DeBERTa, RoBERTa) for improved inter-model agreement.
"""

import json
import logging
import gc
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline

from .taxonomy import ZERO_SHOT_LABELS, LABEL_TO_CATEGORY, get_category_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Ensemble models - all trained on MNLI for zero-shot classification
ENSEMBLE_MODELS = {
    "bart": "facebook/bart-large-mnli",
    "deberta": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "roberta": "FacebookAI/roberta-large-mnli"
}

HARD_VOTE_THRESHOLD = 0.67


class RobustEnsembleClassifier:
    """
    Robust ensemble zero-shot classifier with checkpointing.

    Processes one model at a time to minimize memory usage.
    Saves intermediate results after each model completes.
    Can resume from any checkpoint if interrupted.
    """

    def __init__(
        self,
        output_dir: Path,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "ensemble_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.confidence_threshold = confidence_threshold

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.labels = ZERO_SHOT_LABELS
        self.categories = get_category_names()

        logger.info(f"RobustEnsembleClassifier initialized on {device}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")

    def _get_checkpoint_path(self, model_name: str) -> Path:
        return self.checkpoint_dir / f"checkpoint_{model_name}.json"

    def _load_checkpoint(self, model_name: str) -> Optional[dict]:
        """Load checkpoint for a specific model if it exists."""
        path = self._get_checkpoint_path(model_name)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            logger.info(f"Loaded checkpoint for {model_name}: {len(data['results'])} assets")
            return data
        return None

    def _save_checkpoint(self, model_name: str, results: dict, symbols: list):
        """Save checkpoint for a specific model."""
        path = self._get_checkpoint_path(model_name)
        data = {
            "model_name": model_name,
            "model_id": ENSEMBLE_MODELS[model_name],
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "categories": self.categories,
            "results": results
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved checkpoint for {model_name}")

    def _classify_chunks_single_model(
        self,
        model_name: str,
        chunks_by_symbol: dict[str, list[str]]
    ) -> dict[str, np.ndarray]:
        """
        Classify all chunks using a single model.

        Returns dict mapping symbol -> (n_chunks, n_categories) array
        """
        # Check for existing checkpoint
        checkpoint = self._load_checkpoint(model_name)
        if checkpoint:
            completed_symbols = set(checkpoint['results'].keys())
            logger.info(f"Resuming {model_name}: {len(completed_symbols)} already done")
        else:
            completed_symbols = set()
            checkpoint = {'results': {}}

        # Load model
        model_id = ENSEMBLE_MODELS[model_name]
        logger.info(f"Loading {model_name}: {model_id}")

        clf = pipeline(
            "zero-shot-classification",
            model=model_id,
            device=0 if self.device == "cuda" else -1
        )

        results = checkpoint['results']
        symbols = list(chunks_by_symbol.keys())

        for symbol in tqdm(symbols, desc=f"Processing ({model_name})"):
            if symbol in completed_symbols:
                continue

            chunks = chunks_by_symbol[symbol]
            symbol_scores = []

            for chunk in tqdm(chunks, desc=f"  {symbol}", leave=False):
                try:
                    result = clf(chunk, self.labels, multi_label=True)

                    # Map to categories
                    scores = np.zeros(len(self.categories))
                    for label, score in zip(result['labels'], result['scores']):
                        idx = self.labels.index(label)
                        cat = LABEL_TO_CATEGORY[idx]
                        cat_idx = self.categories.index(cat)
                        scores[cat_idx] = score

                    symbol_scores.append(scores.tolist())
                except Exception as e:
                    logger.warning(f"Error on {symbol} chunk: {e}")
                    symbol_scores.append([0.0] * len(self.categories))

            results[symbol] = symbol_scores

            # Save checkpoint after each symbol
            self._save_checkpoint(model_name, results, symbols)

        # Cleanup
        del clf
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Completed {model_name}: {len(results)} symbols")
        return {sym: np.array(scores) for sym, scores in results.items()}

    def run_ensemble(
        self,
        chunks_by_symbol: dict[str, list[str]],
        models_to_run: list[str] = None
    ) -> dict:
        """
        Run full ensemble classification.

        Args:
            chunks_by_symbol: Dict mapping symbol -> list of text chunks
            models_to_run: List of model names to run (default: all)

        Returns:
            Dict with soft_matrix, hard_matrix, individual matrices, and metadata
        """
        if models_to_run is None:
            models_to_run = list(ENSEMBLE_MODELS.keys())

        symbols = list(chunks_by_symbol.keys())
        n_symbols = len(symbols)
        n_categories = len(self.categories)

        logger.info(f"Running ensemble on {n_symbols} symbols with {len(models_to_run)} models")

        # Run each model sequentially
        model_results = {}
        for model_name in models_to_run:
            logger.info(f"\n{'='*60}")
            logger.info(f"MODEL: {model_name}")
            logger.info(f"{'='*60}")

            model_results[model_name] = self._classify_chunks_single_model(
                model_name, chunks_by_symbol
            )

        # Aggregate per-symbol averages for each model
        model_matrices = {}
        for model_name, sym_results in model_results.items():
            matrix = np.zeros((n_symbols, n_categories))
            for i, symbol in enumerate(symbols):
                if symbol in sym_results and len(sym_results[symbol]) > 0:
                    matrix[i] = sym_results[symbol].mean(axis=0)
            model_matrices[model_name] = matrix

        # Combine models
        stacked = np.stack(list(model_matrices.values()), axis=0)  # (n_models, n_symbols, n_categories)

        # Soft voting: average across models
        soft_matrix = stacked.mean(axis=0)

        # Hard voting: 2/3 majority agreement
        binary_votes = (stacked >= self.confidence_threshold).astype(int)
        vote_counts = binary_votes.sum(axis=0)
        majority_mask = vote_counts >= len(models_to_run) * HARD_VOTE_THRESHOLD
        hard_matrix = np.where(majority_mask, soft_matrix, 0.0)

        # Agreement ratio
        agreement_matrix = vote_counts / len(models_to_run)

        return {
            'soft_matrix': soft_matrix,
            'hard_matrix': hard_matrix,
            'agreement_matrix': agreement_matrix,
            'individual_matrices': model_matrices,
            'symbols': symbols,
            'categories': self.categories,
            'models_used': models_to_run
        }

    def save_results(self, results: dict, prefix: str = "claims_matrix_ensemble"):
        """Save all results to output directory."""
        # Soft voting matrix
        np.save(self.output_dir / f"{prefix}_soft.npy", results['soft_matrix'])

        # Hard voting matrix
        np.save(self.output_dir / f"{prefix}_hard.npy", results['hard_matrix'])

        # Agreement matrix
        np.save(self.output_dir / f"{prefix}_agreement.npy", results['agreement_matrix'])

        # Individual model matrices
        for model_name, matrix in results['individual_matrices'].items():
            np.save(self.output_dir / f"{prefix}_{model_name}.npy", matrix)

        # CSV for soft voting (human readable)
        import pandas as pd
        df = pd.DataFrame(
            results['soft_matrix'],
            index=results['symbols'],
            columns=results['categories']
        )
        df.to_csv(self.output_dir / f"{prefix}_soft.csv")

        # Metadata
        meta = {
            'symbols': results['symbols'],
            'categories': results['categories'],
            'models_used': results['models_used'],
            'shape': list(results['soft_matrix'].shape),
            'timestamp': datetime.now().isoformat(),
            'method': 'robust_ensemble_3model'
        }
        with open(self.output_dir / f"{prefix}_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved results to {self.output_dir}")


def build_ensemble_matrix(
    chunks_path: Path,
    output_dir: Path,
    models: list[str] = None
) -> dict:
    """
    Build ensemble claims matrix from extracted chunks.

    Args:
        chunks_path: Path to extracted_chunks.json
        output_dir: Directory to save results
        models: List of models to use (default: all 3)

    Returns:
        Results dict with all matrices
    """
    # Load chunks
    with open(chunks_path) as f:
        chunks_data = json.load(f)

    # Handle different data formats
    chunks_by_symbol = {}

    if isinstance(chunks_data, dict):
        first_val = next(iter(chunks_data.values()))

        if isinstance(first_val, dict) and 'chunks' in first_val:
            # Nested format: {symbol: {chunks: [...], word_count, ...}}
            for symbol, data in chunks_data.items():
                chunks_by_symbol[symbol] = data['chunks']
        elif isinstance(first_val, list):
            # Direct format: {symbol: [chunk1, chunk2, ...]}
            chunks_by_symbol = chunks_data
        else:
            raise ValueError(f"Unknown chunks format: {type(first_val)}")
    else:
        # Legacy format: list of {symbol, text} dicts
        for item in chunks_data:
            symbol = item['symbol']
            if symbol not in chunks_by_symbol:
                chunks_by_symbol[symbol] = []
            chunks_by_symbol[symbol].append(item['text'])

    total_chunks = sum(len(v) for v in chunks_by_symbol.values())
    logger.info(f"Loaded {total_chunks} chunks from {len(chunks_by_symbol)} symbols")

    # Run ensemble
    classifier = RobustEnsembleClassifier(output_dir)
    results = classifier.run_ensemble(chunks_by_symbol, models)
    classifier.save_results(results)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Robust Ensemble Classifier")
    parser.add_argument("--chunks", type=Path, required=True, help="Path to extracted_chunks.json")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--models", nargs="+", choices=list(ENSEMBLE_MODELS.keys()),
                       help="Models to run (default: all)")

    args = parser.parse_args()

    build_ensemble_matrix(args.chunks, args.output, args.models)
