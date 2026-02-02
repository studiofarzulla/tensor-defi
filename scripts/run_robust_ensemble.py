#!/usr/bin/env python3
"""
Run robust ensemble classifier with checkpointing.

Usage:
    python run_robust_ensemble.py                    # Run all 3 models
    python run_robust_ensemble.py --models bart      # Run just BART
    python run_robust_ensemble.py --resume           # Resume from checkpoints
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp.ensemble_classifier_robust import build_ensemble_matrix, ENSEMBLE_MODELS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Robust Ensemble Classifier")
    parser.add_argument("--models", nargs="+", choices=list(ENSEMBLE_MODELS.keys()),
                       default=list(ENSEMBLE_MODELS.keys()),
                       help="Models to run (default: all)")

    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    output_dir = base_path / "outputs" / "nlp"

    print(f"Chunks: {chunks_path}")
    print(f"Output: {output_dir}")
    print(f"Models: {args.models}")
    print()

    results = build_ensemble_matrix(chunks_path, output_dir, args.models)

    print("\n" + "="*60)
    print("ENSEMBLE COMPLETE")
    print("="*60)
    print(f"Soft matrix shape: {results['soft_matrix'].shape}")
    print(f"Models used: {results['models_used']}")
    print(f"Mean agreement: {results['agreement_matrix'].mean():.3f}")
