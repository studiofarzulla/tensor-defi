#!/usr/bin/env python3
"""
TENSOR-DEFI Chaos Expansion Orchestration Script

Runs the full expanded analysis pipeline:
1. NLP with original BART + 3-model ensemble
2. Taxonomy reduction (10 → 5 categories)
3. All alignment methods (Procrustes, CCA, RV coefficient)
4. Comparison across all combinations

Usage:
    python run_expansion_v2.py --start-from 1  # Run everything
    python run_expansion_v2.py --start-from 3  # Skip NLP, run alignment only
    python run_expansion_v2.py --skip-ensemble  # Skip GPU-intensive ensemble

Outputs:
    outputs/nlp/claims_matrix.npy              (original BART)
    outputs/nlp/claims_matrix_ensemble_*.npy   (3-model ensemble)
    outputs/nlp/claims_matrix_*_5cat.npy       (reduced taxonomy)
    outputs/alignment/alignment_comparison.json (method comparison)
    outputs/expansion/expansion_results.json   (full comparison table)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_original_nlp(base_path: Path):
    """Step 1a: Run original BART-only NLP pipeline."""
    from nlp.pdf_extractor import PDFExtractor
    from nlp.zero_shot_classifier import ZeroShotClassifier

    print("\n" + "="*70)
    print("STEP 1a: Original NLP Pipeline (BART-MNLI)")
    print("="*70)

    # Extract text
    extractor = PDFExtractor(
        whitepaper_dir=base_path / "data" / "whitepapers",
        output_dir=base_path / "outputs" / "nlp"
    )
    metadata_path = base_path / "data" / "metadata" / "whitepaper_metadata.json"
    extractor.process_all(metadata_path)

    # Classify
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    classifier = ZeroShotClassifier()
    classifier.build_claims_matrix(
        chunks_path=chunks_path,
        output_dir=base_path / "outputs" / "nlp"
    )


def run_ensemble_nlp(base_path: Path):
    """Step 1b: Run 3-model ensemble NLP pipeline."""
    from nlp.ensemble_classifier import EnsembleClassifier

    print("\n" + "="*70)
    print("STEP 1b: Ensemble NLP Pipeline (BART + DeBERTa + RoBERTa)")
    print("="*70)

    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    if not chunks_path.exists():
        logger.error("Run original NLP first to extract chunks")
        return None

    classifier = EnsembleClassifier()
    results = classifier.build_claims_matrices(
        chunks_path=chunks_path,
        output_dir=base_path / "outputs" / "nlp"
    )

    return results


def run_taxonomy_reduction(base_path: Path):
    """Step 2: Reduce 10-category matrices to 5-category."""
    from nlp.taxonomy_reduced import ReducedTaxonomyConverter, convert_ensemble_matrices

    print("\n" + "="*70)
    print("STEP 2: Taxonomy Reduction (10 → 5 categories)")
    print("="*70)

    nlp_dir = base_path / "outputs" / "nlp"
    converter = ReducedTaxonomyConverter()

    # Convert original
    claims_10_path = nlp_dir / "claims_matrix.npy"
    if claims_10_path.exists():
        converter.convert_matrix(
            input_path=claims_10_path,
            output_dir=nlp_dir,
            meta_path=nlp_dir / "claims_matrix_meta.json"
        )

    # Convert ensemble matrices
    convert_ensemble_matrices(base_path)


def run_alignment_comparison(base_path: Path):
    """Step 3: Run all alignment methods and compare."""
    from alignment.cca_alignment import ExtendedAlignmentTester

    print("\n" + "="*70)
    print("STEP 3: Alignment Method Comparison")
    print("="*70)

    # Load matrices
    nlp_dir = base_path / "outputs" / "nlp"
    market_dir = base_path / "outputs" / "market"
    tensor_dir = base_path / "outputs" / "tensor"

    # Original 10-category claims
    claims_10 = np.load(nlp_dir / "claims_matrix.npy")
    stats_matrix = np.load(market_dir / "stats_matrix.npy")
    factors = np.load(tensor_dir / "cp_asset_factors.npy")

    # Load metadata for alignment
    with open(nlp_dir / "claims_matrix_meta.json") as f:
        claims_symbols = json.load(f)['symbols']
    with open(market_dir / "stats_matrix_meta.json") as f:
        stats_symbols = json.load(f)['symbols']
    with open(tensor_dir / "cp_factors_meta.json") as f:
        factors_symbols = json.load(f)['symbols']

    # Find common symbols
    common = sorted(set(claims_symbols) & set(stats_symbols) & set(factors_symbols))
    logger.info(f"Common entities for alignment: {len(common)}")

    # Align matrices
    claims_idx = [claims_symbols.index(s) for s in common]
    stats_idx = [stats_symbols.index(s) for s in common]
    factors_idx = [factors_symbols.index(s) for s in common]

    claims_10 = claims_10[claims_idx]
    stats_matrix = stats_matrix[stats_idx]
    factors = factors[factors_idx]

    # Run comparison
    tester = ExtendedAlignmentTester(output_dir=base_path / "outputs" / "alignment")
    results = tester.run_comparison(claims_10, stats_matrix, factors)
    tester.save_comparison(results)

    return results, common


def run_full_comparison(base_path: Path):
    """Step 4: Compare all NLP × taxonomy × alignment combinations."""
    print("\n" + "="*70)
    print("STEP 4: Full Combination Comparison")
    print("="*70)

    nlp_dir = base_path / "outputs" / "nlp"
    market_dir = base_path / "outputs" / "market"
    tensor_dir = base_path / "outputs" / "tensor"
    output_dir = base_path / "outputs" / "expansion"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all available matrices
    matrices = {}

    # Original BART
    if (nlp_dir / "claims_matrix.npy").exists():
        matrices['bart_10'] = np.load(nlp_dir / "claims_matrix.npy")
    if (nlp_dir / "claims_matrix_5cat.npy").exists():
        matrices['bart_5'] = np.load(nlp_dir / "claims_matrix_5cat.npy")

    # Ensemble soft
    if (nlp_dir / "claims_matrix_ensemble_soft.npy").exists():
        matrices['ensemble_soft_10'] = np.load(nlp_dir / "claims_matrix_ensemble_soft.npy")
    if (nlp_dir / "claims_matrix_ensemble_soft_5cat.npy").exists():
        matrices['ensemble_soft_5'] = np.load(nlp_dir / "claims_matrix_ensemble_soft_5cat.npy")

    # Ensemble hard
    if (nlp_dir / "claims_matrix_ensemble_hard.npy").exists():
        matrices['ensemble_hard_10'] = np.load(nlp_dir / "claims_matrix_ensemble_hard.npy")
    if (nlp_dir / "claims_matrix_ensemble_hard_5cat.npy").exists():
        matrices['ensemble_hard_5'] = np.load(nlp_dir / "claims_matrix_ensemble_hard_5cat.npy")

    # Market matrices
    stats_matrix = np.load(market_dir / "stats_matrix.npy")
    factors = np.load(tensor_dir / "cp_asset_factors.npy")

    # Alignment methods
    from alignment.congruence import CongruenceCoefficient
    from alignment.cca_alignment import CCAAlignment, RVCoefficient

    procrustes = CongruenceCoefficient()
    cca = CCAAlignment()
    rv = RVCoefficient()

    # Get common symbols for alignment
    with open(nlp_dir / "claims_matrix_meta.json") as f:
        claims_symbols = json.load(f)['symbols']
    with open(market_dir / "stats_matrix_meta.json") as f:
        stats_symbols = json.load(f)['symbols']
    with open(tensor_dir / "cp_factors_meta.json") as f:
        factors_symbols = json.load(f)['symbols']

    common = sorted(set(claims_symbols) & set(stats_symbols) & set(factors_symbols))

    # Align all matrices to common symbols
    claims_idx = [claims_symbols.index(s) for s in common]
    stats_idx = [stats_symbols.index(s) for s in common]
    factors_idx = [factors_symbols.index(s) for s in common]

    for key in matrices:
        matrices[key] = matrices[key][claims_idx]
    stats_matrix = stats_matrix[stats_idx]
    factors = factors[factors_idx]

    # Run all combinations
    results = {}

    for nlp_key, claims in matrices.items():
        logger.info(f"\nTesting {nlp_key}...")

        # Claims ↔ Factors alignment
        pair_key = f"{nlp_key}_factors"

        try:
            # Procrustes
            proc_result = procrustes.matrix_congruence(claims, factors)
            phi = proc_result['mean_phi']

            # CCA
            cca_result = cca.fit_transform(claims, factors)
            rho = cca_result['rho_1']

            # RV
            rv_result = rv.compute(claims, factors)
            rv_coef = rv_result['rv_coefficient']

            results[pair_key] = {
                'nlp_method': nlp_key.split('_')[0],
                'taxonomy': '5cat' if '_5' in nlp_key else '10cat',
                'procrustes_phi': float(phi),
                'cca_rho1': float(rho),
                'rv_coefficient': float(rv_coef),
                'n_entities': len(common)
            }

        except Exception as e:
            logger.error(f"Failed {pair_key}: {e}")

    # Save results
    with open(output_dir / "expansion_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    _print_expansion_table(results)

    return results


def _print_expansion_table(results: dict):
    """Print formatted comparison table."""
    print(f"\n{'='*90}")
    print("EXPANSION COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"{'NLP Method':<20} {'Taxonomy':<10} {'Procrustes φ':>15} {'CCA ρ₁':>12} {'RV':>10}")
    print(f"{'-'*90}")

    # Sort by Procrustes phi
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]['procrustes_phi'], reverse=True)

    for key in sorted_keys:
        r = results[key]
        nlp = r['nlp_method']
        tax = r['taxonomy']
        phi = r['procrustes_phi']
        rho = r['cca_rho1']
        rv = r['rv_coefficient']
        print(f"{nlp:<20} {tax:<10} {phi:>15.3f} {rho:>12.3f} {rv:>10.3f}")

    print(f"{'='*90}")

    # Find best configuration
    best_key = max(results.keys(), key=lambda k: results[k]['procrustes_phi'])
    best = results[best_key]
    print(f"\nBest configuration: {best['nlp_method']} + {best['taxonomy']}")
    print(f"  Procrustes φ = {best['procrustes_phi']:.3f}")
    print(f"  CCA ρ₁ = {best['cca_rho1']:.3f}")
    print(f"  RV = {best['rv_coefficient']:.3f}")
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="TENSOR-DEFI Expansion Pipeline")
    parser.add_argument(
        '--start-from', type=int, default=1,
        help='Start from step (1=NLP, 2=taxonomy, 3=alignment, 4=comparison)'
    )
    parser.add_argument(
        '--skip-ensemble', action='store_true',
        help='Skip GPU-intensive 3-model ensemble'
    )
    parser.add_argument(
        '--skip-original-nlp', action='store_true',
        help='Skip original BART NLP (assume already run)'
    )
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent

    print("="*70)
    print("TENSOR-DEFI CHAOS EXPANSION")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)

    try:
        if args.start_from <= 1:
            if not args.skip_original_nlp:
                run_original_nlp(base_path)
            if not args.skip_ensemble:
                run_ensemble_nlp(base_path)

        if args.start_from <= 2:
            run_taxonomy_reduction(base_path)

        if args.start_from <= 3:
            run_alignment_comparison(base_path)

        if args.start_from <= 4:
            run_full_comparison(base_path)

        print("\n" + "="*70)
        print("EXPANSION COMPLETE")
        print(f"Finished: {datetime.now().isoformat()}")
        print("="*70)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
