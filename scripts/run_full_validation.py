#!/usr/bin/env python3
"""
Full validation pipeline for whitepaper-claims analysis.

Runs multiple classification methods, then validates convergence via:
1. Inter-method agreement (Fleiss' Kappa)
2. RMT analysis (Marchenko-Pastur)
3. Matrix alignment comparison

Usage:
    python scripts/run_full_validation.py [--skip-embedding] [--skip-local-llm]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_all_matrices(nlp_dir: Path) -> dict[str, np.ndarray]:
    """Load all available claims matrices."""
    matrices = {}

    # Known matrix files
    matrix_files = [
        ("llm", "claims_matrix_llm.npy"),
        ("embedding", "claims_matrix_embedding.npy"),
        ("local_llm", "claims_matrix_local_llm.npy"),
        ("ensemble_soft", "claims_matrix_ensemble_soft.npy"),
        ("ensemble_hard", "claims_matrix_ensemble_hard.npy"),
    ]

    # Also check checkpoints for BART/DeBERTa
    checkpoint_dir = nlp_dir / "ensemble_checkpoints"
    if checkpoint_dir.exists():
        for ckpt in ["checkpoint_bart.json", "checkpoint_deberta.json", "checkpoint_roberta.json"]:
            path = checkpoint_dir / ckpt
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                name = ckpt.replace("checkpoint_", "").replace(".json", "")
                # Convert scores dict to matrix
                symbols = data['symbols']
                categories = data['categories']
                scores = data['scores']  # {symbol: {category: score}}
                matrix = np.array([
                    [scores[sym][cat] for cat in categories]
                    for sym in symbols
                ])
                matrices[name] = matrix
                logger.info(f"Loaded {name} from checkpoint: {matrix.shape}")

    for name, fname in matrix_files:
        path = nlp_dir / fname
        if path.exists():
            matrices[name] = np.load(path)
            logger.info(f"Loaded {name}: {matrices[name].shape}")

    return matrices


def run_embedding_classifier(chunks_path: Path, output_dir: Path) -> np.ndarray:
    """Run Sentence Transformers embedding classifier."""
    from nlp.embedding_classifier import EmbeddingClassifier

    logger.info("Running Sentence Transformers classifier...")
    classifier = EmbeddingClassifier()
    results = classifier.build_claims_matrix(chunks_path, output_dir)
    return results['matrix']


def run_local_llm_classifier(chunks_path: Path, output_dir: Path) -> np.ndarray:
    """Run local LLM (Prometheus/LM Studio) classifier."""
    from nlp.llm_classifier import LLMClassifier

    logger.info("Running local LLM classifier (Prometheus)...")
    try:
        classifier = LLMClassifier(provider="local")
        results = classifier.build_claims_matrix(chunks_path, output_dir, parallel=False)

        # Save with different name
        np.save(output_dir / "claims_matrix_local_llm.npy", results['matrix'])

        return results['matrix']
    except Exception as e:
        logger.error(f"Local LLM failed (is LM Studio running?): {e}")
        return None


def compute_fleiss_kappa(matrices: dict[str, np.ndarray], threshold: float = 0.1) -> float:
    """
    Compute Fleiss' Kappa for inter-method agreement.

    Treats each (entity, category) as a "subject" and each method as a "rater".
    Binarizes scores at threshold.
    """
    if len(matrices) < 2:
        return None

    # Stack all matrices: (n_methods, n_entities, n_categories)
    names = list(matrices.keys())
    stacked = np.stack([matrices[n] for n in names])

    n_raters = len(names)
    n_entities, n_categories = stacked.shape[1:]
    n_subjects = n_entities * n_categories

    # Binarize: score > threshold = 1, else 0
    binary = (stacked > threshold).astype(int)

    # Reshape to (n_subjects, n_raters)
    binary = binary.reshape(n_raters, -1).T  # (n_subjects, n_raters)

    # Fleiss' Kappa
    # n_ij = count of raters assigning subject i to category j (here j in {0,1})
    n_1 = binary.sum(axis=1)  # Count of 1s per subject
    n_0 = n_raters - n_1  # Count of 0s per subject

    # P_i = proportion of agreeing pairs for subject i
    P_i = (n_0 * (n_0 - 1) + n_1 * (n_1 - 1)) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()

    # P_e = expected agreement by chance
    p_1 = n_1.sum() / (n_subjects * n_raters)  # Overall proportion of 1s
    p_0 = 1 - p_1
    P_e = p_0**2 + p_1**2

    kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0

    return kappa


def compute_pairwise_correlations(matrices: dict[str, np.ndarray]) -> dict:
    """Compute pairwise Pearson correlations between matrices."""
    names = list(matrices.keys())
    n = len(names)

    correlations = {}
    for i in range(n):
        for j in range(i+1, n):
            m1 = matrices[names[i]].flatten()
            m2 = matrices[names[j]].flatten()
            r = np.corrcoef(m1, m2)[0, 1]
            correlations[f"{names[i]}_vs_{names[j]}"] = r

    return correlations


def main():
    parser = argparse.ArgumentParser(description="Full validation pipeline")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding classifier")
    parser.add_argument("--skip-local-llm", action="store_true", help="Skip local LLM classifier")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap iterations (0=skip)")
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    nlp_dir = base_path / "outputs" / "nlp"
    stats_dir = base_path / "outputs" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        logger.error("Run pdf_extractor.py first")
        return

    # 1. Run classification methods
    if not args.skip_embedding:
        if not (nlp_dir / "claims_matrix_embedding.npy").exists():
            run_embedding_classifier(chunks_path, nlp_dir)
        else:
            logger.info("Embedding matrix exists, skipping...")

    if not args.skip_local_llm:
        if not (nlp_dir / "claims_matrix_local_llm.npy").exists():
            run_local_llm_classifier(chunks_path, nlp_dir)
        else:
            logger.info("Local LLM matrix exists, skipping...")

    # 2. Load all matrices
    matrices = load_all_matrices(nlp_dir)

    if len(matrices) < 2:
        logger.warning("Need at least 2 matrices for comparison")
        return

    # 3. RMT Analysis
    logger.info("\n" + "="*60)
    logger.info("RANDOM MATRIX THEORY ANALYSIS")
    logger.info("="*60)

    from stats.rmt import compare_matrices_rmt
    rmt_results = compare_matrices_rmt(matrices)

    # Save RMT results
    rmt_summary = {}
    for name, res in rmt_results.items():
        rmt_summary[name] = {
            'n_signal': res['mp']['n_signal'],
            'tw_p_value': res['tw']['p_value'],
            'tw_significant': res['tw']['significant'],
            'signal_variance_ratio': res['mp']['signal_variance_ratio']
        }

    with open(stats_dir / "rmt_analysis.json", 'w') as f:
        json.dump(rmt_summary, f, indent=2)

    # 4. Inter-method agreement
    logger.info("\n" + "="*60)
    logger.info("INTER-METHOD AGREEMENT")
    logger.info("="*60)

    kappa = compute_fleiss_kappa(matrices)
    correlations = compute_pairwise_correlations(matrices)

    logger.info(f"Fleiss' Kappa: {kappa:.3f}")
    logger.info("Interpretation: <0.2=poor, 0.2-0.4=fair, 0.4-0.6=moderate, 0.6-0.8=substantial, >0.8=almost perfect")

    logger.info("\nPairwise correlations:")
    for pair, r in sorted(correlations.items(), key=lambda x: -x[1]):
        logger.info(f"  {pair}: {r:.3f}")

    agreement_results = {
        'fleiss_kappa': kappa,
        'pairwise_correlations': correlations,
        'n_methods': len(matrices),
        'methods': list(matrices.keys())
    }

    with open(stats_dir / "agreement_analysis.json", 'w') as f:
        json.dump(agreement_results, f, indent=2)

    # 5. Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Methods compared: {len(matrices)} ({', '.join(matrices.keys())})")
    print(f"Fleiss' Kappa: {kappa:.3f}")
    print(f"Mean pairwise correlation: {np.mean(list(correlations.values())):.3f}")
    print(f"RMT signal factors (consensus): {max(r['mp']['n_signal'] for r in rmt_results.values())}")
    print("="*60)

    # 6. Optional bootstrap
    if args.bootstrap > 0:
        logger.info(f"\nRunning bootstrap with {args.bootstrap} iterations...")
        # Would run bootstrap here
        logger.info("Bootstrap not yet implemented in this runner")

    logger.info(f"\nResults saved to {stats_dir}")


if __name__ == "__main__":
    main()
