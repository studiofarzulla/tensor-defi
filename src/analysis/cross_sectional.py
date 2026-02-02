"""Cross-sectional analysis for TENSOR-DEFI."""

import logging
from pathlib import Path
import numpy as np
import json

logger = logging.getLogger(__name__)


class CrossSectionalAnalyzer:
    """Analyze entity-level alignment patterns."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_entity_alignment(self, claims_matrix: np.ndarray, stats_matrix: np.ndarray, symbols: list[str]) -> dict:
        """Compute alignment contribution per entity using leave-one-out."""
        import sys
        sys.path.insert(0, str(self.output_dir.parent.parent / "src"))
        from alignment.procrustes import ProcrustesAlignment
        from alignment.congruence import CongruenceCoefficient

        aligner = ProcrustesAlignment()
        congruence = CongruenceCoefficient()
        n = len(symbols)

        # Full alignment
        result_full = aligner.align_matrices(claims_matrix, stats_matrix)
        aligned_full = result_full['source_rotated']
        target_full = result_full['target_centered']
        phi_full = congruence.matrix_congruence(aligned_full, target_full)['mean_phi']

        # Leave-one-out
        entity_impact = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            result_loo = aligner.align_matrices(claims_matrix[mask], stats_matrix[mask])
            aligned_loo = result_loo['source_rotated']
            target_loo = result_loo['target_centered']
            phi_loo = congruence.matrix_congruence(aligned_loo, target_loo)['mean_phi']

            impact = float(phi_full - phi_loo)
            entity_impact.append({
                'symbol': symbols[i],
                'phi_without': float(phi_loo),
                'impact': impact,
                'interpretation': 'helps' if impact > 0.01 else 'hurts' if impact < -0.01 else 'neutral'
            })

        entity_impact.sort(key=lambda x: x['impact'], reverse=True)

        return {
            'phi_full': float(phi_full),
            'entity_analysis': entity_impact,
            'best_aligned': [e['symbol'] for e in entity_impact if e['impact'] > 0.01],
            'worst_aligned': [e['symbol'] for e in entity_impact if e['impact'] < -0.01],
        }

    def cluster_entities(self, claims_matrix: np.ndarray, stats_matrix: np.ndarray, symbols: list[str], n_clusters: int = 3) -> dict:
        """Cluster entities by alignment residuals."""
        from sklearn.cluster import KMeans
        import sys
        sys.path.insert(0, str(self.output_dir.parent.parent / "src"))
        from alignment.procrustes import ProcrustesAlignment

        aligner = ProcrustesAlignment()
        result = aligner.align_matrices(claims_matrix, stats_matrix)
        residuals = result['source_rotated'] - result['target_centered']

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(residuals)

        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(symbols[i])

        return {'n_clusters': n_clusters, 'clusters': {int(k): v for k, v in clusters.items()}}

    def save_results(self, results: dict):
        output_path = self.output_dir / "cross_sectional_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {output_path}")

        print(f"\n{'='*60}")
        print("CROSS-SECTIONAL ANALYSIS RESULTS")
        print(f"{'='*60}")
        if 'entity_analysis' in results:
            print(f"\nEntity impact on alignment:")
            for e in results['entity_analysis']:
                print(f"  {e['symbol']:6s}: impact = {e['impact']:+.3f} ({e['interpretation']})")
