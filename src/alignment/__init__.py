"""
Alignment Testing Module

Tests whether functional claims in whitepapers align with market factor structures
using tensor decomposition and statistical alignment measures.

Key methods:
- Tucker's Congruence Coefficient
- Orthogonal Procrustes Rotation
- Granger Causality Testing
"""

from .procrustes import orthogonal_procrustes, align_factor_matrices
from .congruence import tuckers_congruence, factor_similarity_matrix
from .alignment_test import AlignmentTest, AlignmentResult

__all__ = [
    'orthogonal_procrustes',
    'align_factor_matrices',
    'tuckers_congruence',
    'factor_similarity_matrix',
    'AlignmentTest',
    'AlignmentResult',
]
