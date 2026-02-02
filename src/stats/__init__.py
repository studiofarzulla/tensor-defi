"""Statistical validation modules for whitepaper-claims analysis."""

from .bootstrap import (
    bootstrap_document_profile,
    bootstrap_claims_matrix,
    bootstrap_alignment_statistic,
    save_bootstrap_results
)

from .rmt import (
    marchenko_pastur_bounds,
    marchenko_pastur_pdf,
    test_against_mp,
    tracy_widom_test,
    eigenvalue_ratio_test,
    full_rmt_analysis,
    compare_matrices_rmt
)

__all__ = [
    # Bootstrap
    'bootstrap_document_profile',
    'bootstrap_claims_matrix',
    'bootstrap_alignment_statistic',
    'save_bootstrap_results',
    # RMT
    'marchenko_pastur_bounds',
    'marchenko_pastur_pdf',
    'test_against_mp',
    'tracy_widom_test',
    'eigenvalue_ratio_test',
    'full_rmt_analysis',
    'compare_matrices_rmt',
]
