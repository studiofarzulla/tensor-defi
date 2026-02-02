"""NLP modules for TENSOR-DEFI."""
from .taxonomy import FUNCTIONAL_CATEGORIES, ZERO_SHOT_LABELS, LABEL_TO_CATEGORY
from .pdf_extractor import PDFExtractor
from .zero_shot_classifier import ZeroShotClassifier

__all__ = [
    'FUNCTIONAL_CATEGORIES',
    'ZERO_SHOT_LABELS',
    'LABEL_TO_CATEGORY',
    'PDFExtractor',
    'ZeroShotClassifier'
]
