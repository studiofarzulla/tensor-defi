#!/usr/bin/env python3
"""
NLP Pipeline for TENSOR-DEFI

Runs PDF extraction and zero-shot classification to build claims matrix.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp.pdf_extractor import PDFExtractor
from nlp.zero_shot_classifier import ZeroShotClassifier


def main():
    base_path = Path(__file__).parent.parent

    print("="*60)
    print("TENSOR-DEFI: NLP Pipeline")
    print("="*60)

    # Step 1: Extract text from PDFs
    print("\n[1/2] Extracting text from whitepapers...")
    extractor = PDFExtractor(
        whitepaper_dir=base_path / "data" / "whitepapers",
        output_dir=base_path / "outputs" / "nlp"
    )

    metadata_path = base_path / "data" / "metadata" / "whitepaper_metadata.json"
    if metadata_path.exists():
        extractor.process_all(metadata_path)
    else:
        print("ERROR: Run data collection first")
        return

    # Step 2: Zero-shot classification
    print("\n[2/2] Running zero-shot classification...")
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    if chunks_path.exists():
        classifier = ZeroShotClassifier()
        classifier.build_claims_matrix(
            chunks_path=chunks_path,
            output_dir=base_path / "outputs" / "nlp"
        )
    else:
        print("ERROR: Text extraction failed")
        return

    print("\n" + "="*60)
    print("NLP PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
