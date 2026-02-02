#!/usr/bin/env python3
"""
Runner script for LLM-based whitepaper classification.

Usage:
    python scripts/run_llm_classifier.py                    # Uses local LM Studio
    python scripts/run_llm_classifier.py --provider openai --api-key sk-...
    python scripts/run_llm_classifier.py --provider anthropic --api-key sk-ant-...
    python scripts/run_llm_classifier.py --batch-size 30 --workers 10
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.llm_classifier import LLMClassifier


def main():
    parser = argparse.ArgumentParser(description="LLM-based whitepaper classification")
    parser.add_argument(
        "--provider",
        default="local",
        choices=["local", "openai", "anthropic"],
        help="LLM provider (default: local/LM Studio)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default varies by provider)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,  # Smaller batches for local models
        help="Chunks per API call (default: 15)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,  # Sequential for local to avoid overload
        help="Parallel workers (default: 1 for local)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit API key (for openai/anthropic)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:1234/v1",
        help="Base URL for local LLM (default: localhost:1234)"
    )
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    chunks_path = base_path / "outputs" / "nlp" / "extracted_chunks.json"
    output_dir = base_path / "outputs" / "nlp"

    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run pdf_extractor.py first.")
        sys.exit(1)

    # Get API key from environment if not explicitly provided
    api_key = args.api_key
    if not api_key and args.provider != "local":
        if args.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            print(f"ERROR: No API key found for {args.provider}")
            print(f"Set {args.provider.upper()}_API_KEY environment variable or use --api-key")
            sys.exit(1)

    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Parallel: {not args.sequential}")
    if args.provider == "local":
        print(f"Base URL: {args.base_url}")
    print()

    classifier = LLMClassifier(
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        max_workers=args.workers,
        api_key=api_key,
        base_url=args.base_url if args.provider == "local" else None
    )

    results = classifier.build_claims_matrix(
        chunks_path,
        output_dir,
        parallel=not args.sequential and args.workers > 1
    )

    print(f"\n[OK] Saved: outputs/nlp/claims_matrix_llm.npy")
    print(f"[OK] Saved: outputs/nlp/claims_matrix_llm.csv")
    print(f"[OK] Saved: outputs/nlp/claims_matrix_llm_meta.json")


if __name__ == "__main__":
    main()
