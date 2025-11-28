"""
NLP Pipeline - End-to-End Whitepaper Processing

Processes all collected whitepapers through:
1. Text extraction and chunking
2. Functional claim classification
3. Functional profile generation

Usage:
    python scripts/run_nlp_pipeline.py
"""

import sys
from pathlib import Path
import json
import pickle
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.nlp.whitepaper_collector import WhitepaperCollector
from src.nlp.text_processor import TextProcessor
from src.nlp.claim_extractor import ClaimExtractor, DocumentClaims
from src.nlp.taxonomy import FunctionalCategory, FUNCTIONAL_TAXONOMY


def process_all_whitepapers(
    use_zero_shot: bool = False,
    chunk_size: int = 500,
    output_dir: str = "outputs/nlp"
) -> dict:
    """
    Process all whitepapers through the NLP pipeline.
    
    Args:
        use_zero_shot: Use BART MNLI (slow) or keyword fallback (fast)
        chunk_size: Text chunk size for processing
        output_dir: Directory for output files
    
    Returns:
        Dict with all processing results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("NLP PIPELINE: WHITEPAPER → FUNCTIONAL CLAIMS")
    print("=" * 60)
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"Classification method: {'Zero-shot (BART MNLI)' if use_zero_shot else 'Keyword matching (fast)'}")
    
    # Step 1: Collect whitepapers
    print("\n" + "=" * 60)
    print("STEP 1: LOADING WHITEPAPERS")
    print("=" * 60)
    
    collector = WhitepaperCollector()
    whitepapers = collector.collect_from_directory()
    
    if not whitepapers:
        print("✗ No whitepapers found! Run: python scripts/collect_whitepapers.py")
        return {}
    
    print(f"\n✓ Loaded {len(whitepapers)} whitepapers")
    
    # Step 2: Process text
    print("\n" + "=" * 60)
    print("STEP 2: TEXT PROCESSING")
    print("=" * 60)
    
    processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=50)
    processed_docs = []
    
    for wp in whitepapers:
        doc = processor.process_document(
            wp.raw_text,
            wp.metadata.project_symbol,
            wp.metadata.project_name
        )
        processed_docs.append(doc)
        print(f"  {doc.project_symbol}: {doc.total_words} words → {doc.num_chunks} chunks")
    
    total_chunks = sum(d.num_chunks for d in processed_docs)
    print(f"\n✓ Total chunks: {total_chunks}")
    
    # Step 3: Extract claims
    print("\n" + "=" * 60)
    print("STEP 3: FUNCTIONAL CLAIM EXTRACTION")
    print("=" * 60)
    
    # Initialize extractor
    if use_zero_shot:
        print("\nLoading BART MNLI classifier (this may take a minute)...")
    
    extractor = ClaimExtractor(
        model="facebook/bart-large-mnli",
        device=-1,  # CPU
        use_keywords_fallback=True
    )
    
    # Process each document
    all_claims = {}
    
    for doc in processed_docs:
        print(f"\n[{doc.project_symbol}] {doc.project_name}")
        
        # Get text chunks
        chunk_texts = [c.text for c in doc.chunks]
        
        # Extract claims
        claims = extractor.extract_document_claims(
            chunks=chunk_texts,
            project_symbol=doc.project_symbol,
            project_name=doc.project_name,
            threshold=0.2,
            show_progress=False
        )
        
        all_claims[doc.project_symbol] = claims
        
        # Show top categories
        primary = claims.primary_functions
        if primary:
            cats_str = ", ".join(c.value for c in primary[:3])
            print(f"  Primary functions: {cats_str}")
        else:
            print(f"  Primary functions: None detected")
    
    # Step 4: Generate summary
    print("\n" + "=" * 60)
    print("STEP 4: FUNCTIONAL PROFILE SUMMARY")
    print("=" * 60)
    
    # Create summary table
    print("\n{:<8} {:<15} {:<45}".format("Symbol", "Words", "Top Functions"))
    print("-" * 70)
    
    for symbol, claims in all_claims.items():
        # Get top functions
        sorted_profile = sorted(
            claims.functional_profile.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_funcs = [f"{cat.value[:15]}({score:.2f})" for cat, score in sorted_profile[:3] if score > 0.1]
        top_funcs_str = ", ".join(top_funcs) if top_funcs else "None"
        
        # Get word count
        wp = next((w for w in whitepapers if w.metadata.project_symbol == symbol), None)
        words = wp.metadata.word_count if wp else 0
        
        print(f"{symbol:<8} {words:<15} {top_funcs_str:<45}")
    
    # Step 5: Save results
    print("\n" + "=" * 60)
    print("STEP 5: SAVING RESULTS")
    print("=" * 60)
    
    # Save functional profiles as JSON
    profiles_json = {}
    for symbol, claims in all_claims.items():
        profiles_json[symbol] = {
            "project_name": claims.project_name,
            "num_chunks": claims.num_claims,
            "primary_functions": [c.value for c in claims.primary_functions],
            "functional_profile": {
                cat.value: float(score) 
                for cat, score in claims.functional_profile.items()
            },
            "feature_vector": claims.to_vector().tolist()
        }
    
    json_path = output_path / "functional_profiles.json"
    with open(json_path, 'w') as f:
        json.dump(profiles_json, f, indent=2)
    print(f"✓ Saved functional profiles: {json_path}")
    
    # Save claims tensor data
    # Shape: (projects × categories)
    symbols = list(all_claims.keys())
    categories = list(FunctionalCategory)
    
    claims_matrix = np.array([
        all_claims[s].to_vector() for s in symbols
    ])
    
    tensor_path = output_path / "claims_matrix.npz"
    np.savez(
        tensor_path,
        matrix=claims_matrix,
        symbols=np.array(symbols),
        categories=np.array([c.value for c in categories])
    )
    print(f"✓ Saved claims matrix: {tensor_path}")
    print(f"  Shape: {claims_matrix.shape} (projects × categories)")
    
    # Save full results as pickle
    results = {
        "timestamp": datetime.now().isoformat(),
        "whitepapers": [(w.metadata.project_symbol, w.metadata.word_count) for w in whitepapers],
        "processed_docs": [(d.project_symbol, d.num_chunks) for d in processed_docs],
        "claims": all_claims,
        "claims_matrix": claims_matrix,
        "symbols": symbols,
        "categories": [c.value for c in categories],
    }
    
    pickle_path = output_path / "nlp_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved full results: {pickle_path}")
    
    # Print functional taxonomy coverage
    print("\n" + "=" * 60)
    print("FUNCTIONAL TAXONOMY COVERAGE")
    print("=" * 60)
    
    # Average score per category across all projects
    avg_scores = claims_matrix.mean(axis=0)
    
    print("\n{:<30} {:<10} {:<10}".format("Category", "Avg Score", "Max Score"))
    print("-" * 50)
    
    for i, cat in enumerate(categories):
        avg = avg_scores[i]
        max_score = claims_matrix[:, i].max()
        bar = "█" * int(avg * 20)
        print(f"{cat.value:<30} {avg:.3f}      {max_score:.3f}  {bar}")
    
    print("\n" + "=" * 60)
    print("✓ NLP PIPELINE COMPLETE!")
    print("=" * 60)
    
    print(f"\nOutput files:")
    print(f"  - {json_path}")
    print(f"  - {tensor_path}")
    print(f"  - {pickle_path}")
    
    print(f"\nNext step: Build claims tensor for alignment testing")
    print(f"  python -c \"import numpy as np; d=np.load('{tensor_path}'); print('Claims matrix:', d['matrix'].shape)\"")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NLP pipeline on whitepapers")
    parser.add_argument("--zero-shot", action="store_true", 
                        help="Use BART MNLI zero-shot classification (slow)")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Text chunk size (default: 500)")
    args = parser.parse_args()
    
    results = process_all_whitepapers(
        use_zero_shot=args.zero_shot,
        chunk_size=args.chunk_size
    )
