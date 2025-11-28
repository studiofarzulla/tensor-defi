"""
Whitepaper Collection Script

Downloads whitepapers for top 20 crypto projects.
For PoC, focuses on projects with direct PDF links.

Usage:
    python scripts/collect_whitepapers.py
"""

import os
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

# Known direct PDF links for whitepapers
# These are the official/commonly referenced versions
WHITEPAPER_URLS = {
    "BTC": {
        "name": "Bitcoin",
        "url": "https://bitcoin.org/bitcoin.pdf",
        "filename": "bitcoin.pdf"
    },
    "ETH": {
        "name": "Ethereum",
        "url": "https://ethereum.org/content/whitepaper/whitepaper-pdf/Ethereum_Whitepaper_-_Buterin_2014.pdf",
        "filename": "ethereum.pdf"
    },
    "SOL": {
        "name": "Solana",
        "url": "https://solana.com/solana-whitepaper.pdf",
        "filename": "solana.pdf"
    },
    "AVAX": {
        "name": "Avalanche",
        "url": "https://assets.website-files.com/5d80307810123f5ffbb34d6e/6008d7bbf8b10d1eb01e7e16_Avalanche%20Platform%20Whitepaper.pdf",
        "filename": "avalanche.pdf"
    },
    "DOT": {
        "name": "Polkadot",
        "url": "https://polkadot.network/PolkaDotPaper.pdf",
        "filename": "polkadot.pdf"
    },
    "ATOM": {
        "name": "Cosmos",
        "url": "https://v1.cosmos.network/resources/whitepaper",
        "filename": "cosmos.pdf",
        "note": "HTML page, not direct PDF"
    },
    "FIL": {
        "name": "Filecoin",
        "url": "https://filecoin.io/filecoin.pdf",
        "filename": "filecoin.pdf"
    },
    "LINK": {
        "name": "Chainlink",
        "url": "https://research.chain.link/whitepaper-v1.pdf",
        "filename": "chainlink.pdf"
    },
    "UNI": {
        "name": "Uniswap",
        "url": "https://uniswap.org/whitepaper.pdf",
        "filename": "uniswap_v1.pdf"
    },
    "ALGO": {
        "name": "Algorand",
        "url": "https://algorandcom.cdn.prismic.io/algorandcom%2Fece77f38-75b3-44de-bc7f-805f0e53a8d9_theoretical.pdf",
        "filename": "algorand.pdf"
    },
    "NEAR": {
        "name": "NEAR Protocol",
        "url": "https://near.org/papers/the-official-near-white-paper/",
        "filename": "near.pdf",
        "note": "HTML page"
    },
    "ADA": {
        "name": "Cardano",
        "url": "https://docs.cardano.org/introduction/",
        "filename": "cardano.pdf",
        "note": "Documentation site"
    },
}

# Headers to mimic browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}


def download_pdf(url: str, output_path: Path, timeout: int = 30) -> bool:
    """
    Download a PDF from URL.
    
    Returns True if successful, False otherwise.
    """
    try:
        print(f"  Downloading from {url[:60]}...")
        response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 200:
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '')
            
            if 'pdf' in content_type.lower() or url.endswith('.pdf'):
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                file_size = output_path.stat().st_size / 1024  # KB
                print(f"  ✓ Saved: {output_path.name} ({file_size:.1f} KB)")
                return True
            else:
                print(f"  ✗ Not a PDF (content-type: {content_type})")
                return False
        else:
            print(f"  ✗ HTTP {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("WHITEPAPER COLLECTION FOR TOP 20 CRYPTO PROJECTS")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("data/whitepapers")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Track results
    successful = []
    failed = []
    skipped = []
    
    print(f"\nAttempting to download {len(WHITEPAPER_URLS)} whitepapers...\n")
    
    for symbol, info in WHITEPAPER_URLS.items():
        print(f"\n[{symbol}] {info['name']}")
        
        output_path = output_dir / info['filename']
        
        # Check if already exists
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024
            print(f"  ✓ Already exists ({file_size:.1f} KB)")
            successful.append(symbol)
            continue
        
        # Skip non-PDF sources
        if info.get('note'):
            print(f"  ⚠ Skipping: {info['note']}")
            skipped.append(symbol)
            continue
        
        # Download
        if download_pdf(info['url'], output_path):
            successful.append(symbol)
        else:
            failed.append(symbol)
        
        # Rate limit
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"\n✓ Successful: {len(successful)}")
    for s in successful:
        print(f"    {s}: {WHITEPAPER_URLS[s]['name']}")
    
    if skipped:
        print(f"\n⚠ Skipped (non-PDF): {len(skipped)}")
        for s in skipped:
            print(f"    {s}: {WHITEPAPER_URLS[s]['name']}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for s in failed:
            print(f"    {s}: {WHITEPAPER_URLS[s]['name']}")
    
    # List collected files
    print("\n" + "=" * 60)
    print("COLLECTED FILES")
    print("=" * 60)
    
    pdf_files = list(output_dir.glob("*.pdf"))
    total_size = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)  # MB
    
    print(f"\nTotal: {len(pdf_files)} PDFs ({total_size:.2f} MB)")
    for pdf in sorted(pdf_files):
        size = pdf.stat().st_size / 1024
        print(f"  {pdf.name}: {size:.1f} KB")
    
    print("\n✓ Collection complete!")
    print(f"\nNext step: Test extraction with:")
    print(f"  python -c \"from src.nlp import WhitepaperCollector; c = WhitepaperCollector(); print(c.collect_from_directory())\"")


if __name__ == "__main__":
    main()
