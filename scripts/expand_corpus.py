#!/usr/bin/env python3
"""
Whitepaper Corpus Expansion Script

Downloads additional cryptocurrency whitepapers from verified sources
to expand the corpus for more robust alignment analysis.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Verified whitepaper sources - manually curated for accuracy
WHITEPAPER_SOURCES = [
    # === EXISTING (keep these) ===
    {
        "symbol": "BTC",
        "name": "Bitcoin",
        "url": "https://bitcoin.org/bitcoin.pdf",
        "type": "original_whitepaper"
    },
    {
        "symbol": "ETH",
        "name": "Ethereum",
        "url": "https://ethereum.org/content/whitepaper/whitepaper-pdf/Ethereum_Whitepaper_-_Buterin_2014.pdf",
        "type": "original_whitepaper"
    },
    {
        "symbol": "SOL",
        "name": "Solana",
        "url": "https://solana.com/solana-whitepaper.pdf",
        "type": "original_whitepaper"
    },
    {
        "symbol": "AVAX",
        "name": "Avalanche",
        "url": "https://assets.website-files.com/5d80307810123f5ffbb34d6e/6008d7bbf8b10d1eb01e7e16_Avalanche%20Platform%20Whitepaper.pdf",
        "type": "consensus_paper"
    },
    {
        "symbol": "ICP",
        "name": "Internet Computer",
        "url": "https://internetcomputer.org/whitepaper.pdf",
        "type": "technical_overview"
    },
    {
        "symbol": "FIL",
        "name": "Filecoin",
        "url": "https://filecoin.io/filecoin.pdf",
        "type": "technical_report"
    },
    {
        "symbol": "AR",
        "name": "Arweave",
        "url": "https://www.arweave.org/yellow-paper.pdf",
        "type": "yellowpaper"
    },
    {
        "symbol": "AAVE",
        "name": "Aave",
        "url": "https://raw.githubusercontent.com/aave/aave-protocol/master/docs/Aave_Protocol_Whitepaper_v1_0.pdf",
        "type": "protocol_whitepaper"
    },

    # === NEW ADDITIONS ===
    # Layer 1s
    {
        "symbol": "DOT",
        "name": "Polkadot",
        "url": "https://polkadot.network/whitepaper/polkadot-whitepaper.pdf",
        "type": "original_whitepaper"
    },
    {
        "symbol": "ATOM",
        "name": "Cosmos",
        "url": "https://v1.cosmos.network/resources/whitepaper",  # Will need special handling
        "type": "original_whitepaper",
        "direct_pdf": "https://github.com/cosmos/cosmos/raw/master/WHITEPAPER.md",
        "fallback_md": True
    },
    {
        "symbol": "ALGO",
        "name": "Algorand",
        "url": "https://arxiv.org/pdf/1607.01341.pdf",  # Original Algorand paper by Silvio Micali
        "type": "academic_paper"
    },
    {
        "symbol": "ADA",
        "name": "Cardano",
        "url": "https://docs.cardano.org/assets/files/why-cardano.pdf",
        "type": "philosophy_paper"
    },
    {
        "symbol": "NEAR",
        "name": "NEAR Protocol",
        "url": "https://pages.near.org/papers/nightshade.pdf",
        "type": "sharding_paper"
    },
    {
        "symbol": "XMR",
        "name": "Monero",
        "url": "https://github.com/monero-project/research-lab/raw/master/whitepaper/whitepaper.pdf",
        "type": "cryptonote_whitepaper"
    },
    {
        "symbol": "ZEC",
        "name": "Zcash",
        "url": "https://raw.githubusercontent.com/zcash/zips/main/protocol/protocol.pdf",
        "type": "protocol_spec"
    },

    # DeFi Protocols
    {
        "symbol": "UNI",
        "name": "Uniswap",
        "url": "https://uniswap.org/whitepaper-v3.pdf",
        "type": "amm_whitepaper"
    },
    {
        "symbol": "MKR",
        "name": "Maker",
        "url": "https://makerdao.com/en/whitepaper/",  # HTML, may need special handling
        "direct_pdf": "https://makerdao.com/whitepaper/Dai-Whitepaper-Dec17-en.pdf",
        "type": "stablecoin_whitepaper"
    },
    {
        "symbol": "COMP",
        "name": "Compound",
        "url": "https://compound.finance/documents/Compound.Whitepaper.pdf",
        "type": "lending_whitepaper"
    },
    {
        "symbol": "CRV",
        "name": "Curve",
        "url": "https://curve.fi/whitepaper",
        "direct_pdf": "https://resources.curve.fi/crvUSD/curve-stablecoin.pdf",
        "type": "stableswap_whitepaper"
    },
    {
        "symbol": "SNX",
        "name": "Synthetix",
        "url": "https://docs.synthetix.io/synthetix-protocol/the-synthetix-protocol/synthetix-litepaper",
        "type": "litepaper"
    },

    # Infrastructure & Oracles
    {
        "symbol": "LINK",
        "name": "Chainlink",
        "url": "https://research.chain.link/whitepaper-v1.pdf",
        "type": "oracle_whitepaper"
    },
    {
        "symbol": "GRT",
        "name": "The Graph",
        "url": "https://thegraph.com/docs/en/about/",
        "direct_pdf": "https://github.com/graphprotocol/research/raw/master/papers/the-graph-whitepaper.pdf",
        "type": "indexing_whitepaper"
    },

    # Layer 2s
    {
        "symbol": "OP",
        "name": "Optimism",
        "url": "https://community.optimism.io/docs/protocol/",
        "type": "rollup_docs"  # No traditional whitepaper, will use docs
    },
    {
        "symbol": "ARB",
        "name": "Arbitrum",
        "url": "https://github.com/OffchainLabs/arbitrum/blob/master/docs/Arbitrum_Nitro_Whitepaper.pdf",
        "direct_pdf": "https://raw.githubusercontent.com/OffchainLabs/arbitrum/master/docs/Arbitrum_Nitro_Whitepaper.pdf",
        "type": "rollup_whitepaper"
    },
    {
        "symbol": "POL",
        "name": "Polygon",
        "url": "https://polygon.technology/papers/pol-whitepaper",
        "direct_pdf": "https://polygon.technology/papers/pol-whitepaper.pdf",
        "type": "token_whitepaper"
    },

    # Storage & Compute
    {
        "symbol": "STORJ",
        "name": "Storj",
        "url": "https://www.storj.io/storj.pdf",
        "type": "storage_whitepaper"
    },
    {
        "symbol": "SC",
        "name": "Siacoin",
        "url": "https://sia.tech/sia.pdf",
        "type": "storage_whitepaper"
    },
    {
        "symbol": "RENDER",
        "name": "Render Network",
        "url": "https://renderfoundation.com/whitepaper",
        "type": "compute_whitepaper"
    },

    # Gaming/Metaverse (have market data, worth including)
    {
        "symbol": "IMX",
        "name": "Immutable X",
        "url": "https://www.immutable.com/whitepaper",
        "type": "gaming_l2_whitepaper"
    },
]


@dataclass
class DownloadResult:
    symbol: str
    success: bool
    path: str | None
    error: str | None
    pages: int
    doc_type: str


class WhitepaperDownloader:
    """Downloads whitepapers from verified sources."""

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/pdf,*/*'
    }

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_pdf(self, url: str, symbol: str) -> tuple[bool, str | None, str | None]:
        """Download a PDF from URL."""
        try:
            # Handle direct_pdf override
            response = requests.get(url, headers=self.HEADERS, timeout=30, allow_redirects=True)
            response.raise_for_status()

            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not response.content[:4] == b'%PDF':
                return False, None, f"Not a PDF (content-type: {content_type})"

            # Save
            filename = f"{symbol.lower()}_whitepaper.pdf"
            filepath = self.output_dir / filename
            filepath.write_bytes(response.content)

            return True, str(filepath), None

        except requests.exceptions.RequestException as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, str(e)

    def download_markdown(self, url: str, symbol: str) -> tuple[bool, str | None, str | None]:
        """Download markdown as fallback."""
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=30)
            response.raise_for_status()

            filename = f"{symbol.lower()}_readme.md"
            filepath = self.output_dir / filename
            filepath.write_text(response.text, encoding='utf-8')

            return True, str(filepath), None

        except Exception as e:
            return False, None, str(e)

    def get_pdf_pages(self, filepath: str) -> int:
        """Count pages in PDF."""
        try:
            import fitz
            doc = fitz.open(filepath)
            pages = len(doc)
            doc.close()
            return pages
        except:
            return 0

    def download_source(self, source: dict) -> DownloadResult:
        """Download a single whitepaper source."""
        symbol = source['symbol']
        doc_type = source.get('type', 'whitepaper')

        # Try direct_pdf first if available
        url = source.get('direct_pdf', source['url'])

        logger.info(f"Downloading {symbol} from {url[:60]}...")

        # Try PDF download
        success, path, error = self.download_pdf(url, symbol)

        # If PDF failed and markdown fallback is available
        if not success and source.get('fallback_md'):
            md_url = source.get('direct_pdf', source['url'])
            if md_url.endswith('.md'):
                success, path, error = self.download_markdown(md_url, symbol)

        pages = self.get_pdf_pages(path) if success and path and path.endswith('.pdf') else 1

        return DownloadResult(
            symbol=symbol,
            success=success,
            path=path,
            error=error,
            pages=pages,
            doc_type=doc_type
        )

    def download_all(self, sources: list[dict], skip_existing: bool = True) -> list[DownloadResult]:
        """Download all whitepapers."""
        results = []

        for source in sources:
            symbol = source['symbol']

            # Check if already exists
            if skip_existing:
                existing = list(self.output_dir.glob(f"{symbol.lower()}_*"))
                if existing:
                    logger.info(f"Skipping {symbol} (already exists)")
                    results.append(DownloadResult(
                        symbol=symbol,
                        success=True,
                        path=str(existing[0]),
                        error=None,
                        pages=self.get_pdf_pages(str(existing[0])) if str(existing[0]).endswith('.pdf') else 1,
                        doc_type=source.get('type', 'whitepaper')
                    ))
                    continue

            result = self.download_source(source)
            results.append(result)

            # Rate limit
            time.sleep(1)

        return results


def update_metadata(results: list[DownloadResult], metadata_path: Path):
    """Update whitepaper_metadata.json with new entries."""
    # Load existing
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = []

    # Get existing symbols
    existing_symbols = {m['symbol'] for m in metadata}

    # Get source info for new entries
    source_by_symbol = {s['symbol']: s for s in WHITEPAPER_SOURCES}

    # Update/add entries
    for result in results:
        if not result.success:
            continue

        source = source_by_symbol.get(result.symbol, {})

        entry = {
            "symbol": result.symbol,
            "name": source.get('name', result.symbol),
            "url": source.get('url', ''),
            "local_path": result.path,
            "pages": result.pages,
            "download_date": datetime.now().isoformat(),
            "fallback_type": None if result.path.endswith('.pdf') else 'markdown',
            "doc_type": result.doc_type
        }

        if result.symbol in existing_symbols:
            # Update existing
            for i, m in enumerate(metadata):
                if m['symbol'] == result.symbol:
                    metadata[i] = entry
                    break
        else:
            metadata.append(entry)

    # Sort by symbol
    metadata.sort(key=lambda x: x['symbol'])

    # Save
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Updated metadata: {len(metadata)} entries")


def main():
    """Run corpus expansion."""
    base_path = Path(__file__).parent.parent
    whitepaper_dir = base_path / "data" / "whitepapers"
    metadata_path = base_path / "data" / "metadata" / "whitepaper_metadata.json"

    print("=" * 60)
    print("WHITEPAPER CORPUS EXPANSION")
    print("=" * 60)
    print(f"Target directory: {whitepaper_dir}")
    print(f"Sources to process: {len(WHITEPAPER_SOURCES)}")
    print("=" * 60)

    downloader = WhitepaperDownloader(whitepaper_dir)
    results = downloader.download_all(WHITEPAPER_SOURCES, skip_existing=True)

    # Summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(successful)}")
    print(f"Failed:     {len(failed)}")

    if failed:
        print("\nFailed downloads:")
        for r in failed:
            print(f"  - {r.symbol}: {r.error}")

    print("\nSuccessful downloads:")
    for r in successful:
        print(f"  - {r.symbol}: {r.pages} pages ({r.doc_type})")

    # Update metadata
    update_metadata(successful, metadata_path)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Run: python run_nlp.py  (extract text)")
    print("  2. Run: python run_alignment.py  (recompute alignment)")
    print("=" * 60)


if __name__ == "__main__":
    main()
