#!/usr/bin/env python3
"""
Whitepaper Collector for TENSOR-DEFI

Downloads and organizes cryptocurrency whitepapers from official sources.
Falls back to GitHub READMEs when whitepapers unavailable.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Known whitepaper URLs (manually curated for reliability)
WHITEPAPER_URLS = {
    "BTC": "https://bitcoin.org/bitcoin.pdf",
    "ETH": "https://ethereum.org/content/whitepaper/whitepaper-pdf/Ethereum_Whitepaper_-_Buterin_2014.pdf",
    "SOL": "https://solana.com/solana-whitepaper.pdf",
    "ADA": "https://docs.cardano.org/cardano-whitepaper.pdf",
    "AVAX": "https://assets.website-files.com/5d80307810123f5ffbb34d6e/6008d7bbf8b10d1eb01e7e16_Avalanche%20Platform%20Whitepaper.pdf",
    "DOT": "https://polkadot.network/Polkadot-lightpaper.pdf",
    "NEAR": "https://near.org/papers/the-official-near-white-paper/",
    "ATOM": "https://v1.cosmos.network/resources/whitepaper",
    "FIL": "https://filecoin.io/filecoin.pdf",
    "LINK": "https://chain.link/whitepaper",
    "ALGO": "https://www.algorand.com/resources/white-papers",
    "UNI": "https://uniswap.org/whitepaper-v3.pdf",
    "AAVE": "https://github.com/aave/aave-protocol/blob/master/docs/Aave_Protocol_Whitepaper_v1_0.pdf",
    "MKR": "https://makerdao.com/en/whitepaper/",
    "XMR": "https://www.getmonero.org/resources/research-lab/",
    "ZEC": "https://z.cash/technology/",
    "GRT": "https://thegraph.com/docs/about/introduction",
    "AR": "https://www.arweave.org/whitepaper.pdf",
    "ICP": "https://internetcomputer.org/whitepaper.pdf",
    "APT": "https://aptos.dev/aptos-white-paper/",
    "SUI": "https://docs.sui.io/concepts/sui-white-paper",
    # More to be populated via scraping
}


@dataclass
class WhitepaperMeta:
    """Metadata for a collected whitepaper."""
    symbol: str
    name: str
    url: str
    local_path: str
    pages: Optional[int]
    download_date: str
    fallback_type: Optional[str]  # None, 'github', 'docs'


class WhitepaperCollector:
    """Collects and organizes cryptocurrency whitepapers."""

    def __init__(self, output_dir: Path, metadata_path: Path):
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load entity list
        self.entities = pd.read_csv(metadata_path / "target_entities.csv")
        self.metadata: list[WhitepaperMeta] = []

    async def download_pdf(
        self,
        session: aiohttp.ClientSession,
        url: str,
        symbol: str
    ) -> Optional[Path]:
        """Download a PDF from URL."""
        try:
            async with session.get(url, timeout=30) as resp:
                if resp.status == 200:
                    content_type = resp.headers.get('content-type', '')
                    if 'pdf' in content_type.lower() or url.endswith('.pdf'):
                        content = await resp.read()
                        output_path = self.output_dir / f"{symbol.lower()}_whitepaper.pdf"
                        output_path.write_bytes(content)
                        logger.info(f"Downloaded: {symbol} whitepaper")
                        return output_path
                    else:
                        logger.warning(f"{symbol}: URL not a PDF ({content_type})")
                else:
                    logger.warning(f"{symbol}: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"{symbol}: Download failed - {e}")
        return None

    async def collect_single(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        name: str
    ) -> Optional[WhitepaperMeta]:
        """Collect whitepaper for a single entity."""
        # Try known URL first
        url = WHITEPAPER_URLS.get(symbol)
        fallback_type = None

        if url and url.endswith('.pdf'):
            path = await self.download_pdf(session, url, symbol)
            if path:
                return WhitepaperMeta(
                    symbol=symbol,
                    name=name,
                    url=url,
                    local_path=str(path),
                    pages=None,  # Will be populated by PDF extractor
                    download_date=datetime.now().isoformat(),
                    fallback_type=None
                )

        # Fallback: Try GitHub search for whitepaper
        github_url = f"https://api.github.com/search/repositories?q={name}+whitepaper"
        try:
            async with session.get(github_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('items'):
                        repo = data['items'][0]
                        readme_url = f"https://raw.githubusercontent.com/{repo['full_name']}/main/README.md"
                        # Store as text fallback
                        async with session.get(readme_url) as readme_resp:
                            if readme_resp.status == 200:
                                content = await readme_resp.text()
                                output_path = self.output_dir / f"{symbol.lower()}_readme.md"
                                output_path.write_text(content)
                                logger.info(f"Fallback README: {symbol}")
                                return WhitepaperMeta(
                                    symbol=symbol,
                                    name=name,
                                    url=readme_url,
                                    local_path=str(output_path),
                                    pages=None,
                                    download_date=datetime.now().isoformat(),
                                    fallback_type='github'
                                )
        except Exception as e:
            logger.warning(f"{symbol}: GitHub fallback failed - {e}")

        logger.error(f"{symbol}: No whitepaper found")
        return None

    async def collect_all(self, max_concurrent: int = 5):
        """Collect all whitepapers with rate limiting."""
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.collect_single(session, row['symbol'], row['name'])
                for _, row in self.entities.iterrows()
            ]

            results = await tqdm.gather(*tasks, desc="Collecting whitepapers")
            self.metadata = [r for r in results if r is not None]

        # Save metadata
        self._save_metadata()
        return self.metadata

    def _save_metadata(self):
        """Save collection metadata to JSON."""
        output_file = self.output_dir.parent / "metadata" / "whitepaper_metadata.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "symbol": m.symbol,
                "name": m.name,
                "url": m.url,
                "local_path": m.local_path,
                "pages": m.pages,
                "download_date": m.download_date,
                "fallback_type": m.fallback_type
            }
            for m in self.metadata
        ]

        output_file.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved metadata: {output_file}")

        # Summary
        total = len(self.entities)
        collected = len(self.metadata)
        pdfs = sum(1 for m in self.metadata if m.fallback_type is None)
        fallbacks = collected - pdfs

        print(f"\n{'='*50}")
        print(f"WHITEPAPER COLLECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total entities:     {total}")
        print(f"Successfully collected: {collected} ({100*collected/total:.1f}%)")
        print(f"  - PDFs:           {pdfs}")
        print(f"  - Fallbacks:      {fallbacks}")
        print(f"  - Missing:        {total - collected}")
        print(f"{'='*50}")


async def main():
    """Run whitepaper collection."""
    base_path = Path(__file__).parent.parent.parent
    collector = WhitepaperCollector(
        output_dir=base_path / "data" / "whitepapers",
        metadata_path=base_path / "data" / "metadata"
    )
    await collector.collect_all()


if __name__ == "__main__":
    asyncio.run(main())
