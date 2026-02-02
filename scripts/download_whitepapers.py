#!/usr/bin/env python3
"""
Quick whitepaper downloader for expansion corpus.

Downloads all whitepapers from URLs in metadata, handles PDF and HTML fallbacks.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

import aiohttp
from tqdm.asyncio import tqdm


async def download_file(session: aiohttp.ClientSession, url: str, output_path: Path, symbol: str) -> bool:
    """Download a single file (binary mode for PDFs)."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=120), headers=headers, allow_redirects=True) as resp:
            if resp.status == 200:
                content = await resp.read()
                output_path.write_bytes(content)
                size_kb = len(content) / 1024
                print(f"  {symbol}: OK ({size_kb:.1f} KB)")
                return True
            else:
                print(f"  {symbol}: HTTP {resp.status}")
                return False
    except asyncio.TimeoutError:
        print(f"  {symbol}: Timeout")
        return False
    except Exception as e:
        print(f"  {symbol}: Failed - {type(e).__name__}: {e}")
        return False


async def download_html_content(session: aiohttp.ClientSession, url: str, output_path: Path, symbol: str) -> bool:
    """Download HTML docs page content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60), headers=headers) as resp:
            if resp.status == 200:
                content = await resp.text()
                output_path.write_text(content)
                print(f"  {symbol}: OK (HTML)")
                return True
            else:
                print(f"  {symbol}: HTTP {resp.status}")
                return False
    except Exception as e:
        print(f"  {symbol}: Failed - {e}")
        return False


async def download_single(session: aiohttp.ClientSession, item: dict, output_dir: Path) -> tuple[str, bool]:
    """Download a single whitepaper."""
    symbol = item['symbol']
    url = item['url']
    fallback = item.get('fallback_type')

    # Determine if it's HTML or binary
    is_html = fallback == 'html' or (
        not url.endswith('.pdf') and
        ('docs.' in url or url.endswith('/'))
    )

    if is_html:
        output_path = output_dir / f"{symbol.lower()}_whitepaper.html"
        success = await download_html_content(session, url, output_path, symbol)
    else:
        output_path = output_dir / f"{symbol.lower()}_whitepaper.pdf"
        success = await download_file(session, url, output_path, symbol)

    return symbol, success


async def main():
    base_path = Path(__file__).parent.parent
    metadata_path = base_path / "data" / "metadata" / "whitepaper_metadata.json"
    output_dir = base_path / "data" / "whitepapers"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"{'='*60}")
    print("WHITEPAPER DOWNLOADER")
    print(f"{'='*60}")
    print(f"Total assets: {len(metadata)}")

    # Check what we already have (and is non-empty)
    existing = set()
    for f in output_dir.iterdir():
        if f.suffix in ['.pdf', '.html', '.md'] and f.stat().st_size > 1000:
            existing.add(f.stem.split('_')[0].upper())

    print(f"Already downloaded: {len(existing)}")

    # Filter to what we need (failed or missing)
    to_download = [m for m in metadata if m['symbol'] not in existing]
    print(f"Need to download: {len(to_download)}")

    if not to_download:
        print("\nAll whitepapers already downloaded!")
        return

    print(f"\nDownloading...\n")

    connector = aiohttp.TCPConnector(limit=3)
    async with aiohttp.ClientSession(connector=connector) as session:

        results = {'success': [], 'failed': []}

        for item in to_download:
            symbol, success = await download_single(session, item, output_dir)

            if success:
                results['success'].append(symbol)
            else:
                results['failed'].append(symbol)

            # Small delay to be nice
            await asyncio.sleep(1)

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Success: {len(results['success'])}")
    print(f"Failed:  {len(results['failed'])}")

    if results['failed']:
        print(f"\nFailed downloads:")
        for symbol in results['failed']:
            item = next(m for m in metadata if m['symbol'] == symbol)
            print(f"  {symbol}: {item['url']}")

    # Update metadata with download status
    updated_metadata = []
    for item in metadata:
        item = item.copy()
        if item['symbol'] in results['success']:
            item['download_date'] = datetime.now().strftime('%Y-%m-%d')
            item['download_status'] = 'success'
        elif item['symbol'] in results['failed']:
            item['download_status'] = 'failed'
        updated_metadata.append(item)

    with open(metadata_path, 'w') as f:
        json.dump(updated_metadata, f, indent=2)

    print(f"\nUpdated metadata")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
