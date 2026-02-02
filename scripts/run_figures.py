#!/usr/bin/env python3
"""
Figure Generation Pipeline for TENSOR-DEFI

Generates all publication-ready figures.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization.plots import FigureGenerator


def main():
    base_path = Path(__file__).parent.parent

    print("="*60)
    print("TENSOR-DEFI: Figure Generation Pipeline")
    print("="*60)

    generator = FigureGenerator(
        output_dir=base_path / "figures",
        data_dir=base_path / "outputs"
    )

    generator.generate_all()

    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
