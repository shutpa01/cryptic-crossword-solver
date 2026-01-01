#!/usr/bin/env python3
"""
Nightly pipeline:
1. Fetch new synonyms from Merriam–Webster.
2. Rebuild the bidirectional synonym_pairs table.
"""

import sys
from pathlib import Path

# Make project imports work regardless of where script is run
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.fetch_synonyms import main as fetch_main
from scripts.synonym_pairs import main as pairs_main


def run():
    print("=" * 60)
    print("NIGHTLY LEXICON PIPELINE")
    print("=" * 60)

    print("\nSTEP 1 — Fetching synonyms...")
    fetch_main()  # or whatever nightly limit you choose

    print("\nSTEP 2 — Rebuilding synonym pairs...")
    pairs_main()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run()
