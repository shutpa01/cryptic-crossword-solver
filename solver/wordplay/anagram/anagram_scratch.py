#!/usr/bin/env python3
"""
Pipeline Data Inspector
Examines the actual data structure returned by pipeline_simulator
to understand what information is available for compound analysis.
"""

import sys
import os
import json
from pprint import pprint

# Import the pipeline simulator
from pipeline_simulator import run_pipeline_probe


def inspect_pipeline_data():
    """Inspect the actual pipeline simulator output structure"""

    print("PIPELINE DATA STRUCTURE INSPECTOR")
    print("=" * 50)

    # Get pipeline results
    print("Running pipeline simulator to get actual data...")
    results, overall = run_pipeline_probe()

    print(f"\nOverall stats:")
    print(f"  Total results: {len(results)}")
    for key, value in overall.items():
        print(f"  {key}: {value}")

    # Find records with anagram hits
    anagram_records = [r for r in results if
                       r.get('summary', {}).get('anagram_hits', 0) > 0]
    print(f"\nRecords with anagram hits: {len(anagram_records)}")

    if anagram_records:
        print(f"\nINSPECTING FIRST 3 ANAGRAM HIT RECORDS:")
        print("=" * 50)

        for i, record in enumerate(anagram_records[:3], 1):
            print(f"\n[{i}] RECORD STRUCTURE:")
            print(f"Clue: {record.get('clue', 'N/A')}")
            print(f"Answer: {record.get('answer', 'N/A')}")

            print(f"\nTop-level keys in record:")
            for key in sorted(record.keys()):
                value = record[key]
                if isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")

            # Look at summary in detail
            if 'summary' in record:
                print(f"\nSUMMARY structure:")
                summary = record['summary']
                for key, value in summary.items():
                    print(f"  {key}: {value}")

            # Look at anagram data in detail
            if 'anagrams' in record:
                print(f"\nANAGRAMS structure:")
                anagrams = record['anagrams']
                print(f"  Type: {type(anagrams)}")
                print(
                    f"  Length: {len(anagrams) if hasattr(anagrams, '__len__') else 'N/A'}")

                if anagrams and hasattr(anagrams, '__iter__'):
                    print(f"  First anagram hit structure:")
                    first_anagram = anagrams[0]
                    if isinstance(first_anagram, dict):
                        for key, value in first_anagram.items():
                            if isinstance(value, str) and len(value) > 50:
                                value = value[:50] + "..."
                            print(f"    {key}: {value}")
                    else:
                        print(f"    {first_anagram}")

            print("-" * 30)

    # Also look at a record without anagram hits for comparison
    non_anagram_records = [r for r in results if
                           r.get('summary', {}).get('anagram_hits', 0) == 0]
    if non_anagram_records:
        print(f"\n\nCOMPARISON - RECORD WITHOUT ANAGRAM HITS:")
        print("=" * 50)
        sample = non_anagram_records[0]
        print(f"Clue: {sample.get('clue', 'N/A')}")
        print(f"Top-level keys: {sorted(sample.keys())}")
        if 'summary' in sample:
            print(f"Summary: {sample['summary']}")

    print(f"\n\nDATA INSPECTION COMPLETE")
    print("=" * 50)
    print("Now I can see the actual data structure and write proper compound analysis!")


if __name__ == "__main__":
    inspect_pipeline_data()