#!/usr/bin/env python3
"""
Debug script for anagram evidence testing.
Shows detailed word combination analysis for the INFRARED clue.
"""

import sys
import os
from anagram_evidence_system import ComprehensiveWordplayDetector


def debug_infrared_clue():
    """Debug the specific INFRARED clue to see word combination testing."""

    # Use the database path
    db_path = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
    detector = ComprehensiveWordplayDetector(db_path=db_path)

    clue = "Woolly ran if colour's outside the visible spectrum (8)"
    candidates = ["INFRARED",
                  "founders"]  # Test both the correct answer and the false positive

    print("=" * 80)
    print(f"🔍 DEBUG ANALYSIS: {clue}")
    print("=" * 80)

    # Test with debug enabled
    evidence_list = detector.analyze_clue_for_anagram_evidence(clue, candidates,
                                                               debug=True)

    print("\n" + "=" * 60)
    print("📊 EVIDENCE SUMMARY:")
    print("=" * 60)

    if evidence_list:
        for evidence in evidence_list:
            print(f"\n🎯 EVIDENCE FOUND for {evidence.candidate}:")
            print(f"  Type: {evidence.evidence_type}")
            print(f"  Fodder: {' + '.join(evidence.fodder_words)}")
            print(f"  Confidence: {evidence.confidence:.2f}")
            if evidence.needed_letters:
                print(f"  Needed letters: '{evidence.needed_letters}'")
            if evidence.excess_letters:
                print(f"  Excess letters: '{evidence.excess_letters}'")

            boost = detector.calculate_anagram_score_boost(evidence)
            print(f"  Score boost: +{boost:.1f}")
    else:
        print("❌ No evidence found for any candidate")


if __name__ == "__main__":
    debug_infrared_clue()