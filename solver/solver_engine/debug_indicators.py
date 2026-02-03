#!/usr/bin/env python3
"""
Test script to debug indicator detection for specific clue
Shows which candidate the evidence system actually proposes
"""

import sys
import os

# Add the project root to Python path
sys.path.append(r'C:\Users\shute\PycharmProjects\cryptic_solver')

from solver.wordplay.anagram.anagram_evidence_system_patched import ComprehensiveWordplayDetector

# Test the problematic clue
detector = ComprehensiveWordplayDetector()
clue = "One who might need a rest during the game"

print("=== INDICATOR DEBUG TEST ===")
print(f"Clue: {clue}")

# Test what indicators are detected
indicators = detector.detect_wordplay_indicators(clue)
print(f"Detected indicators: {indicators}")

# Show if any anagram indicators found
if indicators['anagram']:
    print(f"Anagram indicators found: {indicators['anagram']}")
    print("ERROR: Should NOT find indicators - this clue has no anagram indicators!")
else:
    print("NO anagram indicators found - should return empty evidence")
    print("This is CORRECT behavior")

print("\n=== CANDIDATE PROPOSAL TEST ===")
print("Testing with multiple candidates to see what gets proposed...")

# Test full evidence analysis with multiple potential candidates
candidates = ["SNOOKER PLAYER", "TENNIS PLAYER", "GOLFER", "FOOTBALLER", "NEED", "GAME", "REST", "MIGHT"]

evidence_list = detector.analyze_clue_for_anagram_evidence(
    clue_text=clue,
    candidates=["managed inside"],
    enumeration="7,6",
    debug=True
)

print(f"\nEvidence results: {len(evidence_list)} items found")
if evidence_list:
    print("Evidence found - showing what the system is actually proposing:")
    for i, evidence in enumerate(evidence_list, 1):
        print(f"\n  [{i}] PROPOSED CANDIDATE: {evidence.candidate}")
        print(f"      Fodder words: {evidence.fodder_words}")
        print(f"      Fodder letters: {evidence.fodder_letters}")
        print(f"      Evidence type: {evidence.evidence_type}")
        print(f"      Confidence: {evidence.confidence}")
        print(f"      Needed letters: {evidence.needed_letters}")
        print(f"      Excess letters: {evidence.excess_letters}")
        print(f"      ---")
        print(f"      ACTUAL CORRECT ANSWER: SNOOKER PLAYER")
        print(f"      MATCH: {'YES' if evidence.candidate.upper() == 'SNOOKER PLAYER' else 'NO - FALSE POSITIVE!'}")
else:
    print("CORRECT: No evidence found (as expected)")

print("\n=== SUMMARY ===")
if evidence_list:
    print("The evidence system IS proposing candidates:")
    for evidence in evidence_list:
        print(f"  - Proposes: {evidence.candidate} (confidence: {evidence.confidence})")
else:
    print("The evidence system correctly found no valid candidates.")