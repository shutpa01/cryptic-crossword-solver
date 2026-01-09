#!/usr/bin/env python3
"""
Debug progressive expansion for specific failing cases
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anagram_evidence_system_working import ComprehensiveWordplayDetector


def debug_expansion_case(clue, candidate, expected_fodder):
    """Debug a specific expansion case step by step."""
    print(f"\n{'=' * 80}")
    print(f"DEBUGGING: {clue}")
    print(f"CANDIDATE: {candidate}")
    print(f"EXPECTED FODDER: {expected_fodder}")
    print(f"{'=' * 80}")

    # Initialize detector
    detector = ComprehensiveWordplayDetector(db_path="nonexistent")

    # Add some common anagram indicators for testing
    detector.anagram_indicators = ['sadly', 'possibly', 'trained', 'confused', 'mixed',
                                   'jumbled']

    # Step 1: Test indicator detection
    indicators = detector.detect_wordplay_indicators(clue)
    print(f"1. DETECTED INDICATORS: {indicators}")

    if not indicators['anagram']:
        print("   ERROR: No anagram indicators found!")
        return

    # Step 2: Split clue into words
    clue_words = [word.strip('.,!?:;()') for word in clue.split()]
    print(f"2. CLUE WORDS: {clue_words}")

    # Step 3: Find indicator positions
    indicator_positions = []
    for i, word in enumerate(clue_words):
        if word.lower() in indicators['anagram']:
            indicator_positions.append(i)
            print(f"   Found indicator '{word}' at position {i}")

    # Step 4: Test individual word contributions
    candidate_normalized = detector.normalize_letters(candidate)
    print(f"3. CANDIDATE LETTERS: {candidate_normalized}")

    print("4. INDIVIDUAL WORD CONTRIBUTION TESTS:")
    for i, word in enumerate(clue_words):
        word_normalized = detector.normalize_letters(word)
        if word_normalized:
            can_contribute, ratio, remaining = detector.can_contribute_letters(
                candidate_normalized, word_normalized
            )
            print(
                f"   [{i}] '{word}' ({word_normalized}) → can_contribute={can_contribute}, remaining='{remaining}'")

    # Step 5: Trace progressive expansion manually
    print("5. PROGRESSIVE EXPANSION TRACE:")

    candidates_list = [candidate]
    link_words = {'in', 'with', 'of', 'to', 'for', 'by', 'from', 'about', 'on', 'after',
                  'and'}

    all_fodder_positions = set()

    for ind_pos in indicator_positions:
        print(
            f"\n   Starting from indicator at position {ind_pos}: '{clue_words[ind_pos]}'")

        # Left expansion
        print("   LEFT EXPANSION:")
        pos = ind_pos - 1
        while pos >= 0:
            word = clue_words[pos]
            word_lower = word.lower()

            print(
                f"     Position {pos}: '{word}' (normalized: '{detector.normalize_letters(word)}')")

            # Skip link words
            if word_lower in link_words:
                print(f"       → LINK WORD - skipping but continuing")
                pos -= 1
                continue

            # Test contribution
            can_contribute = detector._can_word_contribute_to_candidates(word,
                                                                         candidates_list)
            print(f"       → can_contribute_to_candidates: {can_contribute}")

            if can_contribute:
                all_fodder_positions.add(pos)
                print(f"       → ADDED to fodder positions")
                pos -= 1
            else:
                print(f"       → STOPPING expansion (word doesn't contribute)")
                break

        # Right expansion
        print("   RIGHT EXPANSION:")
        pos = ind_pos + 1
        while pos < len(clue_words):
            word = clue_words[pos]
            word_lower = word.lower()

            print(
                f"     Position {pos}: '{word}' (normalized: '{detector.normalize_letters(word)}')")

            # Skip link words
            if word_lower in link_words:
                print(f"       → LINK WORD - skipping but continuing")
                pos += 1
                continue

            # Test contribution
            can_contribute = detector._can_word_contribute_to_candidates(word,
                                                                         candidates_list)
            print(f"       → can_contribute_to_candidates: {can_contribute}")

            if can_contribute:
                all_fodder_positions.add(pos)
                print(f"       → ADDED to fodder positions")
                pos += 1
            else:
                print(f"       → STOPPING expansion (word doesn't contribute)")
                break

    # Step 6: Show final fodder words
    print(f"\n6. FINAL FODDER POSITIONS: {sorted(all_fodder_positions)}")
    fodder_words = []
    for pos in sorted(all_fodder_positions):
        word = clue_words[pos]
        if word.lower() not in indicators['anagram'] and len(word) > 1:
            fodder_words.append(word)

    print(f"7. FINAL FODDER WORDS: {fodder_words}")
    fodder_letters = ''.join(detector.normalize_letters(word) for word in fodder_words)
    print(f"8. TOTAL FODDER LETTERS: '{fodder_letters}' ({len(fodder_letters)} letters)")

    # Compare with expected
    expected_normalized = detector.normalize_letters(expected_fodder)
    print(
        f"9. EXPECTED LETTERS: '{expected_normalized}' ({len(expected_normalized)} letters)")

    if fodder_letters == expected_normalized:
        print("✅ SUCCESS: Found expected fodder!")
    else:
        print("❌ MISMATCH: Found different fodder than expected")


if __name__ == "__main__":
    # Debug the three failing cases
    test_cases = [
        ("No cure, sadly — time for support (9)", "ENCOURAGE", "No cure"),
        ("One travelling from Earth to Saturn, possibly around end of era (9)",
         "ASTRONAUT", "to Saturn"),
        ("Work a lot in being trained? It's not mandatory (8)", "OPTIONAL", "a lot in")
    ]

    for clue, candidate, expected in test_cases:
        debug_expansion_case(clue, candidate, expected)
        input("\nPress Enter to continue to next test case...")