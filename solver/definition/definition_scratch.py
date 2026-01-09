#!/usr/bin/env python3
"""
Debug script for the UNICORNS clue evidence issue.

Analyzes why "cousins" isn't getting strong evidence while "at + sea" is.
"""

from anagram_evidence_system import ComprehensiveWordplayDetector


def debug_unicorns_clue():
    """Debug the UNICORNS clue evidence detection."""

    # Use the database path
    db_path = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
    detector = ComprehensiveWordplayDetector(db_path=db_path)

    clue = "Creatures shown by navy netted by cousin at sea (8)"

    # Test specific candidates
    candidates = ["UNICORNS", "lambaste", "manatees"]

    print("=" * 80)
    print(f"🔍 DEBUG ANALYSIS: {clue}")
    print("=" * 80)

    # Show what indicators are detected
    indicators = detector.detect_wordplay_indicators(clue)
    print(f"DETECTED INDICATORS: {indicators}")

    # Test each candidate with full debug output
    for candidate in candidates:
        print(f"\n{'=' * 60}")
        print(f"🎯 TESTING CANDIDATE: {candidate}")
        print(f"{'=' * 60}")

        evidence = detector.test_anagram_evidence(candidate, clue, indicators, debug=True)

        if evidence:
            boost = detector.calculate_anagram_score_boost(evidence)
            print(f"\n🎯 EVIDENCE SUMMARY for {candidate}:")
            print(f"  Type: {evidence.evidence_type}")
            print(f"  Fodder: {' + '.join(evidence.fodder_words)}")
            print(f"  Confidence: {evidence.confidence:.2f}")
            print(f"  Score boost: +{boost:.1f}")
            if evidence.needed_letters:
                print(f"  Needed letters: '{evidence.needed_letters}'")
        else:
            print(f"\n❌ NO EVIDENCE FOUND for {candidate}")


def test_direct_combinations():
    """Test specific word combinations directly."""

    db_path = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
    detector = ComprehensiveWordplayDetector(db_path=db_path)

    print(f"\n{'=' * 80}")
    print(f"🧪 DIRECT COMBINATION TESTING")
    print(f"{'=' * 80}")

    # Test key combinations directly
    test_cases = [
        ("UNICORNS", ["cousin"], "cousins vs UNICORNS"),
        ("UNICORNS", ["cousins"], "cousins vs UNICORNS (alternative)"),
        ("lambaste", ["at", "sea"], "at + sea vs lambaste"),
        ("manatees", ["at", "sea"], "at + sea vs manatees"),
    ]

    for target, fodder_words, description in test_cases:
        print(f"\n🔬 Testing: {description}")
        print(f"   Target: {target} ({len(target)} letters)")
        print(
            f"   Fodder: {fodder_words} → {''.join(fodder_words)} ({len(''.join(fodder_words))} letters)")

        # Test contribution
        target_letters = detector.normalize_letters(target)
        fodder_letters = detector.normalize_letters(' '.join(fodder_words))

        can_contribute, contribution_ratio, remaining = detector.can_contribute_letters(
            target_letters, fodder_letters)

        print(f"   Can contribute: {can_contribute}")
        if can_contribute:
            explained_ratio = (len(target_letters) - len(remaining)) / len(target_letters)
            print(
                f"   Explained ratio: {explained_ratio:.2f} ({len(target_letters) - len(remaining)}/{len(target_letters)} letters)")
            print(f"   Remaining: '{remaining}'")

            # Calculate confidence like the real system
            confidence = 0.4 + (explained_ratio * 0.4)
            evidence_score = confidence * len(fodder_words)
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Evidence score: {evidence_score:.2f}")
        else:
            print(f"   No contribution possible")


def check_word_filtering():
    """Check if 'cousins' is being filtered out somewhere."""

    clue = "Creatures shown by navy netted by cousin at sea (8)"

    print(f"\n{'=' * 80}")
    print(f"🔍 WORD FILTERING ANALYSIS")
    print(f"{'=' * 80}")

    # Check raw words
    words = clue.split()
    print(f"Raw words: {words}")

    # Check after stop word filtering (like in the real system)
    stop_words = {'the', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'by', 'from'}
    clue_words = [word.strip('.,!?:;()') for word in clue.split()]
    content_words = [w for w in clue_words if w.lower() not in stop_words and len(w) > 1]

    print(f"After stop word filter: {content_words}")
    print(f"'cousin' present: {'cousin' in content_words}")
    print(f"'cousins' present: {'cousins' in content_words}")

    # Check word combinations that would include cousin
    from itertools import combinations

    print(f"\nSingle word combinations containing 'cousin':")
    for word in content_words:
        if 'cousin' in word.lower():
            print(f"  - {word}")


def main():
    """Run all debugging tests."""
    print("🐛 DEBUGGING UNICORNS EVIDENCE ISSUE")
    print("=" * 80)

    # Test 1: Full evidence detection
    debug_unicorns_clue()

    # Test 2: Direct combination testing
    test_direct_combinations()

    # Test 3: Check word filtering
    check_word_filtering()

    print(f"\n✅ Debugging complete!")


if __name__ == "__main__":
    main()