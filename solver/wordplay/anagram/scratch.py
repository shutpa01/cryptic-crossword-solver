#!/usr/bin/env python3
"""
Analyze indicator frequency from actual cryptic clue corpus.

For each ANAGRAM clue:
1. Find which clue words anagram to the answer (fodder)
2. Check adjacent words for known indicators
3. Tally indicator occurrences

This gives us empirical confidence scores based on real usage.
"""

import sqlite3
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Set, Tuple, Optional


def get_letters(text: str) -> str:
    """Extract just letters, lowercase."""
    return ''.join(c.lower() for c in text if c.isalpha())


def is_anagram(word1: str, word2: str) -> bool:
    """Check if two strings are anagrams of each other."""
    return sorted(get_letters(word1)) == sorted(get_letters(word2))


def find_anagram_fodder(clue_words: List[str], answer: str) -> List[
    Tuple[List[str], int, int]]:
    """
    Find combinations of clue words that anagram to the answer.
    Returns list of (fodder_words, start_idx, end_idx) tuples.
    """
    answer_letters = get_letters(answer)
    answer_sorted = sorted(answer_letters)
    results = []

    # Try contiguous sequences of words
    for start in range(len(clue_words)):
        letters_so_far = ""
        for end in range(start, min(start + 5, len(clue_words))):  # Max 5 words
            word = clue_words[end]
            letters_so_far += get_letters(word)

            if len(letters_so_far) > len(answer_letters):
                break

            if sorted(letters_so_far) == answer_sorted:
                results.append((clue_words[start:end + 1], start, end))

    return results


def load_anagram_indicators(db_path: str) -> Set[str]:
    """Load all known anagram indicators from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT LOWER(word) FROM indicators 
        WHERE wordplay_type = 'anagram'
    """)
    indicators = {row[0] for row in cursor.fetchall()}
    conn.close()
    return indicators


def tokenize_clue(clue_text: str) -> List[str]:
    """Split clue into words, preserving order."""
    # Remove enumeration pattern at end
    clue_text = re.sub(r'\s*\(\d+(?:[,-]\d+)*\)\s*$', '', clue_text)
    # Split on whitespace and punctuation (but keep words)
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clue_text)
    return words


def analyze_corpus(db_path: str, limit: int = None, debug: bool = False) -> Counter:
    """
    Analyze anagram clues to count indicator frequency.

    Returns Counter of {indicator_word: count}
    """
    # Load known indicators
    known_indicators = load_anagram_indicators(db_path)
    print(f"Loaded {len(known_indicators)} known anagram indicators")

    # Frequency counter
    indicator_counts = Counter()

    # Track clues where we found indicators
    clues_with_indicators = 0
    clues_without_indicators = 0

    # Track potential new indicators (adjacent to fodder but not in DB)
    potential_new_indicators = Counter()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query anagram clues
    query = """
        SELECT clue, answer FROM clues 
        WHERE wordplay_type = 'anagram'
        AND answer IS NOT NULL
        AND LENGTH(answer) >= 4
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    print(f"Processing {len(rows)} anagram clues...")

    for i, (clue_text, answer) in enumerate(rows):
        if not clue_text or not answer:
            continue

        clue_words = tokenize_clue(clue_text)
        if len(clue_words) < 2:
            continue

        # Find fodder (words that anagram to answer)
        fodder_matches = find_anagram_fodder(clue_words, answer)

        if not fodder_matches:
            continue

        found_indicator = False

        for fodder_words, start_idx, end_idx in fodder_matches:
            # Check word BEFORE fodder
            if start_idx > 0:
                prev_word = clue_words[start_idx - 1].lower()
                if prev_word in known_indicators:
                    indicator_counts[prev_word] += 1
                    found_indicator = True
                    if debug and i < 20:
                        print(f"  [{prev_word}] {' '.join(fodder_words)} -> {answer}")
                else:
                    # Track as potential new indicator
                    potential_new_indicators[prev_word] += 1

            # Check word AFTER fodder
            if end_idx < len(clue_words) - 1:
                next_word = clue_words[end_idx + 1].lower()
                if next_word in known_indicators:
                    indicator_counts[next_word] += 1
                    found_indicator = True
                    if debug and i < 20:
                        print(f"  {' '.join(fodder_words)} [{next_word}] -> {answer}")
                else:
                    potential_new_indicators[next_word] += 1

            # Also check 2 words before/after for two-word indicators
            if start_idx > 1:
                two_word = f"{clue_words[start_idx - 2].lower()} {clue_words[start_idx - 1].lower()}"
                if two_word in known_indicators:
                    indicator_counts[two_word] += 1
                    found_indicator = True

            if end_idx < len(clue_words) - 2:
                two_word = f"{clue_words[end_idx + 1].lower()} {clue_words[end_idx + 2].lower()}"
                if two_word in known_indicators:
                    indicator_counts[two_word] += 1
                    found_indicator = True

        if found_indicator:
            clues_with_indicators += 1
        else:
            clues_without_indicators += 1

        # Progress
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} clues...")

    print(f"\nResults:")
    print(f"  Clues with identified indicators: {clues_with_indicators}")
    print(f"  Clues without identified indicators: {clues_without_indicators}")

    return indicator_counts, potential_new_indicators


def show_current_rankings(db_path: str, top_n: int = 50):
    """Show current indicator rankings from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if frequency column exists
    cursor.execute("PRAGMA table_info(indicators)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'frequency' not in columns:
        print("No frequency data in database yet. Run analysis first.")
        conn.close()
        return

    print(f"\n{'=' * 60}")
    print(f"TOP {top_n} ANAGRAM INDICATORS BY FREQUENCY (from DB):")
    print('=' * 60)

    cursor.execute("""
        SELECT word, frequency, confidence FROM indicators 
        WHERE wordplay_type = 'anagram' AND frequency > 0
        ORDER BY frequency DESC
        LIMIT ?
    """, (top_n,))

    for word, freq, conf in cursor.fetchall():
        print(f"  {word}: {freq} ({conf})")

    # Also show indicators with no frequency data
    cursor.execute("""
        SELECT COUNT(*) FROM indicators 
        WHERE wordplay_type = 'anagram' AND (frequency IS NULL OR frequency = 0)
    """)
    no_data = cursor.fetchone()[0]
    print(f"\nIndicators with no frequency data: {no_data}")

    conn.close()


def save_results(db_path: str, indicator_counts: Counter):
    """Save indicator frequency directly to database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add frequency column if not exists
    try:
        cursor.execute("ALTER TABLE indicators ADD COLUMN frequency INTEGER DEFAULT 0")
        print("Added 'frequency' column to indicators table")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Update frequency for each indicator
    updated = 0
    for indicator, count in indicator_counts.items():
        cursor.execute("""
            UPDATE indicators 
            SET frequency = ?
            WHERE LOWER(word) = ? AND wordplay_type = 'anagram'
        """, (count, indicator))
        updated += cursor.rowcount

    conn.commit()
    conn.close()
    print(f"Updated frequency for {updated} indicator records")


def update_indicator_confidence(db_path: str, dry_run: bool = True):
    """
    Update indicator confidence based on frequency stored in DB.

    Tiers:
    - very_high: top 10% by frequency
    - high: top 25%
    - medium: top 50%
    - low: bottom 50% or frequency = 0
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all anagram indicators with their frequency
    cursor.execute("""
        SELECT word, frequency FROM indicators 
        WHERE wordplay_type = 'anagram'
        ORDER BY frequency DESC
    """)
    rows = cursor.fetchall()

    if not rows:
        print("No anagram indicators found")
        conn.close()
        return

    # Get frequency values (excluding zeros for percentile calculation)
    frequencies = [f for _, f in rows if f and f > 0]

    if not frequencies:
        print("No frequency data found - run frequency analysis first")
        conn.close()
        return

    # Calculate percentile thresholds
    frequencies_sorted = sorted(frequencies, reverse=True)
    p90 = frequencies_sorted[int(len(frequencies_sorted) * 0.1)] if len(
        frequencies_sorted) > 10 else frequencies_sorted[0]
    p75 = frequencies_sorted[int(len(frequencies_sorted) * 0.25)] if len(
        frequencies_sorted) > 4 else frequencies_sorted[0]
    p50 = frequencies_sorted[int(len(frequencies_sorted) * 0.5)] if len(
        frequencies_sorted) > 2 else 1

    print(f"\nConfidence thresholds based on {len(frequencies)} indicators with data:")
    print(f"  very_high: >= {p90} occurrences")
    print(f"  high: >= {p75} occurrences")
    print(f"  medium: >= {p50} occurrences")
    print(f"  low: < {p50} occurrences (or no data)")

    updates = []
    for word, freq in rows:
        freq = freq or 0
        if freq >= p90:
            confidence = 'very_high'
        elif freq >= p75:
            confidence = 'high'
        elif freq >= p50:
            confidence = 'medium'
        else:
            confidence = 'low'
        updates.append((word, freq, confidence))

    if dry_run:
        print(f"\nDry run - would update {len(updates)} indicators:")
        print("\nTop 30 (very_high/high):")
        for word, freq, conf in sorted(updates, key=lambda x: -(x[1] or 0))[:30]:
            print(f"  {word}: {freq} -> {conf}")
        print("\nBottom 20 (low):")
        for word, freq, conf in [u for u in updates if u[2] == 'low'][:20]:
            print(f"  {word}: {freq} -> {conf}")
    else:
        for word, freq, confidence in updates:
            cursor.execute("""
                UPDATE indicators 
                SET confidence = ?
                WHERE word = ? AND wordplay_type = 'anagram'
            """, (confidence, word))

        conn.commit()
        print(f"\nUpdated confidence for {len(updates)} anagram indicators")

    conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze anagram indicator frequency')
    parser.add_argument('db_path', help='Path to cryptic.db')
    parser.add_argument('--limit', type=int, help='Limit number of clues to process')
    parser.add_argument('--update-confidence', action='store_true',
                        help='Update confidence levels based on frequency')
    parser.add_argument('--debug', action='store_true', help='Show debug output')
    parser.add_argument('--apply', action='store_true',
                        help='Actually apply changes to database')
    parser.add_argument('--show', action='store_true',
                        help='Just show current rankings from DB (no analysis)')
    parser.add_argument('--top', type=int, default=50,
                        help='Number of top indicators to show (default: 50)')

    args = parser.parse_args()

    # Just show current rankings if requested
    if args.show:
        show_current_rankings(args.db_path, args.top)
        return

    # Step 1: Analyze corpus and count indicator frequency
    indicator_counts, potential_new = analyze_corpus(
        args.db_path,
        limit=args.limit,
        debug=args.debug
    )

    print(f"\n{'=' * 60}")
    print("TOP 50 INDICATORS BY FREQUENCY:")
    print('=' * 60)
    for indicator, count in indicator_counts.most_common(50):
        print(f"  {indicator}: {count}")

    print(f"\n{'=' * 60}")
    print("POTENTIAL NEW INDICATORS (not in DB, but adjacent to fodder):")
    print('=' * 60)
    potential_to_add = []
    for word, count in potential_new.most_common(50):
        if count >= 5:  # Only show if appears 5+ times
            print(f"  {word}: {count}")
            if count >= 10:  # Strong candidates
                potential_to_add.append((word, count))

    if potential_to_add and args.apply:
        print(f"\n{'=' * 60}")
        print("ADDING NEW INDICATORS (frequency >= 10):")
        print('=' * 60)
        conn = sqlite3.connect(args.db_path)
        cursor = conn.cursor()
        added = 0
        for word, count in potential_to_add:
            # Check if already exists
            cursor.execute("""
                SELECT 1 FROM indicators WHERE LOWER(word) = ? AND wordplay_type = 'anagram'
            """, (word.lower(),))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO indicators (word, wordplay_type, subtype, confidence, frequency)
                    VALUES (?, 'anagram', 'general', 'medium', ?)
                """, (word, count))
                print(f"  Added: {word} (frequency: {count})")
                added += 1
        conn.commit()
        conn.close()
        print(f"Added {added} new anagram indicators")

    # Step 2: Save frequency to database
    if args.apply:
        save_results(args.db_path, indicator_counts)

        # Step 3: Optionally update confidence levels
        if args.update_confidence:
            update_indicator_confidence(args.db_path, dry_run=False)
    else:
        print(f"\n{'=' * 60}")
        print("DRY RUN - No changes made to database")
        print("Use --apply to save frequency data")
        print("Use --apply --update-confidence to also update confidence levels")
        print('=' * 60)

        if args.update_confidence:
            # Show what confidence updates would look like
            # First simulate saving frequencies
            print("\nSimulating confidence updates based on counted frequencies...")

            # Calculate thresholds from counts
            counts = sorted(indicator_counts.values(), reverse=True)
            if counts:
                p90 = counts[int(len(counts) * 0.1)] if len(counts) > 10 else counts[0]
                p75 = counts[int(len(counts) * 0.25)] if len(counts) > 4 else counts[0]
                p50 = counts[int(len(counts) * 0.5)] if len(counts) > 2 else 1

                print(f"\nConfidence thresholds:")
                print(f"  very_high: >= {p90} occurrences")
                print(f"  high: >= {p75} occurrences")
                print(f"  medium: >= {p50} occurrences")
                print(f"  low: < {p50} occurrences")


if __name__ == "__main__":
    main()