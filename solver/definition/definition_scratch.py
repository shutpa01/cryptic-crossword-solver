#!/usr/bin/env python3
"""
Analyze the answer_word_frequency.csv file without loading it entirely.
Shows samples, statistics, and top entries to validate the data.
"""

import csv
import os
from collections import defaultdict, Counter
from pathlib import Path


def analyze_csv_file(csv_path):
    """Analyze the CSV file with sampling to avoid memory issues."""

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return

    file_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB")
    print("=" * 50)

    # Counters for analysis
    total_rows = 0
    answer_counts = Counter()
    word_counts = Counter()
    high_frequency_words = []
    sample_data = []

    print("Scanning file (sampling every 1000th row)...")

    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            total_rows += 1

            answer = row['answer']
            word = row['word']
            count = int(row['count'])

            # Full counting for answers and words
            answer_counts[answer] += count
            word_counts[word] += count

            # Sample every 1000th row for detailed viewing
            if i % 1000 == 0:
                sample_data.append((answer, word, count))

            # Track high-frequency word instances
            if count >= 10:
                high_frequency_words.append((answer, word, count))

            # Progress update
            if total_rows % 100000 == 0:
                print(f"  Processed {total_rows:,} rows...")

    print(f"\nFile Analysis Complete!")
    print(f"Total rows: {total_rows:,}")
    print(f"Unique answers: {len(answer_counts):,}")
    print(f"Unique words: {len(word_counts):,}")

    # Top answers by total word frequency
    print(f"\nTop 15 answers by total word occurrences:")
    for i, (answer, total_count) in enumerate(answer_counts.most_common(15)):
        print(f"  {i + 1:2d}. {answer:12s} - {total_count:,} total occurrences")

    # Most common words across all clues
    print(f"\nTop 20 most common words across all clues:")
    for i, (word, total_count) in enumerate(word_counts.most_common(20)):
        print(f"  {i + 1:2d}. '{word:12s}' - {total_count:,} occurrences")

    # High-frequency word-answer pairs
    high_frequency_words.sort(key=lambda x: x[2], reverse=True)
    print(f"\nTop 20 highest frequency word-answer pairs:")
    for i, (answer, word, count) in enumerate(high_frequency_words[:20]):
        print(f"  {i + 1:2d}. {answer:10s} + '{word:12s}' = {count:3d} times")

    # Sample data for validation
    print(f"\nSample data (every 1000th row):")
    for i, (answer, word, count) in enumerate(sample_data[:15]):
        print(f"  {answer:10s}, {word:15s}, {count:3d}")

    return {
        'total_rows': total_rows,
        'unique_answers': len(answer_counts),
        'unique_words': len(word_counts),
        'top_answers': answer_counts.most_common(10),
        'top_words': word_counts.most_common(10)
    }


def lookup_answer_words(csv_path, target_answer, top_n=20):
    """Look up all words for a specific answer."""

    print(f"\nLooking up words for answer: '{target_answer}'")
    print("-" * 40)

    answer_words = []

    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row['answer'].upper() == target_answer.upper():
                word = row['word']
                count = int(row['count'])
                answer_words.append((word, count))

    # Sort by frequency
    answer_words.sort(key=lambda x: x[1], reverse=True)

    if answer_words:
        print(f"Found {len(answer_words)} unique words for '{target_answer}':")
        for i, (word, count) in enumerate(answer_words[:top_n]):
            print(f"  {i + 1:2d}. '{word:15s}' - {count:3d} times")
    else:
        print(f"No data found for answer '{target_answer}'")

    return answer_words


def main():
    # File path
    data_dir = Path(r"C:\Users\shute\PycharmProjects\cryptic_solver\data")
    csv_path = data_dir / "answer_word_frequency.csv"

    print("Analyzing answer_word_frequency.csv")
    print("=" * 50)

    # Analyze the full file
    stats = analyze_csv_file(csv_path)

    if stats:
        # Look up specific examples
        test_answers = ['AREA', 'SPEED', 'TIME', 'ANTE', 'ERA']
        for answer in test_answers:
            lookup_answer_words(csv_path, answer, top_n=15)


if __name__ == "__main__":
    main()