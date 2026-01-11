#!/usr/bin/env python3
"""
Anagram Cohort Structure Inspector
Analyzes the current anagram detection system to understand:
- How evidence is structured and scored
- Candidate organization patterns
- Common wordplay patterns in successful detections
- Areas for compound wordplay extension
"""

import sqlite3
from collections import defaultdict, Counter
import json


def inspect_anagram_cohorts(db_path):
    """Inspect the current anagram cohort structure and evidence system"""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name

    print("=== ANAGRAM COHORT STRUCTURE ANALYSIS ===\n")

    # 1. Check the evidence table structure
    print("1. EVIDENCE TABLE STRUCTURE:")
    cursor = conn.execute("PRAGMA table_info(evidence)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"   {col['name']}: {col['type']}")
    print()

    # 2. Sample evidence records to understand format
    print("2. SAMPLE EVIDENCE RECORDS:")
    cursor = conn.execute("""
        SELECT clue_id, evidence_type, evidence_data, confidence_score
        FROM evidence 
        WHERE evidence_type = 'anagram'
        ORDER BY confidence_score DESC
        LIMIT 10
    """)

    for row in cursor:
        print(f"   Clue ID: {row['clue_id']}")
        print(f"   Type: {row['evidence_type']}")
        print(f"   Confidence: {row['confidence_score']}")
        print(f"   Data: {row['evidence_data'][:100]}...")  # First 100 chars
        print("   ---")
    print()

    # 3. Evidence type distribution
    print("3. EVIDENCE TYPE DISTRIBUTION:")
    cursor = conn.execute("""
        SELECT evidence_type, COUNT(*) as count, 
               AVG(confidence_score) as avg_confidence,
               MAX(confidence_score) as max_confidence
        FROM evidence 
        GROUP BY evidence_type 
        ORDER BY count DESC
    """)

    for row in cursor:
        print(f"   {row['evidence_type']}: {row['count']} records, "
              f"avg confidence: {row['avg_confidence']:.3f}, "
              f"max: {row['max_confidence']:.3f}")
    print()

    # 4. Confidence score distribution for anagrams
    print("4. ANAGRAM CONFIDENCE SCORE DISTRIBUTION:")
    cursor = conn.execute("""
        SELECT 
            CASE 
                WHEN confidence_score >= 0.9 THEN '0.9-1.0'
                WHEN confidence_score >= 0.8 THEN '0.8-0.9'
                WHEN confidence_score >= 0.7 THEN '0.7-0.8'
                WHEN confidence_score >= 0.6 THEN '0.6-0.7'
                WHEN confidence_score >= 0.5 THEN '0.5-0.6'
                ELSE '<0.5'
            END as score_range,
            COUNT(*) as count
        FROM evidence 
        WHERE evidence_type = 'anagram'
        GROUP BY score_range
        ORDER BY MIN(confidence_score) DESC
    """)

    for row in cursor:
        print(f"   {row['score_range']}: {row['count']} anagrams")
    print()

    # 5. Analyze anagram evidence data structure
    print("5. ANAGRAM EVIDENCE DATA ANALYSIS:")
    cursor = conn.execute("""
        SELECT evidence_data, confidence_score
        FROM evidence 
        WHERE evidence_type = 'anagram'
        AND confidence_score > 0.7
        LIMIT 5
    """)

    print("   Sample high-confidence anagram evidence structures:")
    for i, row in enumerate(cursor, 1):
        print(f"   Example {i} (confidence: {row['confidence_score']:.3f}):")
        try:
            data = json.loads(row['evidence_data'])
            print(f"      Keys: {list(data.keys())}")
            if 'indicator_words' in data:
                print(f"      Indicators: {data['indicator_words']}")
            if 'source_letters' in data:
                print(f"      Source letters: {data['source_letters']}")
            if 'remaining_letters' in data:
                print(f"      Remaining: {data['remaining_letters']}")
        except json.JSONDecodeError:
            print(f"      Raw data: {row['evidence_data'][:50]}...")
        print()

    # 6. Check for multi-evidence clues (potential compound wordplay)
    print("6. MULTI-EVIDENCE CLUES (Potential Compound Wordplay):")
    cursor = conn.execute("""
        SELECT clue_id, COUNT(*) as evidence_count,
               GROUP_CONCAT(evidence_type) as types,
               MAX(confidence_score) as max_confidence
        FROM evidence 
        GROUP BY clue_id 
        HAVING COUNT(*) > 1
        ORDER BY evidence_count DESC, max_confidence DESC
        LIMIT 10
    """)

    print("   Clues with multiple evidence types:")
    for row in cursor:
        print(f"   Clue {row['clue_id']}: {row['evidence_count']} types "
              f"({row['types']}) max confidence: {row['max_confidence']:.3f}")
    print()

    # 7. Sample actual clues with high-confidence anagram evidence
    print("7. SAMPLE HIGH-CONFIDENCE ANAGRAM CLUES:")
    cursor = conn.execute("""
        SELECT c.clue, c.answer, e.confidence_score, c.length
        FROM clues c
        JOIN evidence e ON c.id = e.clue_id
        WHERE e.evidence_type = 'anagram'
        AND e.confidence_score > 0.8
        ORDER BY e.confidence_score DESC
        LIMIT 10
    """)

    for row in cursor:
        print(f"   \"{row['clue']}\" = {row['answer']} ({row['length']})")
        print(f"   Confidence: {row['confidence_score']:.3f}")
        print("   ---")

    conn.close()


def analyze_compound_candidates(db_path):
    """Look for clues that might benefit from compound wordplay analysis"""

    print("\n=== COMPOUND WORDPLAY CANDIDATES ===\n")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Look for clues with partial anagram evidence but no complete solution
    print("1. PARTIAL ANAGRAM EVIDENCE (Potential Compound Cases):")
    cursor = conn.execute("""
        SELECT c.clue, c.answer, c.length, e.confidence_score, e.evidence_data
        FROM clues c
        JOIN evidence e ON c.id = e.clue_id
        WHERE e.evidence_type = 'anagram'
        AND e.confidence_score BETWEEN 0.3 AND 0.7
        ORDER BY e.confidence_score DESC
        LIMIT 15
    """)

    for row in cursor:
        print(f"   \"{row['clue']}\" = {row['answer']} ({row['length']})")
        print(f"   Confidence: {row['confidence_score']:.3f}")

        try:
            data = json.loads(row['evidence_data'])
            if 'remaining_letters' in data and data['remaining_letters']:
                print(f"   Remaining letters: {data['remaining_letters']}")
        except:
            pass
        print("   ---")

    conn.close()


if __name__ == "__main__":
    # Database path relative from solver_engine folder
    DB_PATH = r"../data/cryptic_new.db"

    try:
        inspect_anagram_cohorts(DB_PATH)
        analyze_compound_candidates(DB_PATH)

        print("\n=== ANALYSIS COMPLETE ===")
        print("This structure analysis will help inform compound wordplay development:")
        print("1. Understanding current evidence data format")
        print("2. Identifying confidence scoring patterns")
        print("3. Finding clues that need additional wordplay analysis")
        print("4. Planning integration points for compound detection")

    except Exception as e:
        print(f"Error: {e}")
        print("Please check the database path and ensure the evidence table exists.")