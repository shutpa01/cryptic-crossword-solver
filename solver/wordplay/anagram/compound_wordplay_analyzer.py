#!/usr/bin/env python3
"""
Compound Wordplay Analysis
Analyzes cohorts from pipeline_simulator for compound wordplay patterns.
Takes cases with partial anagram matches + remaining words and tests for additional wordplay.

Based on evidence_analysis.py pattern for debugging compound wordplay development.
"""

import sqlite3
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Set, Tuple, Optional

from solver.solver_engine.resources import connect_db, norm_letters


class CompoundWordplayAnalyzer:
    """
    Analyzes remaining words from anagram cases for compound wordplay patterns.
    Uses substitution rules and indicators to find multi-stage solutions.
    """

    def __init__(self):
        self.substitution_rules = {}
        self.wordplay_indicators = defaultdict(list)
        self.stats = {
            'total_cases': 0,
            'substitution_hits': 0,
            'multi_wordplay_hits': 0,
            'compound_solutions': 0,
            'confidence_scores': []
        }
        self._load_resources()

    def _load_resources(self):
        """Load substitution rules and wordplay indicators from database"""
        conn = connect_db()

        # Load substitution rules from wordplay table
        cursor = conn.execute("""
            SELECT indicator, substitution, category, confidence 
            FROM wordplay 
            WHERE substitution IS NOT NULL AND substitution != ''
            ORDER BY confidence DESC
        """)

        for row in cursor:
            indicator = row[0].lower().strip()
            substitution = row[1].strip()
            category = row[2] or 'unknown'
            confidence = row[3] or 'medium'

            self.substitution_rules[indicator] = {
                'substitution': substitution,
                'category': category,
                'confidence': confidence
            }

        print(f"Loaded {len(self.substitution_rules)} substitution rules")

        # Load wordplay indicators by type
        cursor = conn.execute("""
            SELECT word, wordplay_type, subtype, confidence
            FROM indicators
            WHERE wordplay_type IS NOT NULL
            ORDER BY wordplay_type, confidence DESC
        """)

        for row in cursor:
            word = row[0].lower().strip()
            wordplay_type = row[1].lower()
            subtype = row[2] or ''
            confidence = row[3] or 'medium'

            self.wordplay_indicators[wordplay_type].append({
                'word': word,
                'subtype': subtype,
                'confidence': confidence
            })

        print(f"Loaded indicators for {len(self.wordplay_indicators)} wordplay types")
        conn.close()

    def analyze_remaining_words(self, remaining_words: List[str]) -> Dict[str, Any]:
        """
        Analyze remaining words for compound wordplay patterns

        Args:
            remaining_words: List of words not accounted for by anagram analysis

        Returns:
            Dictionary with detected patterns and confidence scores
        """
        analysis = {
            'substitutions_found': [],
            'other_wordplay_found': [],
            'total_letters_from_substitutions': '',
            'confidence_score': 0.0,
            'explanation': []
        }

        total_substitution_letters = ""

        for word in remaining_words:
            word_lower = word.lower()

            # Check for substitution rules
            if word_lower in self.substitution_rules:
                rule = self.substitution_rules[word_lower]
                substitution_info = {
                    'word': word,
                    'becomes': rule['substitution'],
                    'category': rule['category'],
                    'confidence': rule['confidence']
                }
                analysis['substitutions_found'].append(substitution_info)
                total_substitution_letters += rule['substitution']
                analysis['explanation'].append(f"{word} → {rule['substitution']}")

            # Check for other wordplay indicators
            for wordplay_type, indicators in self.wordplay_indicators.items():
                if wordplay_type == 'anagram':  # Skip anagram - already processed
                    continue

                for indicator_info in indicators:
                    if indicator_info['word'] == word_lower:
                        wordplay_info = {
                            'word': word,
                            'type': wordplay_type,
                            'subtype': indicator_info['subtype'],
                            'confidence': indicator_info['confidence']
                        }
                        analysis['other_wordplay_found'].append(wordplay_info)
                        analysis['explanation'].append(
                            f"{word} ({wordplay_type} indicator)")
                        break

        analysis['total_letters_from_substitutions'] = total_substitution_letters

        # Calculate compound confidence score
        if analysis['substitutions_found'] or analysis['other_wordplay_found']:
            base_score = 0.3  # Base for finding any compound patterns
            substitution_bonus = len(analysis['substitutions_found']) * 0.2
            wordplay_bonus = len(analysis['other_wordplay_found']) * 0.1
            analysis['confidence_score'] = min(1.0,
                                               base_score + substitution_bonus + wordplay_bonus)

        return analysis

    def test_compound_construction(self, anagram_letters: str, substitution_letters: str,
                                   target_answer: str) -> Dict[str, Any]:
        """
        Test if anagram letters + substitution letters can form the target answer
        NOTE: target_answer is ONLY used for verification - never for solving
        """
        combined_letters = norm_letters(anagram_letters + substitution_letters)
        target_letters = norm_letters(target_answer)

        # Check if we have the right letters (for debugging purposes only)
        combined_sorted = ''.join(sorted(combined_letters))
        target_sorted = ''.join(sorted(target_letters))

        construction_test = {
            'anagram_letters': anagram_letters,
            'substitution_letters': substitution_letters,
            'combined_letters': combined_letters,
            'target_letters': target_letters,
            'letters_match': combined_sorted == target_sorted,
            'missing_letters': '',
            'excess_letters': ''
        }

        if not construction_test['letters_match']:
            # Calculate missing/excess for debugging
            combined_counts = Counter(combined_letters)
            target_counts = Counter(target_letters)

            missing = []
            for letter, count in target_counts.items():
                if combined_counts[letter] < count:
                    missing.extend([letter] * (count - combined_counts[letter]))

            excess = []
            for letter, count in combined_counts.items():
                if target_counts[letter] < count:
                    excess.extend([letter] * (count - target_counts[letter]))

            construction_test['missing_letters'] = ''.join(sorted(missing))
            construction_test['excess_letters'] = ''.join(sorted(excess))

        return construction_test

    def analyze_compound_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single case for compound wordplay patterns

        Args:
            case_data: Case from pipeline_simulator with anagram evidence and remaining words

        Returns:
            Comprehensive compound analysis
        """
        self.stats['total_cases'] += 1

        clue = case_data.get('clue', '')
        answer = case_data.get('answer', '')
        remaining_words = case_data.get('remaining_words', [])
        anagram_evidence = case_data.get('anagram_evidence', {})

        # Skip cases with no remaining words
        if not remaining_words:
            return None

        # Analyze remaining words for compound patterns
        compound_analysis = self.analyze_remaining_words(remaining_words)

        # If substitutions found, test complete construction
        construction_result = None
        if compound_analysis['substitutions_found']:
            anagram_letters = anagram_evidence.get('fodder_letters', '')
            substitution_letters = compound_analysis['total_letters_from_substitutions']

            construction_result = self.test_compound_construction(
                anagram_letters, substitution_letters, answer
            )

            if construction_result['letters_match']:
                self.stats['compound_solutions'] += 1

        # Update statistics
        if compound_analysis['substitutions_found']:
            self.stats['substitution_hits'] += 1
        if compound_analysis['other_wordplay_found']:
            self.stats['multi_wordplay_hits'] += 1
        if compound_analysis['confidence_score'] > 0:
            self.stats['confidence_scores'].append(compound_analysis['confidence_score'])

        result = {
            'clue': clue,
            'answer': answer,
            'remaining_words': remaining_words,
            'anagram_evidence': anagram_evidence,
            'compound_analysis': compound_analysis,
            'construction_result': construction_result,
            'overall_confidence': compound_analysis['confidence_score']
        }

        return result

    def print_analysis_results(self, results: List[Dict[str, Any]],
                               max_display: int = 20):
        """Print compound analysis results in evidence_analysis.py style"""

        print("\n" + "=" * 80)
        print("COMPOUND WORDPLAY ANALYSIS RESULTS")
        print("=" * 80)

        # Print statistics
        print(f"\nSTATISTICS:")
        print(f"  Total cases analyzed: {self.stats['total_cases']}")
        print(f"  Cases with substitutions: {self.stats['substitution_hits']}")
        print(f"  Cases with other wordplay: {self.stats['multi_wordplay_hits']}")
        print(f"  Compound solutions found: {self.stats['compound_solutions']}")
        if self.stats['confidence_scores']:
            avg_confidence = sum(self.stats['confidence_scores']) / len(
                self.stats['confidence_scores'])
            print(f"  Average confidence: {avg_confidence:.3f}")

        print(f"\nDETAILED RESULTS (showing first {max_display}):")
        print("-" * 80)

        # Sort by confidence score (highest first)
        sorted_results = sorted([r for r in results if r],
                                key=lambda x: x['overall_confidence'], reverse=True)

        for i, result in enumerate(sorted_results[:max_display], 1):
            print(f"\n[{i}] CLUE: {result['clue']}")
            print(f"    ANSWER: {result['answer']}")
            print(f"    REMAINING WORDS: {result['remaining_words']}")

            # Anagram evidence summary
            anag = result['anagram_evidence']
            print(
                f"    ANAGRAM: {' + '.join(anag.get('fodder_words', []))} → {anag.get('fodder_letters', '')}")

            # Compound analysis
            comp = result['compound_analysis']
            if comp['substitutions_found']:
                print(f"    SUBSTITUTIONS:")
                for sub in comp['substitutions_found']:
                    print(f"      {sub['word']} → {sub['becomes']} ({sub['category']})")

            if comp['other_wordplay_found']:
                print(f"    OTHER WORDPLAY:")
                for wp in comp['other_wordplay_found']:
                    print(f"      {wp['word']} ({wp['type']})")

            # Construction test
            if result['construction_result']:
                cons = result['construction_result']
                print(f"    CONSTRUCTION TEST:")
                print(
                    f"      {cons['anagram_letters']} + {cons['substitution_letters']} = {cons['combined_letters']}")
                print(f"      Target: {cons['target_letters']}")
                print(f"      Match: {'✓' if cons['letters_match'] else '✗'}")
                if not cons['letters_match']:
                    if cons['missing_letters']:
                        print(f"      Missing: {cons['missing_letters']}")
                    if cons['excess_letters']:
                        print(f"      Excess: {cons['excess_letters']}")

            print(f"    CONFIDENCE: {result['overall_confidence']:.3f}")
            print(f"    EXPLANATION: {' | '.join(comp['explanation'])}")
            print("-" * 80)


def main():
    """
    Main analysis function - processes cohort from pipeline_simulator
    """
    # For now, create some test cases to validate the compound analysis
    # In practice, this would read from pipeline_simulator output

    print("COMPOUND WORDPLAY ANALYZER")
    print("Loading resources...")

    analyzer = CompoundWordplayAnalyzer()

    # Test case based on HANDSOME example
    test_cases = [
        {
            'clue': 'Good-looking husband one\'s mad to change',
            'answer': 'HANDSOME',
            'remaining_words': ['Good-looking', 'husband', 'to', 'change'],
            'anagram_evidence': {
                'fodder_words': ['one\'s', 'mad'],
                'fodder_letters': 'ONESAMD',
                'evidence_type': 'partial',
                'confidence': 0.7
            }
        }
    ]

    print(f"\nAnalyzing {len(test_cases)} test cases...")
    results = []

    for case in test_cases:
        result = analyzer.analyze_compound_case(case)
        if result:
            results.append(result)

    analyzer.print_analysis_results(results)

    print(f"\n{'=' * 80}")
    print("COMPOUND ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()