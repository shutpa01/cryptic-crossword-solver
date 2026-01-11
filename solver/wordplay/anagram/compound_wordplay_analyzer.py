#!/usr/bin/env python3
"""
Explanation System Builder
Builds systematic explanations for compound cryptic constructions using enhanced pipeline data.
"""

import sys
import os
import json
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(r'C:\Users\shute\PycharmProjects\cryptic_solver')

from solver.wordplay.anagram.anagram_evidence_system import ComprehensiveWordplayDetector
from solver.wordplay.anagram.anagram_stage import generate_anagram_hypotheses


class ExplanationSystemBuilder:
    """Builds systematic explanations for cryptic constructions."""

    def __init__(self):
        """Initialize with evidence system detector for read-only operations."""
        self.detector = ComprehensiveWordplayDetector()

    def extract_definition_window(self, case: Dict[str, Any]) -> Optional[str]:
        """Extract definition window from pipeline case data."""
        answer = case.get('answer', '').upper()
        window_support = case.get('window_support', {})

        # Find window that contains the correct answer
        for window, candidates in window_support.items():
            normalized_candidates = [c.upper().replace(' ', '') for c in candidates]
            normalized_answer = answer.replace(' ', '')

            if normalized_answer in normalized_candidates:
                return window

        return None

    def detect_indicators_readonly(self, clue_text: str) -> Dict[str, List[str]]:
        """Read-only call to evidence system for indicator detection."""
        try:
            return self.detector.detect_wordplay_indicators(clue_text)
        except Exception as e:
            print(f"Warning: Could not detect indicators for '{clue_text}': {e}")
            return {'anagram': []}

    def extract_anagram_data(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Extract anagram evidence data using the same working approach as evidence_analysis.py."""
        clue_text = case.get('clue', '')
        answer = case.get('answer', '')

        # Get definition candidates - we need this for generate_anagram_hypotheses
        # For compound analysis, we can use a simple list with just the answer
        candidates = [answer] if answer else []

        if not candidates:
            return {}

        # Use the SAME approach as evidence_analysis.py (which works perfectly)
        enumeration_num = len(answer.replace(' ', '')) if answer else 0

        # Call the working anagram function that evidence_analysis.py uses
        hypotheses = generate_anagram_hypotheses(clue_text, enumeration_num, candidates)

        if not hypotheses:
            return {}

        # Take the best (first) hypothesis - same as evidence_analysis.py approach
        best_hypothesis = hypotheses[0]

        # Convert hypothesis to the format expected by the rest of the system
        return {
            'candidate': best_hypothesis.get('answer', ''),
            'fodder_words': best_hypothesis.get('fodder_words', []),
            'fodder_letters': best_hypothesis.get('fodder_letters', ''),
            'confidence': best_hypothesis.get('confidence', 0.5),
            'evidence_type': best_hypothesis.get('evidence_type',
                                                 best_hypothesis.get('solve_type',
                                                                     'exact')),
            'needed_letters': best_hypothesis.get('needed_letters', ''),
            'excess_letters': best_hypothesis.get('excess_letters', '')
        }

    def build_word_attribution(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete word attribution from case data."""
        clue_text = case.get('clue', '')
        clue_words = clue_text.split()

        # Extract core components
        definition_window = self.extract_definition_window(case)
        indicators = self.detect_indicators_readonly(clue_text)
        anagram_data = self.extract_anagram_data(case)

        # Build accounted words set
        accounted_words = set()

        # Add definition window words
        if definition_window:
            # Clean and add definition words
            def_words = [w.strip('.,!?:;()') for w in definition_window.split()]
            accounted_words.update(def_words)

        # Add anagram fodder words
        fodder_words = anagram_data.get('fodder_words', [])
        accounted_words.update(fodder_words)

        # Add anagram indicators
        anagram_indicators = indicators.get('anagram', [])
        accounted_words.update(anagram_indicators)

        # Add common link words that might connect components
        link_words = {'in', 'with', 'of', 'to', 'for', 'by', 'from', 'and', 'but', 'the',
                      'a'}
        for word in clue_words:
            clean_word = word.strip('.,!?:;()').lower()
            if clean_word in link_words:
                accounted_words.add(clean_word)

        # Calculate remaining words
        remaining_words = []
        for word in clue_words:
            clean_word = word.strip('.,!?:;()')
            if clean_word.lower() not in [w.lower() for w in accounted_words]:
                # Skip enumeration patterns like (6,3)
                if not (clean_word.startswith('(') and clean_word.endswith(')')):
                    remaining_words.append(clean_word)

        return {
            'clue': clue_text,
            'answer': case.get('answer_raw', ''),
            'definition_window': definition_window,
            'anagram_indicators': anagram_indicators,
            'anagram_data': anagram_data,
            'accounted_words': list(accounted_words),
            'remaining_words': remaining_words,
            'enumeration': case.get('enumeration', ''),
            # Preserve original case data
            'original_case': case
        }

    def enhance_pipeline_data(self, raw_pipeline_results: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        Extract complete word attribution from pipeline simulator results.
        Adds definition windows, indicators, and proper remaining words calculation.
        """
        enhanced_cases = []

        print("Enhancing pipeline data with complete word attribution...")

        for i, case in enumerate(raw_pipeline_results):
            # Only process cases with anagram hits
            if case.get('anagrams') and len(case['anagrams']) > 0:
                try:
                    enhanced_case = self.build_word_attribution(case)
                    enhanced_cases.append(enhanced_case)

                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1} cases...")

                except Exception as e:
                    print(f"Warning: Could not enhance case {i + 1}: {e}")
                    continue

        print(f"Enhanced {len(enhanced_cases)} cases with complete attribution.")
        return enhanced_cases

    def analyze_case_quality(self, enhanced_case: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of a case for explanation system development."""
        anagram_data = enhanced_case.get('anagram_data', {})
        fodder_letters = anagram_data.get('fodder_letters', '')
        answer = enhanced_case.get('answer', '')
        confidence = anagram_data.get('confidence', 0.0)
        remaining_words = enhanced_case.get('remaining_words', [])

        # Handle string confidence values (like 'provisional')
        if isinstance(confidence, str):
            confidence = 0.5  # Default numeric value for string confidences

        # Calculate fodder coverage
        fodder_coverage = len(fodder_letters) / len(answer) if answer else 0.0

        # Classify quality
        if fodder_coverage >= 0.8 and confidence >= 0.7:
            quality = "high"
        elif fodder_coverage >= 0.6 and confidence >= 0.5:
            quality = "medium"
        else:
            quality = "low"

        return {
            'quality': quality,
            'fodder_coverage': fodder_coverage,
            'confidence': confidence,
            'has_remaining_words': len(remaining_words) > 0,
            'remaining_word_count': len(remaining_words)
        }

    def build_explanations(self, enhanced_cases: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        Main explanation building logic using complete attribution data.
        """
        explanations = []

        print("\nBuilding systematic explanations...")

        for case in enhanced_cases:
            quality_analysis = self.analyze_case_quality(case)

            explanation = {
                'clue': case['clue'],
                'answer': case['answer'],
                'quality': quality_analysis['quality'],
                'definition_component': {
                    'text': case['definition_window'],
                    'contributes_to': case['answer']
                },
                'anagram_component': {
                    'indicator': case['anagram_indicators'],
                    'fodder': case['anagram_data'].get('fodder_words', []),
                    'letters_provided': case['anagram_data'].get('fodder_letters', ''),
                    'confidence': case['anagram_data'].get('confidence', 0.0)
                },
                'remaining_analysis': {
                    'words': case['remaining_words'],
                    'needs_additional_wordplay': len(case['remaining_words']) > 0,
                    'suggested_types': self.suggest_wordplay_types(
                        case['remaining_words'])
                },
                'word_attribution': {
                    'accounted_for': case['accounted_words'],
                    'remaining': case['remaining_words']
                }
            }

            explanations.append(explanation)

        return explanations

    def suggest_wordplay_types(self, remaining_words: List[str]) -> List[str]:
        """Suggest likely wordplay types for remaining words."""
        suggestions = []

        for word in remaining_words:
            word_lower = word.lower()

            # Common substitution candidates
            substitution_words = {
                'husband': 'H', 'wife': 'W', 'married': 'M', 'single': 'S',
                'right': 'R', 'left': 'L', 'king': 'K', 'queen': 'Q',
                'north': 'N', 'south': 'S', 'east': 'E', 'west': 'W'
            }

            if word_lower in substitution_words:
                suggestions.append('substitution')
            elif word_lower in ['back', 'backwards', 'reverse', 'reversed']:
                suggestions.append('reversal')
            elif word_lower in ['inside', 'in', 'within', 'containing']:
                suggestions.append('container')
            elif word_lower in ['around', 'about', 'circling']:
                suggestions.append('container')

        return suggestions

    def filter_by_quality(self, explanations: List[Dict[str, Any]],
                          min_quality: str = 'medium') -> List[Dict[str, Any]]:
        """Filter explanations by quality level."""
        quality_order = {'low': 0, 'medium': 1, 'high': 2}
        min_level = quality_order.get(min_quality, 1)

        return [exp for exp in explanations
                if quality_order.get(exp['quality'], 0) >= min_level]

    def export_explanations(self, explanations: List[Dict[str, Any]],
                            filepath: str = None) -> None:
        """Export explanations to JSON file."""
        if not filepath:
            filepath = 'explanation_results.json'

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(explanations, f, indent=2, ensure_ascii=False)

        print(f"Exported {len(explanations)} explanations to {filepath}")


def main():
    """Main execution function."""
    print("=== EXPLANATION SYSTEM BUILDER ===")

    # Initialize explanation system
    builder = ExplanationSystemBuilder()

    # TODO: Load pipeline results
    # For now, return placeholder
    print("Ready to process pipeline simulator results.")
    print("Next step: Load pipeline data and call enhance_pipeline_data()")

    return builder


if __name__ == "__main__":
    builder = main()