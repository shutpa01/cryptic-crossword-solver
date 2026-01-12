#!/usr/bin/env python3
"""
Compound Wordplay Analyzer Engine - CORRECTED VERSION

This engine uses ranked candidates from evidence analysis instead of doing anagram detection.
Receives evidence-enhanced cases and builds compound explanations from them.
"""

import sys
import os
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(r'C:\Users\shute\PycharmProjects\cryptic_solver')


class ExplanationSystemBuilder:
    """
    Builds systematic explanations using ranked candidates from evidence analysis.
    NO anagram detection - works with pre-ranked evidence results.
    """

    def __init__(self):
        """Initialize explanation builder - no anagram detection needed."""
        pass

    def enhance_pipeline_data(self, evidence_enhanced_cases: List[Dict[str, Any]]) -> \
    List[Dict[str, Any]]:
        """
        Enhance cases with complete word attribution using evidence analysis results.
        NO anagram detection - uses ranked candidates already provided.
        """
        enhanced_cases = []

        print("Enhancing pipeline data with complete word attribution...")

        for i, case in enumerate(evidence_enhanced_cases):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} cases...")

            try:
                enhanced_case = self.build_complete_attribution(case)
                enhanced_cases.append(enhanced_case)
            except Exception as e:
                print(f"Warning: Could not enhance case {i + 1}: {e}")
                continue

        print(f"Enhanced {len(enhanced_cases)} cases with complete attribution.")
        return enhanced_cases

    def build_complete_attribution(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build complete word attribution using evidence analysis results.
        Uses top ranked candidate from evidence analysis - no detection needed.
        """
        clue_text = case.get('clue', '')
        clue_words = clue_text.split()
        answer = case.get('answer', '')

        # Extract evidence analysis results (already ranked by evidence analysis)
        evidence_analysis = case.get('evidence_analysis', {})
        scored_candidates = evidence_analysis.get('scored_candidates', [])

        if not scored_candidates:
            # Fallback for cases without evidence analysis
            return self.build_fallback_attribution(case, clue_words)

        # Use the top ranked candidate from evidence analysis (rank #1)
        top_candidate = scored_candidates[0]
        evidence = top_candidate.get('evidence')

        if not evidence:
            return self.build_fallback_attribution(case, clue_words)

        # Extract components from ranked evidence (no detection needed)
        candidate_word = evidence.candidate
        fodder_words = evidence.fodder_words or []
        evidence_type = evidence.evidence_type

        # Account for words used in the anagram component
        accounted_words = set()

        # Add anagram fodder words to accounted words
        for fodder_word in fodder_words:
            # Find this word in the clue (case insensitive matching)
            for clue_word in clue_words:
                if clue_word.lower() == fodder_word.lower():
                    accounted_words.add(clue_word)

        # Simple definition detection (uses pipeline data when available)
        definition_window = self.detect_definition_window(case, clue_words,
                                                          candidate_word)
        if definition_window:
            def_words = definition_window.split()
            for def_word in def_words:
                if def_word in clue_words:
                    accounted_words.add(def_word)

        # Add common link words to accounted
        link_words = {'of', 'in', 'the', 'a', 'an', 'to', 'for', 'with', 'and', 'or'}
        for word in clue_words:
            if word.lower() in link_words:
                accounted_words.add(word)

        # Find remaining words for compound analysis
        remaining_words = [w for w in clue_words if w not in accounted_words]

        # Build enhanced case with attribution
        enhanced_case = case.copy()
        enhanced_case.update({
            'anagram_data': {
                'candidate': candidate_word,
                'fodder_words': fodder_words,
                'evidence_type': evidence_type,
                'confidence': evidence.confidence,
                'fodder_letters': evidence.fodder_letters  # For display compatibility
            },
            'definition_window': definition_window or "None identified",
            'anagram_indicators': self.extract_indicators_from_evidence(evidence),
            'accounted_words': list(accounted_words),
            'remaining_words': remaining_words
        })

        return enhanced_case

    def build_fallback_attribution(self, case: Dict[str, Any], clue_words: List[str]) -> \
    Dict[str, Any]:
        """Fallback attribution when no evidence analysis available."""
        enhanced_case = case.copy()
        enhanced_case.update({
            'anagram_data': {
                'candidate': case.get('answer', ''),
                'fodder_words': [],
                'evidence_type': 'unknown',
                'confidence': 0.0,
                'fodder_letters': ''
            },
            'definition_window': "None identified",
            'anagram_indicators': [],
            'accounted_words': [],
            'remaining_words': clue_words
        })
        return enhanced_case

    def detect_definition_window(self, case: Dict[str, Any], clue_words: List[str],
                                 candidate_word: str) -> Optional[str]:
        """
        Simple definition detection using pipeline data when available.
        Can be improved with proper definition detection.
        """
        # Try to use window_support from pipeline if available
        window_support = case.get('window_support', {})
        if window_support:
            answer = case.get('answer', '').upper()
            for window_text, candidates in window_support.items():
                if isinstance(candidates, list):
                    normalized_candidates = [c.upper().replace(' ', '') for c in
                                             candidates]
                    normalized_answer = answer.replace(' ', '')
                    if normalized_answer in normalized_candidates:
                        return window_text

        # Fallback: simple heuristic (take last 1-2 words)
        if len(clue_words) >= 2:
            potential_def = ' '.join(clue_words[-2:])
            return potential_def
        elif len(clue_words) >= 1:
            return clue_words[-1]

        return None

    def extract_indicators_from_evidence(self, evidence) -> List[str]:
        """Extract indicator information from evidence if available."""
        # This would be enhanced to extract actual indicators
        # For now, return empty list as indicators aren't critical for compound analysis
        return []

    def analyze_case_quality(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of a compound candidate."""
        anagram_data = case.get('anagram_data', {})
        remaining_words = case.get('remaining_words', [])

        # Quality based on confidence and remaining word count
        confidence = anagram_data.get('confidence', 0.0)
        num_remaining = len(remaining_words)

        # Convert string confidence to numeric
        if isinstance(confidence, str):
            confidence_map = {'high': 0.9, 'medium': 0.7, 'provisional': 0.5, 'low': 0.3}
            confidence = confidence_map.get(confidence.lower(), 0.5)

        # Quality scoring based on confidence and remaining words
        if confidence >= 0.7 and num_remaining >= 2:
            quality = "high"
        elif confidence >= 0.5 and num_remaining >= 1:
            quality = "medium"
        else:
            quality = "low"

        return {
            'quality': quality,
            'confidence': confidence,
            'remaining_word_count': num_remaining
        }

    def build_explanations(self, enhanced_cases: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        Build systematic explanations using complete attribution data.
        """
        explanations = []

        print("Building systematic explanations...")

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
                    'letters_provided': case['anagram_data'].get('candidate',
                                                                 case['answer']),
                    # Show final answer
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

        return list(set(suggestions))  # Remove duplicates