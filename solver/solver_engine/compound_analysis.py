#!/usr/bin/env python3
"""
Compound wordplay analysis for cryptic crossword solver.

This file analyzes cases where anagram detection finds partial matches
that need additional wordplay techniques (substitution, reversal, etc.)
to complete the construction.

Architecture:
1. Run original pipeline simulator (untouched)
2. Filter for clues with anagram hits but remaining unused words
3. Apply ExplanationSystemBuilder for proper word attribution
4. Show compound wordplay opportunities
"""

import sys
import os

# Add project root to path
sys.path.append(r'C:\Users\shute\PycharmProjects\cryptic_solver')

# Import the original pipeline simulator (maintaining sanctity)
from pipeline_simulator import run_pipeline_probe, MAX_CLUES, WORDPLAY_TYPE
from solver.wordplay.anagram.compound_wordplay_analyzer import ExplanationSystemBuilder

# Import EvidenceAnalyzer to apply evidence ranking to compound candidates
sys.path.append(os.path.join(os.path.dirname(__file__)))
from evidence_analysis import EvidenceAnalyzer


class CompoundAnalyzer:
    """Analyzes compound wordplay opportunities using proper word attribution."""

    def __init__(self):
        """Initialize the explanation system builder."""
        self.explanation_builder = None
        try:
            self.explanation_builder = ExplanationSystemBuilder()
            print("Explanation system builder loaded successfully.")
        except Exception as e:
            print(f"WARNING: Explanation builder failed to load: {e}")
            self.explanation_builder = None

    def is_compound_candidate(self, record):
        """
        Check if a clue is a compound candidate:
        - Has anagram hits (partial or complete)
        - Still has meaningful unused words for additional wordplay
        """
        summary = record.get("summary", {})
        has_anagram_hits = summary.get("anagram_hits", 0) > 0

        if not has_anagram_hits:
            return False

        # Check if anagram hits have meaningful unused words
        anagrams = record.get('anagrams', [])
        for anagram_hit in anagrams:
            unused_words = anagram_hit.get('unused_words', [])
            meaningful_unused = [w for w in unused_words
                                 if len(w) > 2 and not w.replace(',', '').replace('-',
                                                                                  '').isdigit()]
            if len(meaningful_unused) >= 1:  # At least 1 meaningful remaining word
                return True

        return False

    def analyze_compound_cohort(self, results):
        """
        Analyze compound candidates using evidence-ranked results.
        UPDATED: Now works with evidence-analyzed results instead of raw pipeline data.
        """
        compound_candidates = [r for r in results if self.is_compound_candidate(r)]

        print(f"\n🧩 COMPOUND WORDPLAY COHORT ANALYSIS:")
        print(f"Total clues processed: {len(results)}")
        print(
            f"Compound candidates (anagram hits with remaining words): {len(compound_candidates)}")

        if not compound_candidates:
            print("No compound candidates found.")
            return []

        # Apply evidence analysis to get ranked candidates for each clue
        evidence_analyzer = EvidenceAnalyzer()
        evidence_enhanced_results = []

        print("Applying evidence analysis to compound candidates...")
        for i, record in enumerate(compound_candidates):
            try:
                # Apply evidence analysis to get complete ranked candidate information
                enhanced_record = evidence_analyzer.apply_evidence_scoring(record,
                                                                           debug=False)
                evidence_enhanced_results.append(enhanced_record)

                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1} compound candidates...")

            except Exception as e:
                print(f"Warning: Could not analyze compound candidate {i + 1}: {e}")
                continue

        # Now work with evidence-enhanced results for compound analysis
        if not self.explanation_builder:
            print("\nCompound analysis disabled - explanation builder not available.")
            return []

        # Use ExplanationSystemBuilder with evidence-enhanced data
        try:
            # Step 1: Apply enhance_pipeline_data to add definition windows and word attribution
            enhanced_cases = self.explanation_builder.enhance_pipeline_data(
                evidence_enhanced_results)

            # Step 2: Build systematic explanations from enhanced cases
            explanations = self.explanation_builder.build_explanations(enhanced_cases)

            # Step 3: Filter for cases with meaningful remaining words
            compound_explanations = [exp for exp in explanations
                                     if exp.get('remaining_analysis', {}).get(
                    'needs_additional_wordplay', False)]

            print(f"Cases with evidence analysis: {len(evidence_enhanced_results)}")
            print(f"Cases with enhanced attribution: {len(enhanced_cases)}")
            print(f"Cases needing additional wordplay: {len(compound_explanations)}")

            return compound_explanations

        except Exception as e:
            print(f"Error in compound analysis: {e}")
            return []

    def build_compound_analysis(self, record):
        """Build compound analysis using existing ranked anagram hits."""
        clue_text = record.get('clue', '')
        answer = record.get('answer', '')
        anagram_hits = record.get('anagrams', [])

        if not anagram_hits:
            return None

        # Use the BEST (first) anagram hit - this is already ranked by pipeline
        best_hit = anagram_hits[0]

        # Extract components from ranked hit
        fodder_words = best_hit.get('fodder_words', [])
        fodder_letters = best_hit.get('fodder_letters', '')
        unused_words = best_hit.get('unused_words', [])
        confidence = best_hit.get('confidence', 0.5)

        # Convert string confidence to numeric
        if isinstance(confidence, str):
            confidence_map = {'provisional': 0.5, 'high': 0.9, 'medium': 0.7, 'low': 0.3}
            confidence = confidence_map.get(confidence.lower(), 0.5)

        # Filter meaningful unused words
        meaningful_unused = [w for w in unused_words
                             if len(w) > 2 and not w.replace(',', '').replace('-',
                                                                              '').isdigit()]

        # Determine quality based on coverage
        letter_coverage = len(fodder_letters) / len(
            answer.replace(' ', '')) if answer else 0.0

        if letter_coverage >= 0.8 and confidence >= 0.7:
            quality = "high"
        elif letter_coverage >= 0.6 and confidence >= 0.5:
            quality = "medium"
        else:
            quality = "low"

        # Build explanation structure
        explanation = {
            'clue': clue_text,
            'answer': answer,
            'quality': quality,
            'definition_component': {
                'text': 'Auto-detected',  # Simplified for now
                'contributes_to': answer
            },
            'anagram_component': {
                'indicator': [],  # Simplified for now
                'fodder': fodder_words,
                'letters_provided': fodder_letters,
                'confidence': confidence
            },
            'remaining_analysis': {
                'words': meaningful_unused,
                'needs_additional_wordplay': len(meaningful_unused) > 0,
                'suggested_types': self.suggest_wordplay_types(meaningful_unused)
            },
            'word_attribution': {
                'accounted_for': fodder_words,
                'remaining': meaningful_unused
            }
        }

        return explanation

    def suggest_wordplay_types(self, remaining_words):
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


def display_compound_results(compound_results, max_display=10):
    """Display compound wordplay analysis results."""

    if not compound_results:
        print("\nNo compound wordplay results to display.")
        return

    # Sort by quality and then by number of remaining words
    quality_order = {'high': 3, 'medium': 2, 'low': 1}
    sorted_results = sorted(compound_results,
                            key=lambda x: (
                                quality_order.get(x.get('quality', 'low'), 1),
                                len(x.get('remaining_analysis', {}).get('words', []))
                            ), reverse=True)

    print(f"\n🧩 COMPOUND WORDPLAY ANALYSIS RESULTS (Top {max_display}):")
    print("=" * 80)

    for i, result in enumerate(sorted_results[:max_display], 1):
        print(f"\n[{i}] CLUE: {result['clue']}")
        print(f"    ANSWER: {result['answer']}")
        print(f"    QUALITY: {result.get('quality', 'unknown')}")

        # Definition component
        def_comp = result.get('definition_component', {})
        def_text = def_comp.get('text', 'None identified')
        print(f"    DEFINITION: {def_text}")

        # Anagram component
        anag_comp = result.get('anagram_component', {})
        anag_indicators = anag_comp.get('indicator', [])
        anag_fodder = anag_comp.get('fodder', [])
        anag_letters = anag_comp.get('letters_provided', '')
        anag_confidence = anag_comp.get('confidence', 0.0)

        # Format confidence value (handle both string and numeric)
        if isinstance(anag_confidence, str):
            confidence_display = anag_confidence
        else:
            confidence_display = f"{anag_confidence:.3f}"

        print(f"    ANAGRAM COMPONENT:")
        print(
            f"      Indicators: {anag_indicators if anag_indicators else 'None detected'}")
        print(f"      Fodder: {anag_fodder} → {anag_letters}")
        print(f"      Confidence: {confidence_display}")

        # Remaining analysis (the key part for compound wordplay)
        remaining_analysis = result.get('remaining_analysis', {})
        remaining_words = remaining_analysis.get('words', [])
        suggested_types = remaining_analysis.get('suggested_types', [])

        print(f"    REMAINING FOR COMPOUND ANALYSIS:")
        print(f"      Words: {remaining_words}")
        if suggested_types:
            print(f"      Suggested wordplay types: {suggested_types}")

        # Word attribution summary
        word_attr = result.get('word_attribution', {})
        accounted = word_attr.get('accounted_for', [])
        print(f"    WORD ATTRIBUTION:")
        print(f"      Accounted for: {accounted}")
        print(f"      Available for compound: {remaining_words}")

        print("-" * 80)


def main():
    """Main analysis function."""
    print("🧩 COMPOUND WORDPLAY ANALYSIS")
    print("=" * 60)
    print("Maintaining absolute sanctity of original pipeline simulator")
    print("Analyzing anagram hits with remaining words for compound wordplay")
    print("=" * 60)

    # Initialize compound analyzer
    analyzer = CompoundAnalyzer()

    # Step 1: Run original pipeline simulator
    print("\n📋 STEP 1: Running original pipeline simulator...")

    # Override the ONLY_MISSING_DEFINITION setting for analysis
    # We need clues where the answer IS in definition candidates
    import pipeline_simulator
    original_setting = pipeline_simulator.ONLY_MISSING_DEFINITION
    pipeline_simulator.ONLY_MISSING_DEFINITION = False  # We want answer in def candidates

    try:
        results, overall = run_pipeline_probe()

        # Restore original setting
        pipeline_simulator.ONLY_MISSING_DEFINITION = original_setting

    except Exception as e:
        # Restore original setting even if error occurs
        pipeline_simulator.ONLY_MISSING_DEFINITION = original_setting
        raise e

    # Show original results summary
    print("\n📊 ORIGINAL PIPELINE RESULTS:")
    print(f"  clues processed           : {overall['clues']}")
    print(f"  clues w/ def answer match : {overall['clues_with_def_match']}")
    print(f"  clues w/ anagram hit      : {overall['clues_with_anagram']}")
    print(f"  clues w/ lurker hit       : {overall['clues_with_lurker']}")
    print(f"  clues w/ DD hit           : {overall['clues_with_dd']}")

    # Step 2: Analyze compound wordplay cohort
    print("\n🧩 STEP 2: Analyzing compound wordplay cohort...")
    enhanced_results = analyzer.analyze_compound_cohort(results)

    # Step 3: Display compound analysis results
    if enhanced_results:
        display_compound_results(enhanced_results)

    print("\n✅ Analysis complete. Original pipeline simulator untouched.")


if __name__ == "__main__":
    main()