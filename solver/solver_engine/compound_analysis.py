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
        Analyze compound candidates using ExplanationSystemBuilder workflow.
        Returns enhanced results with proper word attribution analysis.
        """
        if not self.explanation_builder:
            print("\nCompound analysis disabled - explanation builder not available.")
            return []

        compound_candidates = [r for r in results if self.is_compound_candidate(r)]

        print(f"\n🧩 COMPOUND WORDPLAY COHORT ANALYSIS:")
        print(f"Total clues processed: {len(results)}")
        print(
            f"Compound candidates (anagram hits with remaining words): {len(compound_candidates)}")

        if not compound_candidates:
            print("No compound candidates found.")
            return []

        # Use ExplanationSystemBuilder workflow
        try:
            # Step 1: Enhance pipeline data with proper word attribution
            enhanced_cases = self.explanation_builder.enhance_pipeline_data(
                compound_candidates)

            # Step 2: Build systematic explanations
            explanations = self.explanation_builder.build_explanations(enhanced_cases)

            # Step 3: Filter for cases with meaningful remaining words
            compound_explanations = [exp for exp in explanations
                                     if exp['remaining_analysis'][
                                         'needs_additional_wordplay']]

            print(f"Cases with proper word attribution: {len(enhanced_cases)}")
            print(f"Cases needing additional wordplay: {len(compound_explanations)}")

            return compound_explanations

        except Exception as e:
            print(f"Error in compound analysis: {e}")
            return []


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