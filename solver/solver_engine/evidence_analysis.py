#!/usr/bin/env python3
"""
Evidence-based scoring analysis for unresolved clues.

This file maintains the absolute sanctity of the original pipeline simulator
by calling it first, then only applying evidence scoring to the remaining
cohort that didn't get wordplay hits.

Architecture:
1. Run original pipeline simulator (untouched)
2. Filter for unresolved clues (def candidates exist, no wordplay hits)
3. Apply evidence scoring only to that subset
4. Show ranking improvements
"""

import sys
import os

# Import the original pipeline simulator (maintaining sanctity)
from pipeline_simulator import run_pipeline_probe, MAX_CLUES, WORDPLAY_TYPE
from anagram_evidence_system import ComprehensiveWordplayDetector

# Evidence scoring configuration
ENABLE_EVIDENCE_SCORING = True
EVIDENCE_SCORE_WEIGHT = 1.0


class EvidenceAnalyzer:
    """Analyzes unresolved clues using evidence-based scoring."""

    def __init__(self):
        """Initialize the evidence detector."""
        self.detector = None
        if ENABLE_EVIDENCE_SCORING:
            try:
                # Load comprehensive detector with database indicators
                db_path = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
                self.detector = ComprehensiveWordplayDetector(db_path=db_path)
                print("Evidence detector loaded successfully.")
            except Exception as e:
                print(f"WARNING: Evidence detector failed to load: {e}")
                self.detector = None

    def is_unresolved_clue(self, record):
        """
        Check if a clue is unresolved (has definition candidates but no wordplay hits).

        Unresolved criteria:
        - Has definition candidates
        - No anagram hits (anagram_hits = 0)
        - No lurker hits (lurker_hits = 0)
        - No DD hits (double_definition_hits = 0)
        """
        summary = record.get("summary", {})

        has_def_candidates = summary.get("definition_candidates", 0) > 0
        no_anagram_hits = summary.get("anagram_hits", 0) == 0
        no_lurker_hits = summary.get("lurker_hits", 0) == 0
        no_dd_hits = summary.get("double_definition_hits", 0) == 0

        return has_def_candidates and no_anagram_hits and no_lurker_hits and no_dd_hits

    def apply_evidence_scoring(self, record):
        """
        Apply evidence-based scoring to unresolved clue.
        Returns enhanced record with scoring information.
        """
        if not self.detector:
            return record

        clue_text = record["clue"]
        candidates = record["definition_candidates"]
        answer = record["answer"]

        if not candidates:
            return record

        # Analyze for anagram evidence
        evidence_list = self.detector.analyze_clue_for_anagram_evidence(
            clue_text, candidates, enumeration=record.get("enumeration"))

        # Create scored candidates list
        scored_candidates = []
        evidence_by_candidate = {ev.candidate.upper(): ev for ev in evidence_list}

        for candidate in candidates:
            candidate_upper = candidate.upper()
            evidence = evidence_by_candidate.get(candidate_upper)

            # Calculate evidence score boost
            evidence_score = 0.0
            if evidence:
                evidence_score = self.detector.calculate_anagram_score_boost(evidence)

            scored_candidates.append({
                "candidate": candidate,
                "evidence_score": evidence_score,
                "evidence": evidence
            })

        # Sort by evidence score (highest first)
        scored_candidates.sort(key=lambda x: x["evidence_score"], reverse=True)

        # Find answer ranking in scored list
        answer_rank_evidence = None
        for i, scored in enumerate(scored_candidates, 1):
            if scored["candidate"].upper() == answer.upper():
                answer_rank_evidence = i
                break

        # Find original answer ranking (unscored)
        answer_rank_original = None
        for i, candidate in enumerate(candidates, 1):
            if candidate.upper() == answer.upper():
                answer_rank_original = i
                break

        # Enhance record with evidence analysis
        enhanced_record = record.copy()
        enhanced_record["evidence_analysis"] = {
            "evidence_found": len(evidence_list),
            "scored_candidates": scored_candidates[:10],  # Top 10
            "answer_rank_original": answer_rank_original,
            "answer_rank_evidence": answer_rank_evidence,
            "ranking_improved": (answer_rank_evidence and answer_rank_original and
                                 answer_rank_evidence < answer_rank_original)
        }

        return enhanced_record

    def analyze_unresolved_cohort(self, results):
        """
        Analyze the unresolved cohort from pipeline simulator results.
        Returns enhanced results with evidence scoring.
        """
        unresolved_clues = [r for r in results if self.is_unresolved_clue(r)]

        print(f"\n🔍 UNRESOLVED COHORT ANALYSIS:")
        print(f"Total clues processed: {len(results)}")
        print(f"Unresolved clues (no wordplay hits): {len(unresolved_clues)}")

        if not unresolved_clues:
            print("No unresolved clues to analyze.")
            return []

        # Apply evidence scoring to unresolved clues
        enhanced_results = []
        evidence_improvements = 0

        for record in unresolved_clues:
            enhanced = self.apply_evidence_scoring(record)
            enhanced_results.append(enhanced)

            # Count improvements
            evidence_analysis = enhanced.get("evidence_analysis", {})
            if evidence_analysis.get("ranking_improved", False):
                evidence_improvements += 1

        print(f"Clues with evidence improvements: {evidence_improvements}")

        return enhanced_results


def display_evidence_results(enhanced_results, max_display=10):
    """Display evidence analysis results."""

    # Sort by evidence improvements first, then by evidence found
    # Ensure all values have proper defaults to avoid None comparison errors
    display_results = sorted(enhanced_results,
                             key=lambda r: (
                                 r.get("evidence_analysis", {}).get(
                                     "ranking_improved") or False,
                                 r.get("evidence_analysis", {}).get("evidence_found") or 0
                             ),
                             reverse=True)

    print(f"\n📊 EVIDENCE ANALYSIS RESULTS (Top {max_display}):")
    print("=" * 80)

    for i, record in enumerate(display_results[:max_display], 1):
        evidence_analysis = record.get("evidence_analysis", {})

        print(f"\n[{i}] CLUE: {record['clue']}")
        print(f"    TYPE: {record['wordplay_type']}")
        print(f"    ANSWER: {record['answer_raw']}")

        # Show ranking change
        orig_rank = evidence_analysis.get("answer_rank_original")
        evid_rank = evidence_analysis.get("answer_rank_evidence")
        improved = evidence_analysis.get("ranking_improved", False)

        if orig_rank and evid_rank:
            improvement_text = "IMPROVED" if improved else "unchanged"
            print(f"    RANKING: {orig_rank} → {evid_rank} ({improvement_text})")

        # Show evidence found
        evidence_found = evidence_analysis.get("evidence_found", 0)
        print(f"    EVIDENCE: {evidence_found} candidates with evidence")

        # Show top evidence candidates (top 5 unique score levels)
        scored_candidates = evidence_analysis.get("scored_candidates", [])
        if scored_candidates:
            print("    TOP EVIDENCE CANDIDATES:")

            # Extract all unique scores from the full list and sort them
            all_scores = []
            for scored in scored_candidates:
                score = round(scored["evidence_score"], 1)
                if score not in all_scores:
                    all_scores.append(score)

            # Sort scores in descending order and take top 5
            all_scores.sort(reverse=True)
            top_5_scores = all_scores[:5]

            print(
                f"    DEBUG: Found {len(all_scores)} unique scores, showing top 5: {top_5_scores}")

            # Show all candidates with those top 5 score levels (normalized to remove case duplicates)
            display_count = 0
            seen_candidates = set()  # Track normalized candidates to avoid duplicates

            for scored in scored_candidates:
                score = round(scored["evidence_score"], 1)
                if score in top_5_scores:
                    candidate = scored["candidate"]
                    candidate_normalized = candidate.upper().strip()  # Normalize for duplicate checking

                    # Skip if we've already shown this normalized candidate
                    if candidate_normalized in seen_candidates:
                        continue

                    seen_candidates.add(candidate_normalized)
                    display_count += 1
                    evidence = scored["evidence"]

                    marker = "★" if candidate.upper() == record["answer"].upper() else " "
                    evidence_marker = "🔤" if evidence else "  "

                    print(
                        f"    {marker}{evidence_marker} {display_count:2d}. {candidate:15} (evidence: +{score:.1f})")

                    if evidence:
                        print(
                            f"          → {evidence.evidence_type}: {' + '.join(evidence.fodder_words)}")

                    # Limit total display to avoid overwhelming output
                    if display_count >= 15:
                        remaining = sum(1 for s in scored_candidates
                                        if round(s["evidence_score"], 1) in top_5_scores
                                        and s[
                                            "candidate"].upper().strip() not in seen_candidates)
                        if remaining > 0:
                            print(
                                f"    ... and {remaining} more unique candidates with these score levels")
                        break


def main():
    """Main analysis function."""
    print("🔧 EVIDENCE-BASED SCORING ANALYSIS")
    print("=" * 60)
    print("Maintaining absolute sanctity of original pipeline simulator")
    print("Only analyzing unresolved cohort (no wordplay hits)")
    print("=" * 60)

    # Initialize evidence analyzer
    analyzer = EvidenceAnalyzer()

    # Step 1: Run original pipeline simulator (with appropriate settings for evidence analysis)
    print("\n📋 STEP 1: Running original pipeline simulator...")

    # Override the ONLY_MISSING_DEFINITION setting for evidence analysis
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

    # Step 2: Analyze unresolved cohort only
    print("\n🔍 STEP 2: Analyzing unresolved cohort...")
    enhanced_results = analyzer.analyze_unresolved_cohort(results)

    # Step 3: Display evidence analysis
    if enhanced_results:
        display_evidence_results(enhanced_results)

    print("\n✅ Analysis complete. Original pipeline simulator untouched.")


if __name__ == "__main__":
    main()