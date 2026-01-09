#!/usr/bin/env python3
"""
Automated Issue Detection for Evidence Analysis

Runs evidence analysis and automatically categorizes problems for batch analysis:
- False positives (wrong answers with high evidence scores)
- Missed opportunities (correct answers with low/no evidence)
- Suspicious patterns (high scores, coherence issues)
- Ranking anomalies

This allows systematic pattern analysis instead of manual case-by-case review.
"""

import sys
import os
from collections import defaultdict

# Import the evidence analysis system
from evidence_analysis import EvidenceAnalyzer
from pipeline_simulator import run_pipeline_probe


class IssueDetector:
    """Automatically detect and categorize evidence analysis issues."""

    def __init__(self):
        """Initialize the issue detector."""
        self.analyzer = EvidenceAnalyzer()
        self.issues = {
            'false_positives': [],  # Wrong answers with high evidence
            'missed_opportunities': [],  # Correct answers with low/no evidence
            'suspicious_high_scores': [],  # Evidence scores >15
            'coherence_issues': [],  # Multi-word beats single-word
            'ranking_anomalies': [],  # Weird ranking patterns
            'top_improvements': [],  # Biggest ranking jumps
        }

    def analyze_batch(self, max_clues=500):
        """Run evidence analysis and categorize issues."""

        print("🔍 AUTOMATED ISSUE DETECTION")
        print("=" * 60)

        # Step 1: Run original pipeline + evidence analysis
        print("📋 Running pipeline analysis...")

        # Override settings for comprehensive analysis
        import pipeline_simulator
        original_setting = pipeline_simulator.ONLY_MISSING_DEFINITION
        pipeline_simulator.ONLY_MISSING_DEFINITION = False

        try:
            results, overall = run_pipeline_probe(max_clues=max_clues)
            pipeline_simulator.ONLY_MISSING_DEFINITION = original_setting
        except Exception as e:
            pipeline_simulator.ONLY_MISSING_DEFINITION = original_setting
            raise e

        # Step 2: Analyze unresolved cohort
        unresolved_results = self.analyzer.analyze_unresolved_cohort(results)

        # Step 3: Categorize issues
        print("\n🔍 Categorizing issues...")
        self.categorize_issues(unresolved_results)

        # Step 4: Generate reports
        self.generate_issue_reports()

        return self.issues

    def categorize_issues(self, enhanced_results):
        """Categorize different types of issues found."""

        for record in enhanced_results:
            evidence_analysis = record.get("evidence_analysis", {})
            scored_candidates = evidence_analysis.get("scored_candidates", [])
            answer = record["answer"].upper()

            if not scored_candidates:
                continue

            # Find correct answer in scored list
            answer_position = None
            answer_evidence_score = 0
            for i, scored in enumerate(scored_candidates, 1):
                if scored["candidate"].upper() == answer:
                    answer_position = i
                    answer_evidence_score = scored["evidence_score"]
                    break

            top_candidate = scored_candidates[0]
            top_score = top_candidate["evidence_score"]

            # Issue 1: False Positives (wrong answer #1 with high evidence)
            if (top_candidate["candidate"].upper() != answer and
                    top_score > 8.0):
                self.issues['false_positives'].append({
                    'clue': record['clue'],
                    'answer': record['answer_raw'],
                    'wrong_winner': top_candidate["candidate"],
                    'wrong_score': top_score,
                    'wrong_evidence': self._format_evidence(top_candidate["evidence"]),
                    'answer_position': answer_position,
                    'answer_score': answer_evidence_score
                })

            # Issue 2: Missed Opportunities (correct answer low/no evidence)
            if answer_position and answer_position > 5 and answer_evidence_score < 5.0:
                self.issues['missed_opportunities'].append({
                    'clue': record['clue'],
                    'answer': record['answer_raw'],
                    'answer_position': answer_position,
                    'answer_score': answer_evidence_score,
                    'top_candidate': top_candidate["candidate"],
                    'top_score': top_score
                })

            # Issue 3: Suspicious High Scores (>15)
            if top_score > 15.0:
                self.issues['suspicious_high_scores'].append({
                    'clue': record['clue'],
                    'answer': record['answer_raw'],
                    'candidate': top_candidate["candidate"],
                    'score': top_score,
                    'evidence': self._format_evidence(top_candidate["evidence"])
                })

            # Issue 4: Coherence Issues (multi-word beats single-word unfairly)
            if len(scored_candidates) >= 2:
                first = scored_candidates[0]
                second = scored_candidates[1]

                if (first["evidence"] and second["evidence"] and
                        len(first["evidence"].fodder_words) > len(
                            second["evidence"].fodder_words) and
                        first["evidence_score"] > second["evidence_score"] and
                        abs(first["evidence_score"] - second["evidence_score"]) > 2.0):
                    self.issues['coherence_issues'].append({
                        'clue': record['clue'],
                        'answer': record['answer_raw'],
                        'multi_word_candidate': first["candidate"],
                        'multi_word_score': first["evidence_score"],
                        'multi_word_fodder': first["evidence"].fodder_words,
                        'single_word_candidate': second["candidate"],
                        'single_word_score': second["evidence_score"],
                        'single_word_fodder': second["evidence"].fodder_words
                    })

            # Issue 5: Top Improvements (biggest ranking jumps)
            orig_rank = evidence_analysis.get("answer_rank_original")
            evid_rank = evidence_analysis.get("answer_rank_evidence")

            if orig_rank and evid_rank and orig_rank > evid_rank:
                improvement = orig_rank - evid_rank
                self.issues['top_improvements'].append({
                    'clue': record['clue'],
                    'answer': record['answer_raw'],
                    'original_rank': orig_rank,
                    'evidence_rank': evid_rank,
                    'improvement': improvement,
                    'evidence_score': answer_evidence_score
                })

    def _format_evidence(self, evidence):
        """Format evidence for display."""
        if not evidence:
            return "none"
        return f"{evidence.evidence_type}: {' + '.join(evidence.fodder_words)}"

    def generate_issue_reports(self):
        """Generate detailed reports for each issue category."""

        print(f"\n📊 ISSUE DETECTION SUMMARY:")
        print(f"  False positives (wrong #1): {len(self.issues['false_positives'])}")
        print(f"  Missed opportunities: {len(self.issues['missed_opportunities'])}")
        print(f"  Suspicious high scores: {len(self.issues['suspicious_high_scores'])}")
        print(f"  Coherence issues: {len(self.issues['coherence_issues'])}")
        print(f"  Top improvements: {len(self.issues['top_improvements'])}")

        # Sort issues by severity/interest
        self.issues['false_positives'].sort(key=lambda x: x['wrong_score'], reverse=True)
        self.issues['missed_opportunities'].sort(key=lambda x: x['answer_position'],
                                                 reverse=True)
        self.issues['suspicious_high_scores'].sort(key=lambda x: x['score'], reverse=True)
        self.issues['top_improvements'].sort(key=lambda x: x['improvement'], reverse=True)

        print("\n" + "=" * 80)
        print("🚨 TOP 10 FALSE POSITIVES (Wrong answers with high evidence)")
        print("=" * 80)
        for i, issue in enumerate(self.issues['false_positives'][:10], 1):
            print(f"{i:2d}. CLUE: {issue['clue']}")
            print(
                f"    ANSWER: {issue['answer']} (position {issue['answer_position']}, score {issue['answer_score']:.1f})")
            print(
                f"    WRONG WINNER: {issue['wrong_winner']} (score {issue['wrong_score']:.1f})")
            print(f"    EVIDENCE: {issue['wrong_evidence']}")
            print()

        print("\n" + "=" * 80)
        print("💔 TOP 10 MISSED OPPORTUNITIES (Correct answers with low evidence)")
        print("=" * 80)
        for i, issue in enumerate(self.issues['missed_opportunities'][:10], 1):
            print(f"{i:2d}. CLUE: {issue['clue']}")
            print(
                f"    ANSWER: {issue['answer']} (position {issue['answer_position']}, score {issue['answer_score']:.1f})")
            print(
                f"    TOP CANDIDATE: {issue['top_candidate']} (score {issue['top_score']:.1f})")
            print()

        print("\n" + "=" * 80)
        print("🔥 TOP 10 SUSPICIOUS HIGH SCORES (Evidence >15)")
        print("=" * 80)
        for i, issue in enumerate(self.issues['suspicious_high_scores'][:10], 1):
            print(f"{i:2d}. CLUE: {issue['clue']}")
            print(f"    ANSWER: {issue['answer']}")
            print(f"    HIGH SCORER: {issue['candidate']} (score {issue['score']:.1f})")
            print(f"    EVIDENCE: {issue['evidence']}")
            print()

        print("\n" + "=" * 80)
        print("🧩 TOP 10 COHERENCE ISSUES (Multi-word beats single-word)")
        print("=" * 80)
        for i, issue in enumerate(self.issues['coherence_issues'][:10], 1):
            print(f"{i:2d}. CLUE: {issue['clue']}")
            print(f"    ANSWER: {issue['answer']}")
            print(
                f"    MULTI-WORD: {issue['multi_word_candidate']} (score {issue['multi_word_score']:.1f}) - {issue['multi_word_fodder']}")
            print(
                f"    SINGLE-WORD: {issue['single_word_candidate']} (score {issue['single_word_score']:.1f}) - {issue['single_word_fodder']}")
            print()

        print("\n" + "=" * 80)
        print("🚀 TOP 20 BIGGEST IMPROVEMENTS (Ranking jumps)")
        print("=" * 80)
        for i, issue in enumerate(self.issues['top_improvements'][:20], 1):
            print(f"{i:2d}. CLUE: {issue['clue']}")
            print(
                f"    ANSWER: {issue['answer']} ({issue['original_rank']} → {issue['evidence_rank']}, +{issue['improvement']} jump)")
            print(f"    EVIDENCE SCORE: {issue['evidence_score']:.1f}")
            print()


def main():
    """Run automated issue detection."""
    print("🤖 AUTOMATED ISSUE DETECTION SYSTEM")
    print("=" * 60)
    print("Systematically categorizing evidence analysis issues")
    print("For batch pattern analysis instead of manual review")
    print("=" * 60)

    detector = IssueDetector()
    issues = detector.analyze_batch(max_clues=500)

    print(f"\n✅ Issue detection complete!")
    print(
        f"Generated reports for {sum(len(category) for category in issues.values())} total issues")
    print(f"Ready for systematic pattern analysis")


if __name__ == "__main__":
    main()