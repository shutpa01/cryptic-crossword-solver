#!/usr/bin/env python3
"""
Comprehensive Anagram Evidence Detection System

This module provides sophisticated anagram detection that goes beyond simple letter counting:
- Tests all possible word combinations from clue text
- Handles deletion cases (DOLLOP → PLOD with deletion indicator)
- Handles insertion cases (fodder + extra letters → candidate)
- Integrates with existing candidate scoring system
- Provides detailed evidence reporting
- Uses progressive expansion from indicators to find anagram fodder
"""

import re
import itertools
from collections import Counter
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class AnagramEvidence:
    """Structured anagram evidence with detailed information."""
    candidate: str
    fodder_words: List[str]
    fodder_letters: str
    evidence_type: str  # "exact", "deletion", "insertion"
    confidence: float
    excess_letters: str = ""  # Letters not used (deletion case)
    needed_letters: str = ""  # Letters needed (insertion case)
    unused_clue_words: List[str] = None

    def __post_init__(self):
        if self.unused_clue_words is None:
            self.unused_clue_words = []


class ComprehensiveWordplayDetector:
    """Comprehensive wordplay evidence detection system using all database indicators."""

    def __init__(self, db_path: str = None):
        """Initialize with database path to load all indicators."""
        self.db_path = db_path or r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"

        # Load all indicators from database by type
        self.anagram_indicators = []
        self.insertion_indicators = []
        self.deletion_indicators = []
        self.reversal_indicators = []
        self.hidden_indicators = []
        self.parts_indicators = []

        # Store confidence levels
        self.indicator_confidence = {}
        self.indicators_loaded = False

        self._load_all_indicators_from_database()

    def _load_all_indicators_from_database(self):
        """Load all indicators from the database organized by wordplay type."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Load all indicators with their types and confidence
            cursor.execute("""
                SELECT word, wordplay_type, confidence FROM indicators 
                ORDER BY 
                    CASE confidence 
                        WHEN 'high' THEN 3
                        WHEN 'medium' THEN 2  
                        WHEN 'low' THEN 1
                        ELSE 0
                    END DESC, word
            """)

            all_indicators = cursor.fetchall()

            # Organize by wordplay type
            for word, wordplay_type, confidence in all_indicators:
                word_lower = word.lower().strip()
                self.indicator_confidence[word_lower] = confidence

                if wordplay_type == 'anagram':
                    self.anagram_indicators.append(word_lower)
                elif wordplay_type == 'insertion':
                    self.insertion_indicators.append(word_lower)
                elif wordplay_type == 'deletion':
                    self.deletion_indicators.append(word_lower)
                elif wordplay_type == 'reversal':
                    self.reversal_indicators.append(word_lower)
                elif wordplay_type == 'hidden':
                    self.hidden_indicators.append(word_lower)
                elif wordplay_type == 'parts':
                    self.parts_indicators.append(word_lower)

            conn.close()

            print(f"Loaded comprehensive indicators from database:")
            print(f"  Anagram indicators: {len(self.anagram_indicators)}")
            print(f"  Insertion indicators: {len(self.insertion_indicators)}")
            print(f"  Deletion indicators: {len(self.deletion_indicators)}")
            print(f"  Reversal indicators: {len(self.reversal_indicators)}")
            print(f"  Hidden word indicators: {len(self.hidden_indicators)}")
            print(f"  Parts indicators: {len(self.parts_indicators)}")
            print(f"  Total indicators loaded: {len(all_indicators)}")

            # Show sample of each type for verification
            if self.anagram_indicators:
                sample = self.anagram_indicators[:5]
                print(f"  Sample anagram indicators: {', '.join(sample)}")

            if self.insertion_indicators:
                sample = self.insertion_indicators[:5]
                print(f"  Sample insertion indicators: {', '.join(sample)}")

            self.indicators_loaded = True

        except Exception as e:
            print(f"ERROR loading indicators from database: {e}")
            print("Falling back to minimal hardcoded indicators...")

            # Minimal fallback list
            self.anagram_indicators = ['confused', 'mixed', 'jumbled', 'corrupted',
                                       'converts', 'exceptional', 'comic']
            self.insertion_indicators = ['in', 'inside', 'within', 'among', 'held by']
            self.deletion_indicators = ['lost', 'missing', 'without', 'dropped']
            self.reversal_indicators = ['back', 'returning', 'reversed']
            self.hidden_indicators = ['hidden', 'concealed', 'some']
            self.parts_indicators = ['initially', 'finally', 'head', 'tail']
            self.indicator_confidence = {}
            self.indicators_loaded = False

    def normalize_letters(self, text: str) -> str:
        """Extract and normalize letters only."""
        return ''.join(re.findall(r'[A-Za-z]', text.upper()))

    def is_anagram(self, letters1: str, letters2: str) -> bool:
        """Check if two letter sequences are anagrams."""
        return Counter(letters1) == Counter(letters2)

    def can_contribute_letters(self, target: str, source: str) -> Tuple[bool, float, str]:
        """
        Check if ALL source letters can contribute to target (complete word contribution only).
        Returns (can_contribute, contribution_ratio, remaining_letters)

        Rule: Complete words must contribute - no cherry-picking individual letters.
        """
        source_count = Counter(source)
        target_count = Counter(target)

        # Check if ALL source letters can be used in target
        used_letters = Counter()

        for letter, needed in source_count.items():
            available_in_target = target_count.get(letter, 0)
            if available_in_target < needed:
                # Can't use all of this source letter - invalid contribution
                return False, 0.0, target
            else:
                # Use all instances of this letter from source
                used_letters[letter] = needed

        # All source letters can be used - calculate remaining
        remaining_count = target_count - used_letters
        remaining_letters = ''.join(remaining_count.elements())

        # Source length for ratio calculation
        source_length = len(source)
        if source_length == 0:
            return False, 0.0, target

        # All source letters are used, so contribution ratio is 100%
        contribution_ratio = 1.0

        return True, contribution_ratio, remaining_letters

    def can_form_by_deletion_strict(self, target: str, source: str) -> Tuple[bool, str]:
        """
        Check if target can be formed by deleting ≤2 letters from source.
        Returns (success, excess_letters)
        """
        source_count = Counter(source)
        target_count = Counter(target)

        # Check if target is subset of source
        for letter, count in target_count.items():
            if source_count[letter] < count:
                return False, ""

        # Calculate excess letters
        excess_count = source_count - target_count
        excess_letters = ''.join(excess_count.elements())

        # Only allow ≤2 excess letters
        can_delete = len(excess_letters) <= 2

        return can_delete, excess_letters

    def can_form_by_deletion(self, target: str, source: str) -> Tuple[bool, str]:
        """
        Check if target can be formed by deleting letters from source.
        Returns (success, excess_letters)
        """
        source_count = Counter(source)
        target_count = Counter(target)

        # Check if target is subset of source
        for letter, count in target_count.items():
            if source_count[letter] < count:
                return False, ""

        # Calculate excess letters
        excess_count = source_count - target_count
        excess_letters = ''.join(excess_count.elements())
        return True, excess_letters

    def can_form_by_insertion(self, target: str, source: str) -> Tuple[bool, str]:
        """
        Check if target can be formed by adding letters to source.
        Returns (success, needed_letters)
        """
        source_count = Counter(source)
        target_count = Counter(target)

        # Check if source is subset of target
        for letter, count in source_count.items():
            if target_count[letter] < count:
                return False, ""

        # Calculate needed letters
        needed_count = target_count - source_count
        needed_letters = ''.join(needed_count.elements())
        return True, needed_letters

    def get_all_word_combinations(self, words: List[str]) -> List[List[str]]:
        """
        Generate all possible combinations of words (1 to all words).
        Returns list of word combinations, sorted by likelihood.
        """
        combinations = []

        # Generate all possible combinations
        for r in range(1, len(words) + 1):
            for combo in itertools.combinations(words, r):
                combinations.append(list(combo))

        # Sort by combination size (smaller first, more likely to be anagram fodder)
        combinations.sort(key=len)

        return combinations

    def detect_wordplay_indicators(self, clue_text: str) -> Dict[str, List[str]]:
        """
        Detect anagram indicators in clue text.
        Returns dict with anagram indicators found.
        """
        clue_words = [word.lower().strip('.,!?:;()') for word in clue_text.split()]

        found = {
            'anagram': []
        }

        for word in clue_words:
            # Only check anagram indicators
            if word in self.anagram_indicators:
                found['anagram'].append(word)

        return found

    def get_progressive_fodder_words(self, clue_text: str, indicators: dict,
                                     candidates: List[str]) -> list:
        """
        Progressive expansion from indicator positions to find anagram fodder.

        Algorithm:
        1. Find indicator positions
        2. Move progressively left and right from each indicator
        3. Test contribution FIRST, then treat as link word only if doesn't contribute
        4. Continue expanding through structural link words

        Args:
            clue_text: The clue text
            indicators: Dictionary of detected indicators
            candidates: List of candidates to test fodder against

        Returns:
            List of words that can contribute as anagram fodder
        """
        anagram_indicators = indicators.get('anagram', [])
        if not anagram_indicators:
            return []

        # Split clue into words and find indicator positions
        clue_words = [word.strip('.,!?:;()') for word in clue_text.split()]
        indicator_positions = []

        for i, word in enumerate(clue_words):
            if word.lower() in anagram_indicators:
                indicator_positions.append(i)

        if not indicator_positions:
            return []

        # Common link words that connect indicators to fodder
        link_words = {'in', 'with', 'of', 'needs','to', 'for', 'by', 'from', 'about',
                      'on',
                      'after', 'and'}

        # Collect fodder words from all indicators
        all_fodder_positions = set()

        for ind_pos in indicator_positions:
            # Progressive expansion left from indicator
            pos = ind_pos - 1
            while pos >= 0:
                word = clue_words[pos]
                word_lower = word.lower()

                # Test if this word can contribute to any candidate FIRST
                if self._can_word_contribute_to_candidates(word, candidates):
                    all_fodder_positions.add(pos)
                    pos -= 1  # Continue expanding left
                # If doesn't contribute, check if it's a structural link word
                elif word_lower in link_words:
                    pos -= 1  # Skip but continue expanding
                    continue
                else:
                    break  # Stop expanding left - word doesn't contribute and isn't structural

            # Progressive expansion right from indicator
            pos = ind_pos + 1
            while pos < len(clue_words):
                word = clue_words[pos]
                word_lower = word.lower()

                # Test if this word can contribute to any candidate FIRST
                if self._can_word_contribute_to_candidates(word, candidates):
                    all_fodder_positions.add(pos)
                    pos += 1  # Continue expanding right
                # If doesn't contribute, check if it's a structural link word
                elif word_lower in link_words:
                    pos += 1  # Skip but continue expanding
                    continue
                else:
                    break  # Stop expanding right - word doesn't contribute and isn't structural

        # Return words at fodder positions (excluding indicators)
        fodder_words = []
        for pos in sorted(all_fodder_positions):
            word = clue_words[pos]
            if word.lower() not in anagram_indicators and len(word) > 1:
                fodder_words.append(word)

        return fodder_words

    def _can_word_contribute_to_candidates(self, word: str,
                                           candidates: List[str]) -> bool:
        """
        Check if a word can contribute letters to any of the candidates.

        Args:
            word: The word to test
            candidates: List of candidate answers

        Returns:
            True if word can contribute to at least one candidate
        """
        word_normalized = self.normalize_letters(word)
        if not word_normalized:
            return False

        for candidate in candidates:
            candidate_normalized = self.normalize_letters(candidate)
            can_contribute, _, _ = self.can_contribute_letters(word_normalized,
                                                               candidate_normalized)
            if can_contribute:
                return True

        return False

    def test_anagram_evidence(self, candidate: str, clue_text: str,
                              indicators: Dict[str, List[str]], enumeration: str = None,
                              debug: bool = False) -> Optional[AnagramEvidence]:
        """
        Test for anagram evidence using progressive expansion from indicators.
        Returns AnagramEvidence if match found, None otherwise.
        """
        candidate_letters = self.normalize_letters(candidate)
        if not candidate_letters:
            return None

        # STAGE B HYGIENE: reject self-anagrams (matching your proven anagram engine)
        if candidate.lower() in clue_text.lower():
            if debug:
                print(
                    f"    REJECTED: Self-anagram - '{candidate}' appears verbatim in clue")
            return None

        # Use progressive expansion to find fodder words near indicators
        candidates_list = [candidate]  # Pass current candidate for fodder testing
        progressive_words = self.get_progressive_fodder_words(clue_text, indicators,
                                                              candidates_list)

        if not progressive_words:
            if debug:
                print(f"    NO PROGRESSIVE FODDER WORDS found near indicators")
            return None

        # Apply stop word filtering to progressive words
        stop_words = {'the', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'by', 'from'}
        content_words = [w for w in progressive_words if
                         w.lower() not in stop_words and len(w) > 1]

        if not content_words:
            if debug:
                print(f"    NO CONTENT WORDS after filtering progressive words")
            return None

        if debug:
            print(
                f"    DEBUG: Testing candidate '{candidate}' ({len(candidate_letters)} letters)")
            print(f"    Progressive fodder words: {progressive_words}")
            print(f"    Content words (after filtering): {content_words}")

        # Test all possible word combinations from progressive words
        word_combinations = self.get_all_word_combinations(content_words)

        best_evidence = None
        best_score = 0.0

        if debug:
            print(f"    Testing {len(word_combinations)} word combinations...")

        for i, word_combo in enumerate(word_combinations):
            fodder_text = ' '.join(word_combo)
            fodder_letters = self.normalize_letters(fodder_text)

            if not fodder_letters:
                continue

            if debug and i < 10:  # Show first 10 combinations
                print(
                    f"      [{i + 1:2d}] Testing: {word_combo} → '{fodder_letters}' ({len(fodder_letters)} letters)")

            # Test exact anagram match first (highest priority)
            if self.is_anagram(candidate_letters, fodder_letters):
                if debug:
                    print(f"      ★ EXACT ANAGRAM MATCH!")
                unused_words = [w for w in content_words if w not in word_combo]
                return AnagramEvidence(
                    candidate=candidate,
                    fodder_words=word_combo,
                    fodder_letters=fodder_letters,
                    evidence_type="exact",
                    confidence=0.9,
                    unused_clue_words=unused_words
                )

            # Test partial contribution (only if anagram indicators present)
            if indicators['anagram']:
                can_contribute, contribution_ratio, remaining_letters = self.can_contribute_letters(
                    candidate_letters, fodder_letters)

                if debug and i < 10:
                    print(
                        f"           Partial: can_contribute={can_contribute}, remaining='{remaining_letters}'")

                if can_contribute:
                    # Calculate explained letters (secondary scoring factor)
                    explained_letters = len(candidate_letters) - len(remaining_letters)
                    total_letters = len(candidate_letters)

                    # Safety check to avoid division by zero
                    if total_letters == 0:
                        continue

                    # PRIMARY: Check enumeration pattern match (highest priority)
                    enumeration_bonus = 0
                    if enumeration:
                        if self._matches_enumeration_pattern(candidate, enumeration):
                            enumeration_bonus = 100  # Massive bonus for correct pattern
                        else:
                            enumeration_bonus = -50  # Penalty for wrong pattern

                    # SECONDARY: Explained letters (0-8 for 8-letter words)
                    primary_score = explained_letters

                    # TERTIARY: Word count penalty (fewer words = higher score)
                    word_count_factor = 1.0 / len(word_combo)

                    # Combined score: enumeration dominates, then letters, then coherence
                    evidence_score = enumeration_bonus + primary_score + word_count_factor

                    # Calculate confidence for display (0.0 to 1.0)
                    confidence = explained_letters / total_letters

                    if debug and i < 10:
                        enum_status = "✅" if enumeration_bonus > 0 else "❌" if enumeration_bonus < 0 else "?"
                        print(
                            f"           → Score: {evidence_score:.2f} ({enum_status} enum={enumeration_bonus}, letters={explained_letters}/{total_letters}, words={len(word_combo)}, confidence={confidence:.2f})")

                    if evidence_score > best_score:
                        unused_words = [w for w in content_words if w not in word_combo]
                        best_evidence = AnagramEvidence(
                            candidate=candidate,
                            fodder_words=word_combo,
                            fodder_letters=fodder_letters,
                            evidence_type="partial",
                            confidence=confidence,
                            needed_letters=remaining_letters,  # Letters still needed
                            unused_clue_words=unused_words
                        )
                        best_score = evidence_score

                        if debug:
                            print(
                                f"           ★ NEW BEST PARTIAL EVIDENCE! Score: {best_score:.2f}")

            # Also test for deletion anagrams (≤2 excess letters)
            if indicators['anagram']:
                can_delete, excess = self.can_form_by_deletion_strict(candidate_letters,
                                                                      fodder_letters)
                if can_delete:
                    if debug and i < 10:
                        print(
                            f"           Deletion: can_delete={can_delete}, excess='{excess}'")

                    unused_words = [w for w in content_words if w not in word_combo]
                    deletion_confidence = 0.8  # High confidence for clean deletion

                    if debug:
                        print(
                            f"           ★ DELETION MATCH! Confidence: {deletion_confidence}")

                    return AnagramEvidence(
                        candidate=candidate,
                        fodder_words=word_combo,
                        fodder_letters=fodder_letters,
                        evidence_type="deletion",
                        confidence=deletion_confidence,
                        excess_letters=excess,
                        unused_clue_words=unused_words
                    )

        if debug and best_evidence:
            print(
                f"    FINAL BEST EVIDENCE: {best_evidence.evidence_type}, score: {best_score:.2f}")
        elif debug:
            print(f"    NO EVIDENCE FOUND for {candidate}")

        # Return best partial evidence found, if any
        return best_evidence

    def _matches_enumeration_pattern(self, candidate: str, enumeration: str) -> bool:
        """
        Check if candidate matches the enumeration pattern.
        Examples:
        - "SPARE RIB" matches "(5,3)"
        - "separate" does NOT match "(5,3)"
        """
        if not enumeration:
            return True  # No enumeration constraint

        # Extract numbers from enumeration like "(5,3)" or "8"
        import re
        numbers = re.findall(r'\d+', enumeration)
        if not numbers:
            return True  # No clear pattern to match

        expected_lengths = [int(n) for n in numbers]

        # Split candidate into words and get their lengths
        candidate_words = candidate.replace('-', ' ').split()
        candidate_lengths = [len(self.normalize_letters(word)) for word in
                             candidate_words]

        # Check if lengths match exactly
        return candidate_lengths == expected_lengths

    def analyze_clue_for_anagram_evidence(self, clue_text: str, candidates: List[str],
                                          enumeration: str = None, debug: bool = False) -> \
            List[AnagramEvidence]:
        """
        Anagram-only analysis for a clue and candidate list.
        Returns list of AnagramEvidence objects for candidates with anagram evidence.
        """
        # Detect only anagram indicators
        indicators = self.detect_wordplay_indicators(clue_text)

        if debug:
            print(f"  DETECTED INDICATORS: {indicators}")

        # If no anagram indicators, skip analysis
        if not indicators['anagram']:
            if debug:
                print(f"  NO ANAGRAM INDICATORS FOUND - skipping analysis")
            return []

        evidence_list = []

        # Test each candidate for anagram evidence
        for candidate in candidates:
            if debug:
                print(f"\n  🔍 TESTING CANDIDATE: {candidate}")
            evidence = self.test_anagram_evidence(candidate, clue_text, indicators,
                                                  enumeration, debug=debug)
            if evidence:
                evidence_list.append(evidence)

        # Sort by confidence (best evidence first)
        evidence_list.sort(key=lambda e: e.confidence, reverse=True)

        return evidence_list

    def is_fodder_contiguous(self, candidate: str, fodder_letters: str) -> bool:
        """
        Check if fodder letters appear contiguously in candidate as an anagram.

        Args:
            candidate: The candidate word (e.g., "UNALTERED")
            fodder_letters: The fodder letters (e.g., "TREE")

        Returns:
            True if any contiguous substring of candidate is an anagram of fodder
        """
        candidate_normalized = self.normalize_letters(candidate).upper()
        fodder_normalized = fodder_letters.upper()
        fodder_length = len(fodder_normalized)

        if fodder_length == 0:
            return True

        # Check all contiguous substrings of length equal to fodder
        for i in range(len(candidate_normalized) - fodder_length + 1):
            substring = candidate_normalized[i:i + fodder_length]
            if self.is_anagram(substring, fodder_normalized):
                return True

        return False

    def calculate_anagram_score_boost(self, evidence: AnagramEvidence) -> float:
        """
        Calculate score boost for candidate based on anagram evidence quality.
        Now supports partial evidence for multi-stage solving.
        Returns additive score boost.
        """
        base_boost = {
            'exact': 20.0,  # Complete anagram match
            'partial': 8.0,  # Partial contribution (new)
            'deletion': 15.0,  # Deletion anagram
            'insertion': 12.0  # Insertion anagram
        }

        boost = base_boost.get(evidence.evidence_type, 0.0)

        # Adjust based on confidence
        boost *= evidence.confidence

        # Check if fodder appears contiguously in candidate
        is_contiguous = self.is_fodder_contiguous(evidence.candidate,
                                                  evidence.fodder_letters)

        # Apply 50% reduction for scattered (non-contiguous) fodder
        if not is_contiguous:
            boost *= 0.5

        # Bonus for using more clue words
        word_count_bonus = len(evidence.fodder_words) * 1.5

        # Special handling for partial evidence
        if evidence.evidence_type == "partial":
            # Additional bonus based on how much of candidate is explained
            if evidence.needed_letters:
                explained_ratio = 1.0 - (
                        len(evidence.needed_letters) / len(evidence.candidate))
                boost += explained_ratio * 5.0  # Up to 5 extra points

        return boost + word_count_bonus


def demo_comprehensive_wordplay_detection():
    """Demonstrate comprehensive wordplay evidence detection including all indicator types."""
    # Use the database path - adjust this to your actual database location
    db_path = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
    detector = ComprehensiveWordplayDetector(db_path=db_path)

    test_cases = [
        # (clue, [candidates...])
        ("Confused traces in garden (6)", ["CASTER", "TRACER", "CRATER"]),
        ("Mixed dollop lost letters makes runner (4)", ["PLOD", "LOOP", "POLO"]),
        ("Established scout corrupted virgin (9)",
         ["CUSTOMARY", "CONFIRMED", "ANCESTRAL"]),
        ("Same sad converts got together (7)", ["AMASSED", "EN MASSE", "UNIFORM"]),
        ("Comic teases the arty types (9)", ["AESTHETES", "BOHEMIANS", "BURLESQUE"]),
        ("Ornament of exceptional age incorporating representation of glory (8)",
         ["GARGOYLE", "BRACELET", "FIGURINE"]),
        ("Shopkeeper's beard has disturbed that woman (11)",
         ["HABERDASHER", "STOREKEEPER", "SHOPKEEPER"]),
    ]

    for clue, candidates in test_cases:
        print(f"\n{'=' * 80}")
        print(f"CLUE: {clue}")
        print(f"CANDIDATES: {', '.join(candidates)}")
        print(f"{'=' * 80}")

        # Show comprehensive indicator detection
        indicators = detector.detect_wordplay_indicators(clue)
        print(f"DETECTED INDICATORS:")
        for wordplay_type, found_indicators in indicators.items():
            if found_indicators:
                print(f"  {wordplay_type.upper()}: {', '.join(found_indicators)}")

        # Show anagram evidence (keeping focus on current implementation)
        evidence_list = detector.analyze_clue_for_anagram_evidence(clue, candidates)

        if evidence_list:
            for evidence in evidence_list:
                print(f"\n★ ANAGRAM EVIDENCE FOUND:")
                print(f"  Candidate: {evidence.candidate}")
                print(f"  Type: {evidence.evidence_type}")
                print(f"  Fodder: {' + '.join(evidence.fodder_words)}")
                print(f"  Confidence: {evidence.confidence:.2f}")

                if evidence.evidence_type == "partial":
                    print(f"  Remaining letters needed: {evidence.needed_letters}")
                elif evidence.excess_letters:
                    print(f"  Excess letters: {evidence.excess_letters}")
                elif evidence.needed_letters:
                    print(f"  Needed letters: {evidence.needed_letters}")

                if evidence.unused_clue_words:
                    print(f"  Unused clue words: {', '.join(evidence.unused_clue_words)}")

                boost = detector.calculate_anagram_score_boost(evidence)
                print(f"  Score boost: +{boost:.1f}")
        else:
            print("No anagram evidence found.")


if __name__ == "__main__":
    demo_comprehensive_wordplay_detection()