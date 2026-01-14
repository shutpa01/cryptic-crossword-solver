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

CORRECTED VERSION: Enforces four rules for fodder:
1. Indicator detection - single word first, then two-word if needed, with positions
2. Proximity - fodder must be adjacent to indicator (one link word allowed)
3. Contiguity - fodder words must be next to each other in the clue
4. Whole words - fodder is complete words, not cherry-picked letters
"""

import re
import itertools
from collections import Counter
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

# Import the working anagram function
import sys
import os

sys.path.append(r'C:\Users\shute\PycharmProjects\cryptic_solver')
from solver.wordplay.anagram.anagram_stage import generate_anagram_hypotheses


@dataclass
class AnagramEvidence:
    """Structured anagram evidence with detailed information."""
    candidate: str
    fodder_words: List[str]
    fodder_letters: str
    evidence_type: str  # "exact", "deletion", "insertion", "partial"
    confidence: float
    excess_letters: str = ""  # Letters not used (deletion case)
    needed_letters: str = ""  # Letters needed (insertion case)

    # NEW: Complete word attribution for compound analysis
    indicator_words: List[str] = None  # The anagram indicator(s)
    indicator_position: int = -1  # Position of indicator in token list
    definition_words: List[str] = None  # Definition window words (if known)
    link_words: List[str] = None  # Link words identified
    remaining_words: List[str] = None  # Words available for compound wordplay

    # Deprecated: keeping for backward compatibility
    unused_clue_words: List[str] = None

    def __post_init__(self):
        if self.unused_clue_words is None:
            self.unused_clue_words = []
        if self.indicator_words is None:
            self.indicator_words = []
        if self.definition_words is None:
            self.definition_words = []
        if self.link_words is None:
            self.link_words = []
        if self.remaining_words is None:
            self.remaining_words = []


@dataclass
class IndicatorMatch:
    """Stores indicator match with position info."""
    words: List[str]
    start_pos: int
    end_pos: int  # inclusive
    is_multi_word: bool


@dataclass
class ContiguousFodder:
    """A valid contiguous fodder sequence adjacent to an indicator."""
    words: List[str]
    positions: List[int]
    letters: str
    indicator: IndicatorMatch
    side: str  # 'left' or 'right' of indicator


# Link words that can appear between indicator and fodder
LINK_WORDS = {
    # Articles and prepositions
    'to', 'of', 'in', 'for', 'with', 'by', 'from', 'a', 'an', 'the',
    'and', 'is', 'are', 'needs', 'about', 'on', 'after', 'at', 'as', 'or',
    # Common verbs
    'be', 'being', 'been', 'has', 'have', 'had', 'having',
    'was', 'were', 'will', 'would', 'could', 'should', 'must', 'may', 'might',
    'gets', 'get', 'getting', 'got', 'makes', 'make', 'making', 'made',
    'gives', 'give', 'given', 'giving', 'sees', 'see', 'seen', 'seeing',
    # Contractions (with apostrophe)
    "it's", "that's", "there's", "here's", "what's",
    # Contractions (apostrophe-stripped)
    'its', 'thats', 'theres', 'heres', 'whats', 'im', 'ive', 'id',
    'youre', 'youve', 'youd', 'hes', 'shes', 'theyre', 'theyve',
    'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'cant', 'couldnt',
    # Conjunctions and connectors
    'but', 'that', 'which', 'when', 'where', 'while', 'so', 'yet',
    # Other common links
    'this', 'these', 'those', 'such', 'one', 'ones', 'some', 'any', 'all',
    'here', 'there', 'into', 'onto', 'within', 'without',
    'find', 'found', 'finding', 'show', 'showing', 'put', 'set',
    'if', 'how', 'why', 'who', 'whom', 'you',
}

# Link words that can appear BETWEEN fodder words without breaking contiguity
# These are words that setters commonly use to join fodder: "birds WITH ale", "cats AND dogs"
TRANSPARENT_LINK_WORDS = {'with', 'and', 'or'}


class ComprehensiveWordplayDetector:
    """Comprehensive wordplay evidence detection system using all database indicators."""

    def __init__(self, db_path: str = None):
        """Initialize with database path to load all indicators."""
        self.db_path = db_path or r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"

        # Load all indicators from database by type
        self.anagram_indicators = []  # Keep for backward compatibility
        self.anagram_indicators_single: Set[str] = set()  # NEW: single-word only
        self.anagram_indicators_two_word: Set[str] = set()  # NEW: two-word only
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
                    # NEW: Split into single and two-word
                    if ' ' in word_lower:
                        self.anagram_indicators_two_word.add(word_lower)
                    else:
                        self.anagram_indicators_single.add(word_lower)
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
            print(
                f"  Anagram indicators: {len(self.anagram_indicators)} ({len(self.anagram_indicators_single)} single-word, {len(self.anagram_indicators_two_word)} two-word)")
            print(f"  Insertion indicators: {len(self.insertion_indicators)}")
            print(f"  Deletion indicators: {len(self.deletion_indicators)}")
            print(f"  Reversal indicators: {len(self.reversal_indicators)}")
            print(f"  Hidden word indicators: {len(self.hidden_indicators)}")
            print(f"  Parts indicators: {len(self.parts_indicators)}")
            print(f"  Total indicators loaded: {len(all_indicators)}")

            # Show sample of each type for verification
            if self.anagram_indicators_single:
                sample = list(self.anagram_indicators_single)[:5]
                print(f"  Sample single-word anagram indicators: {', '.join(sample)}")

            if self.anagram_indicators_two_word:
                sample = list(self.anagram_indicators_two_word)[:5]
                print(f"  Sample two-word anagram indicators: {', '.join(sample)}")

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
            self.anagram_indicators_single = {'confused', 'mixed', 'jumbled', 'corrupted',
                                              'converts', 'exceptional', 'comic'}
            self.anagram_indicators_two_word = {'move round', 'going round', 'mixed up'}
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

        NOTE: This is kept for backward compatibility but should not be used
        for fodder finding. Use get_contiguous_fodder_sequences instead.
        """
        combinations = []

        # Generate all possible combinations
        for r in range(1, len(words) + 1):
            for combo in itertools.combinations(words, r):
                combinations.append(list(combo))

        # Sort by combination size (smaller first, more likely to be anagram fodder)
        combinations.sort(key=len)

        return combinations

    def _tokenize_clue(self, clue_text: str) -> List[str]:
        """
        Split clue into word tokens.
        Filters out punctuation and enumeration markers.
        Returns list of clean word tokens.
        """
        raw_tokens = clue_text.split()
        tokens = []
        for token in raw_tokens:
            # Strip punctuation from edges
            cleaned = token.strip('.,!?:;()—-"\'')
            # Skip empty, pure punctuation, or enumeration like (8)
            if cleaned and cleaned.isalpha():
                tokens.append(cleaned)
        return tokens

    def detect_wordplay_indicators(self, clue_text: str) -> Dict[str, List[str]]:
        """
        Detect anagram indicators in clue text.
        Returns dict with anagram indicators found.

        CORRECTED: Now also populates indicator_positions for use by
        get_contiguous_fodder_sequences. Check single-word first,
        then two-word only if no single-word found.
        """
        tokens = self._tokenize_clue(clue_text)

        found = {
            'anagram': [],
            'anagram_matches': []  # NEW: List of IndicatorMatch objects
        }

        # First pass: single-word indicators
        for i, token in enumerate(tokens):
            if token.lower() in self.anagram_indicators_single:
                found['anagram'].append(token.lower())
                found['anagram_matches'].append(IndicatorMatch(
                    words=[token],
                    start_pos=i,
                    end_pos=i,
                    is_multi_word=False
                ))

        # Second pass: two-word indicators (only if no single-word found)
        if not found['anagram_matches']:
            for i in range(len(tokens) - 1):
                two_word = f"{tokens[i].lower()} {tokens[i + 1].lower()}"
                if two_word in self.anagram_indicators_two_word:
                    found['anagram'].append(two_word)
                    found['anagram_matches'].append(IndicatorMatch(
                        words=[tokens[i], tokens[i + 1]],
                        start_pos=i,
                        end_pos=i + 1,
                        is_multi_word=True
                    ))

        return found

    def get_progressive_fodder_words(self, clue_text: str, indicators: dict,
                                     candidates: List[str]) -> list:
        """
        Progressive expansion from indicator positions to find anagram fodder.

        CORRECTED: Actually implements contiguous expansion from indicators.
        This method now returns words that are:
        1. Adjacent to an indicator (with at most one link word between)
        2. Contiguous (next to each other)
        3. Whole words

        Returns list of fodder word lists (each is a contiguous sequence).
        """
        anagram_matches = indicators.get('anagram_matches', [])
        if not anagram_matches:
            return []

        tokens = self._tokenize_clue(clue_text)
        all_fodder_sequences = []

        for indicator in anagram_matches:
            # Expand left
            left_sequences = self._expand_from_indicator(tokens, indicator, 'left')
            all_fodder_sequences.extend(left_sequences)

            # Expand right
            right_sequences = self._expand_from_indicator(tokens, indicator, 'right')
            all_fodder_sequences.extend(right_sequences)

        # Return flat list of words from all sequences for backward compatibility
        # The caller can use get_contiguous_fodder_sequences for structured data
        all_words = []
        for seq in all_fodder_sequences:
            all_words.extend(seq.words)

        return list(set(all_words))  # Deduplicate

    def _expand_from_indicator(self, tokens: List[str], indicator: IndicatorMatch,
                               direction: str) -> List[ContiguousFodder]:
        """
        Expand in one direction from indicator to find valid contiguous fodder.
        Allows skipping one link word at the boundary.

        Returns list of ContiguousFodder objects (all valid contiguous sequences).
        """
        results = []
        n = len(tokens)

        if direction == 'left':
            boundary = indicator.start_pos - 1
            step = -1
        else:
            boundary = indicator.end_pos + 1
            step = 1

        if boundary < 0 or boundary >= n:
            return results

        # Check if boundary word is a link word - if so, skip it
        start_pos = boundary
        if tokens[boundary].lower() in LINK_WORDS:
            start_pos = boundary + step
            if start_pos < 0 or start_pos >= n:
                return results

        # Expand contiguously from start_pos
        current_words = []
        current_positions = []
        pos = start_pos
        skipped_transparent_link = False  # Only allow skipping ONE transparent link word

        while 0 <= pos < n:
            token = tokens[pos]
            token_lower = token.lower()

            # Stop if we hit an indicator word
            if token_lower in self.anagram_indicators_single:
                break

            # Handle link words after fodder has started
            if token_lower in LINK_WORDS and current_words:
                # For transparent link words (with, and, or), peek ahead to see if more fodder exists
                # But only skip ONE transparent link word per expansion
                if token_lower in TRANSPARENT_LINK_WORDS and not skipped_transparent_link:
                    peek_pos = pos + step
                    # Check if there's a non-link word ahead (potential fodder)
                    if 0 <= peek_pos < n:
                        peek_token = tokens[peek_pos].lower()
                        # If next word is NOT a link word and NOT an indicator, skip this link word
                        if (peek_token not in LINK_WORDS and
                            peek_token not in self.anagram_indicators_single):
                            # Skip the transparent link word, continue to next
                            skipped_transparent_link = True  # Mark that we've used our one skip
                            pos += step
                            continue
                # Stop for non-transparent link words, if already skipped one, or if no fodder ahead
                break

            # Skip link words at the start (these don't count against our one-skip limit)
            if token_lower in LINK_WORDS and not current_words:
                pos += step
                continue

            # Add this word to fodder
            current_words.append(token)
            current_positions.append(pos)

            # Get ordered words (left expansion needs reversal)
            if direction == 'left':
                ordered_words = list(reversed(current_words))
                ordered_positions = list(reversed(current_positions))
            else:
                ordered_words = current_words[:]
                ordered_positions = current_positions[:]

            letters = self.normalize_letters(' '.join(ordered_words))

            # Record this as a valid contiguous sequence
            results.append(ContiguousFodder(
                words=ordered_words,
                positions=ordered_positions,
                letters=letters,
                indicator=indicator,
                side=direction
            ))

            pos += step

        return results

    def get_contiguous_fodder_sequences(self, clue_text: str, indicators: dict,
                                        target_length: int = None) -> List[
        ContiguousFodder]:
        """
        Get all valid contiguous fodder sequences for a clue.

        This is the main method for finding fodder that enforces all four rules:
        1. Indicator position tracking
        2. Proximity to indicator
        3. Contiguity of fodder words
        4. Whole words only

        Args:
            clue_text: The clue text
            indicators: Dict from detect_wordplay_indicators
            target_length: Optional filter for exact letter count match

        Returns:
            List of ContiguousFodder objects
        """
        anagram_matches = indicators.get('anagram_matches', [])
        if not anagram_matches:
            return []

        tokens = self._tokenize_clue(clue_text)
        all_sequences = []

        for indicator in anagram_matches:
            # Expand left
            left = self._expand_from_indicator(tokens, indicator, 'left')
            all_sequences.extend(left)

            # Expand right
            right = self._expand_from_indicator(tokens, indicator, 'right')
            all_sequences.extend(right)

        # Filter by target length if specified
        if target_length is not None:
            all_sequences = [s for s in all_sequences if len(s.letters) == target_length]

        return all_sequences

    def _can_word_contribute_to_candidates(self, word: str,
                                           candidates: List[str]) -> bool:
        """
        FIXED: Check if a word can contribute letters to any of the candidates.
        CRITICAL FIX: Parameter order corrected.
        """
        word_normalized = self.normalize_letters(word)
        if not word_normalized:
            return False

        for candidate in candidates:
            candidate_normalized = self.normalize_letters(candidate)
            # FIXED: Correct parameter order - can source (word) contribute to target (candidate)
            can_contribute, _, _ = self.can_contribute_letters(candidate_normalized,
                                                               word_normalized)
            if can_contribute:
                return True

        return False

    def test_anagram_evidence(self, candidate: str, clue_text: str,
                              indicators: Dict[str, List[str]], enumeration: str = None,
                              debug: bool = False) -> Optional[AnagramEvidence]:
        """
        Test for anagram evidence using contiguous fodder from indicators.

        CORRECTED: Now only tests contiguous fodder sequences adjacent to indicators,
        not all possible word combinations.

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

        # Get contiguous fodder sequences (enforces all four rules)
        target_length = len(candidate_letters)
        fodder_sequences = self.get_contiguous_fodder_sequences(
            clue_text, indicators, target_length=None  # Get all, filter later
        )

        if not fodder_sequences:
            if debug:
                print(f"    NO CONTIGUOUS FODDER SEQUENCES found near indicators")
            return None

        if debug:
            print(
                f"    DEBUG: Testing candidate '{candidate}' ({len(candidate_letters)} letters)")
            print(f"    Found {len(fodder_sequences)} contiguous fodder sequences")

        # Get all tokens for unused word calculation
        tokens = self._tokenize_clue(clue_text)

        best_evidence = None
        best_score = 0.0

        for i, fodder in enumerate(fodder_sequences):
            fodder_letters = fodder.letters

            if debug and i < 10:
                print(
                    f"      [{i + 1:2d}] Testing: {fodder.words} → '{fodder_letters}' ({len(fodder_letters)} letters)")

            # Test exact anagram match first (highest priority)
            if self.is_anagram(candidate_letters, fodder_letters):
                if debug:
                    print(f"      ★ EXACT ANAGRAM MATCH!")

                # Calculate complete word attribution
                indicator_words = list(fodder.indicator.words)
                fodder_word_set = set(w.lower() for w in fodder.words)
                indicator_word_set = set(w.lower() for w in indicator_words)

                # Identify link words and remaining words
                link_words_found = []
                remaining_words = []
                for t in tokens:
                    t_lower = t.lower()
                    if t_lower in fodder_word_set:
                        continue  # Already accounted as fodder
                    if t_lower in indicator_word_set:
                        continue  # Already accounted as indicator
                    if t_lower in LINK_WORDS:
                        link_words_found.append(t)
                    else:
                        remaining_words.append(t)

                return AnagramEvidence(
                    candidate=candidate,
                    fodder_words=list(fodder.words),
                    fodder_letters=fodder_letters,
                    evidence_type="exact",
                    confidence=0.9,
                    indicator_words=indicator_words,
                    indicator_position=fodder.indicator.start_pos,
                    link_words=link_words_found,
                    remaining_words=remaining_words,
                    unused_clue_words=remaining_words  # Backward compatibility
                )

            # Test partial contribution (only if anagram indicators present)
            if indicators.get('anagram'):
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
                    word_count_factor = 1.0 / len(fodder.words)

                    # Combined score: enumeration dominates, then letters, then coherence
                    evidence_score = enumeration_bonus + primary_score + word_count_factor

                    # Calculate confidence for display (0.0 to 1.0)
                    confidence = explained_letters / total_letters

                    if debug and i < 10:
                        enum_status = "✅" if enumeration_bonus > 0 else "❌" if enumeration_bonus < 0 else "?"
                        print(
                            f"           → Score: {evidence_score:.2f} ({enum_status} enum={enumeration_bonus}, letters={explained_letters}/{total_letters}, words={len(fodder.words)}, confidence={confidence:.2f})")

                    if evidence_score > best_score:
                        # Calculate complete word attribution
                        indicator_words = list(fodder.indicator.words)
                        fodder_word_set = set(w.lower() for w in fodder.words)
                        indicator_word_set = set(w.lower() for w in indicator_words)

                        link_words_found = []
                        remaining_words = []
                        for t in tokens:
                            t_lower = t.lower()
                            if t_lower in fodder_word_set:
                                continue
                            if t_lower in indicator_word_set:
                                continue
                            if t_lower in LINK_WORDS:
                                link_words_found.append(t)
                            else:
                                remaining_words.append(t)

                        best_evidence = AnagramEvidence(
                            candidate=candidate,
                            fodder_words=list(fodder.words),
                            fodder_letters=fodder_letters,
                            evidence_type="partial",
                            confidence=confidence,
                            needed_letters=remaining_letters,
                            indicator_words=indicator_words,
                            indicator_position=fodder.indicator.start_pos,
                            link_words=link_words_found,
                            remaining_words=remaining_words,
                            unused_clue_words=remaining_words  # Backward compatibility
                        )
                        best_score = evidence_score

                        if debug:
                            print(
                                f"           ★ NEW BEST PARTIAL EVIDENCE! Score: {best_score:.2f}")

            # Also test for deletion anagrams (≤2 excess letters)
            if indicators.get('anagram'):
                can_delete, excess = self.can_form_by_deletion_strict(candidate_letters,
                                                                      fodder_letters)
                if can_delete:
                    if debug and i < 10:
                        print(
                            f"           Deletion: can_delete={can_delete}, excess='{excess}'")

                    # Calculate complete word attribution
                    indicator_words = list(fodder.indicator.words)
                    fodder_word_set = set(w.lower() for w in fodder.words)
                    indicator_word_set = set(w.lower() for w in indicator_words)

                    link_words_found = []
                    remaining_words = []
                    for t in tokens:
                        t_lower = t.lower()
                        if t_lower in fodder_word_set:
                            continue
                        if t_lower in indicator_word_set:
                            continue
                        if t_lower in LINK_WORDS:
                            link_words_found.append(t)
                        else:
                            remaining_words.append(t)

                    deletion_confidence = 0.8

                    if debug:
                        print(
                            f"           ★ DELETION MATCH! Confidence: {deletion_confidence}")

                    return AnagramEvidence(
                        candidate=candidate,
                        fodder_words=list(fodder.words),
                        fodder_letters=fodder_letters,
                        evidence_type="deletion",
                        confidence=deletion_confidence,
                        excess_letters=excess,
                        indicator_words=indicator_words,
                        indicator_position=fodder.indicator.start_pos,
                        link_words=link_words_found,
                        remaining_words=remaining_words,
                        unused_clue_words=remaining_words  # Backward compatibility
                    )

        if debug and best_evidence:
            print(
                f"    FINAL BEST EVIDENCE: {best_evidence.evidence_type}, score: {best_score:.2f}")
        elif debug:
            print(f"    NO EVIDENCE FOUND for {candidate}")

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

    def set_definition_words(self, evidence: AnagramEvidence,
                             definition_words: List[str]) -> AnagramEvidence:
        """
        Set definition words from external pipeline data and recalculate remaining words.

        The definition typically comes from the pipeline's window_support.
        This method updates the evidence object with the definition and
        removes definition words from remaining_words.

        Args:
            evidence: AnagramEvidence object to update
            definition_words: List of words that form the definition

        Returns:
            Updated AnagramEvidence object
        """
        evidence.definition_words = definition_words

        # Remove definition words from remaining_words
        def_words_lower = set(w.lower() for w in definition_words)
        evidence.remaining_words = [
            w for w in evidence.remaining_words
            if w.lower() not in def_words_lower
        ]

        # Also update deprecated field for backward compatibility
        evidence.unused_clue_words = evidence.remaining_words

        return evidence

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
            print(f"  DETECTED INDICATORS: {indicators['anagram']}")
            for match in indicators.get('anagram_matches', []):
                print(f"    Position {match.start_pos}: '{' '.join(match.words)}'")

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

    def analyze_and_rank_anagram_candidates(self, clue_text: str, candidates: List[str],
                                            answer: str, debug: bool = False) -> Dict[
        str, any]:
        """
        ACTUAL WORKING LOGIC MOVED FROM evidence_analysis.py

        Performs comprehensive anagram analysis and ranking for all candidates.
        This is the proven working method that evidence_analysis.py was using directly.

        Args:
            clue_text: The cryptic clue text
            candidates: List of all definition candidates to analyze
            answer: The target answer for validation
            debug: Enable debug output

        Returns:
            Dict containing complete ranked candidate information
        """
        if not candidates:
            return {
                "evidence_list": [],
                "scored_candidates": [],
                "answer_rank_original": None,
                "answer_rank_evidence": None,
                "ranking_improved": False,
                "evidence_found": 0
            }

        # Use the working anagram system that evidence_analysis.py uses
        enumeration_num = len(answer) if answer else 0

        if debug:
            print(f"DEBUG: Calling generate_anagram_hypotheses for '{clue_text[:50]}...'")
            print(
                f"DEBUG: enumeration_num={enumeration_num}, candidates_count={len(candidates)}")

        hypotheses = generate_anagram_hypotheses(clue_text, enumeration_num, candidates)

        if debug:
            print(f"DEBUG: Got {len(hypotheses)} hypotheses")
            if hypotheses:
                print(f"DEBUG: First hypothesis: {hypotheses[0]}")

        # Convert hypotheses to AnagramEvidence objects
        evidence_list = []
        for hyp in hypotheses:
            # Convert confidence from string to float if needed
            confidence_raw = hyp.get("confidence", 1.0)
            if debug:
                print(
                    f"      DEBUG: Raw confidence: {confidence_raw} (type: {type(confidence_raw)})")

            if isinstance(confidence_raw, str):
                # Map string confidence to numeric values
                confidence_map = {'provisional': 0.5, 'high': 0.9, 'medium': 0.7,
                                  'low': 0.3}
                confidence = confidence_map.get(confidence_raw.lower(), 0.5)
                if debug:
                    print(
                        f"      DEBUG: Converted confidence '{confidence_raw}' to {confidence}")
            else:
                confidence = confidence_raw

            # Create evidence object using the AnagramEvidence dataclass
            evidence = AnagramEvidence(
                candidate=hyp.get("answer", ""),
                fodder_words=hyp.get("fodder_words", []),
                fodder_letters=hyp.get("fodder_letters", ""),
                evidence_type=hyp.get("evidence_type", hyp.get("solve_type", "exact")),
                confidence=confidence,
                excess_letters=hyp.get("excess_letters", ""),
                needed_letters=hyp.get("needed_letters", ""),
                unused_clue_words=hyp.get("unused_words", [])
            )
            evidence_list.append(evidence)

        # Create scored candidates list - PRESERVES ALL RANKED CANDIDATE INFORMATION
        scored_candidates = []
        evidence_by_candidate = {ev.candidate.upper(): ev for ev in evidence_list}

        for candidate in candidates:
            candidate_upper = candidate.upper()
            evidence = evidence_by_candidate.get(candidate_upper)

            # Calculate evidence score boost using existing proven method
            evidence_score = 0.0
            if evidence:
                evidence_score = self.calculate_anagram_score_boost(evidence)

            scored_candidates.append({
                "candidate": candidate,
                "evidence_score": evidence_score,
                "evidence": evidence,
                "has_evidence": evidence is not None
            })

        # Sort by evidence score (highest first) - PRESERVES COMPLETE RANKING
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

        # Return complete ranked candidate information
        return {
            "evidence_list": evidence_list,
            "scored_candidates": scored_candidates,  # COMPLETE RANKED LIST
            "answer_rank_original": answer_rank_original,
            "answer_rank_evidence": answer_rank_evidence,
            "ranking_improved": (answer_rank_evidence and answer_rank_original and
                                 answer_rank_evidence < answer_rank_original),
            "evidence_found": len(evidence_list)
        }