#!/usr/bin/env python3
"""
Compound Wordplay Analyzer Engine - DATABASE-INTEGRATED VERSION

This engine:
1. Receives evidence-enhanced cases from evidence analysis
2. Queries the indicators table to identify wordplay types for remaining words
3. Queries the wordplay table for substitutions
4. Applies operation solvers (insertion, container, deletion, reversal, etc.)
5. Builds complete explanations with formula notation

Database tables used:
- indicators: word -> wordplay_type, subtype, confidence
- wordplay: indicator -> substitution, category
- synonyms_pairs: word -> synonym (fallback)
"""

import sys
import sqlite3
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(r'C:\Users\shute\PycharmProjects\cryptic_solver')

from solver.solver_engine.resources import norm_letters

DB_PATH = r'C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db'


class WordplayType(Enum):
    """Wordplay types from indicators table."""
    ANAGRAM = 'anagram'
    CONTAINER = 'container'
    INSERTION = 'insertion'
    DELETION = 'deletion'
    REVERSAL = 'reversal'
    HIDDEN = 'hidden'
    HOMOPHONE = 'homophone'
    PARTS = 'parts'
    ACROSTIC = 'acrostic'
    SELECTION = 'selection'
    UNKNOWN = 'unknown'


@dataclass
class IndicatorMatch:
    """Result of looking up a word in the indicators table."""
    word: str
    wordplay_type: str
    subtype: Optional[str]
    confidence: str


@dataclass
class SubstitutionMatch:
    """Result of looking up a word in the wordplay table."""
    word: str
    letters: str
    category: str
    notes: Optional[str] = None


@dataclass
class WordRole:
    """Role of a word in the clue."""
    word: str
    role: str  # definition, fodder, indicator, substitution, positional, link, etc.
    contributes: str  # What letters/meaning it contributes
    source: str  # Where we determined this (evidence, database, heuristic)


class DatabaseLookup:
    """Handles all database queries for indicators and substitutions."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn = None
        self._indicator_cache = {}
        self._substitution_cache = {}

    def _get_connection(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def lookup_indicator(self, word: str) -> Optional[IndicatorMatch]:
        """Look up a word in the indicators table."""
        # Strip punctuation but preserve spaces for two-word indicators
        word_clean = ''.join(c for c in word.lower() if c.isalpha() or c == ' ')
        word_clean = ' '.join(word_clean.split())  # Normalize whitespace

        if not word_clean:
            return None

        if word_clean in self._indicator_cache:
            return self._indicator_cache[word_clean]

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT word, wordplay_type, subtype, confidence
            FROM indicators
            WHERE LOWER(word) = ?
        """, (word_clean,))

        result = cursor.fetchone()

        if result:
            match = IndicatorMatch(
                word=result[0],
                wordplay_type=result[1],
                subtype=result[2],
                confidence=result[3]
            )
            self._indicator_cache[word_clean] = match
            return match

        self._indicator_cache[word_clean] = None
        return None

    def lookup_substitution(self, word: str) -> List[SubstitutionMatch]:
        """Look up a word in the wordplay table for substitutions."""
        # Strip punctuation for lookup
        word_clean = ''.join(c for c in word.lower() if c.isalpha())

        if not word_clean:
            return []

        if word_clean in self._substitution_cache:
            return self._substitution_cache[word_clean]

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT indicator, substitution, category, notes
            FROM wordplay
            WHERE LOWER(indicator) = ?
        """, (word_clean,))

        results = cursor.fetchall()

        matches = [
            SubstitutionMatch(
                word=r[0],
                letters=r[1],
                category=r[2],
                notes=r[3]
            )
            for r in results
        ]

        self._substitution_cache[word_clean] = matches
        return matches

    def lookup_synonym_as_substitution(self, word: str, max_length: int = 3) -> List[
        Tuple[str, str]]:
        """
        Look up synonyms that could be short substitutions.
        Returns list of (synonym, source) tuples.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT synonym FROM synonyms_pairs
            WHERE LOWER(word) = ? AND LENGTH(synonym) <= ?
        """, (word.lower(), max_length))

        return [(r[0], 'synonym') for r in cursor.fetchall()]


class CompoundSolver:
    """
    Solves compound wordplay by identifying and applying operations.
    """

    def __init__(self, db_lookup: DatabaseLookup):
        self.db = db_lookup

    def find_substitution_for_letters(self, word: str, needed_letters: str,
                                      used_letters: Set[str]) -> Optional[
        SubstitutionMatch]:
        """
        Find a substitution that provides exactly the needed letters.
        Returns the first valid match.
        """
        matches = self.db.lookup_substitution(word)

        for match in matches:
            if match.letters.upper() == needed_letters.upper():
                # Check letters aren't already used
                if not any(c in used_letters for c in match.letters.upper()):
                    return match

        return None

    def solve_insertion(self, anagram_letters: str, extra_letters: str,
                        answer: str) -> Optional[Dict[str, Any]]:
        """
        Solve insertion: extra_letters go INTO anagram_letters to form answer.

        Example: ABUNDANT = anagram(TUNABAN) with D inserted
        """
        answer_upper = answer.upper().replace(' ', '')
        anagram_upper = anagram_letters.upper()
        extra_upper = extra_letters.upper()

        # Try inserting extra_letters at each position in anagram
        for i in range(len(anagram_upper) + 1):
            combined = anagram_upper[:i] + extra_upper + anagram_upper[i:]
            if sorted(combined) == sorted(answer_upper):
                return {
                    'operation': 'insertion',
                    'base': anagram_letters,
                    'inserted': extra_letters,
                    'position': i,
                    'result': answer
                }

        return None

    def solve_container(self, outer_letters: str, inner_letters: str,
                        answer: str) -> Optional[Dict[str, Any]]:
        """
        Solve container: outer_letters wrap AROUND inner_letters.

        Example: C + ART + ON = CARTON (ART inside CON)
        """
        answer_upper = answer.upper().replace(' ', '')

        # Try each split of outer as prefix + suffix around inner
        outer_upper = outer_letters.upper()
        inner_upper = inner_letters.upper()

        for i in range(len(outer_upper) + 1):
            prefix = outer_upper[:i]
            suffix = outer_upper[i:]
            combined = prefix + inner_upper + suffix
            if combined == answer_upper:
                return {
                    'operation': 'container',
                    'outer': outer_letters,
                    'inner': inner_letters,
                    'prefix': prefix,
                    'suffix': suffix,
                    'result': answer
                }

        return None

    def solve_deletion(self, base_letters: str, delete_letters: str,
                       answer: str) -> Optional[Dict[str, Any]]:
        """
        Solve deletion: remove delete_letters from base_letters.
        """
        base_upper = base_letters.upper()
        delete_upper = delete_letters.upper()
        answer_upper = answer.upper().replace(' ', '')

        # Try removing the delete letters
        remaining = base_upper
        for c in delete_upper:
            idx = remaining.find(c)
            if idx >= 0:
                remaining = remaining[:idx] + remaining[idx + 1:]

        if sorted(remaining) == sorted(answer_upper):
            return {
                'operation': 'deletion',
                'base': base_letters,
                'deleted': delete_letters,
                'result': answer
            }

        return None

    def solve_reversal(self, letters: str, answer: str) -> Optional[Dict[str, Any]]:
        """
        Solve reversal: reversed letters form the answer.
        """
        if letters.upper()[::-1] == answer.upper().replace(' ', ''):
            return {
                'operation': 'reversal',
                'original': letters,
                'result': answer
            }
        return None


class CompoundWordplayAnalyzer:
    """
    Main analyzer that integrates evidence analysis with database lookups
    to build complete explanations.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db = DatabaseLookup(db_path)
        self.solver = CompoundSolver(self.db)

        # Link words that don't contribute letters
        # Expanded based on analysis of unresolved words
        self.link_words = {
            # Articles and prepositions
            'of', 'in', 'the', 'a', 'an', 'to', 'for', 'with',
            'and', 'or', 'by', 'from', 'as', 'on', 'at',
            # Common verbs used as links
            'is', 'are', 'be', 'being', 'been',
            'has', 'have', 'having', 'had',
            'was', 'were', 'will', 'would',
            'could', 'should', 'must', 'may', 'might',
            'gets', 'get', 'getting', 'got',
            'needs', 'need', 'needs',
            'makes', 'make', 'making', 'made',
            'gives', 'give', 'given', 'giving',
            'sees', 'see', 'seen', 'seeing',
            'brings', 'bring', 'bringing', 'brought',
            # Contractions (with apostrophe)
            "it's", "that's", "there's", "here's", "what's",
            "i'm", "i've", "i'd", "you're", "you've", "you'd",
            "he's", "she's", "we're", "we've", "they're", "they've",
            "don't", "doesn't", "didn't", "won't", "wouldn't",
            "can't", "couldn't", "shouldn't", "isn't", "aren't",
            # Contractions (apostrophe-stripped - for norm_letters matching)
            'its', 'thats', 'theres', 'heres', 'whats',
            'im', 'ive', 'id', 'youre', 'youve', 'youd',
            'hes', 'shes', 'were', 'weve', 'theyre', 'theyve',
            'dont', 'doesnt', 'didnt', 'wont', 'wouldnt',
            'cant', 'couldnt', 'shouldnt', 'isnt', 'arent',
            # Conjunctions and connectors
            'but', 'that', 'which', 'when', 'where', 'while',
            'so', 'yet', 'thus', 'hence', 'therefore',
            # Other common links
            'this', 'these', 'those', 'such',
            'one', 'ones', 'some', 'any', 'all',
            'here', 'there', 'maybe',
            # Common link phrases often appearing
            'into', 'onto', 'within', 'without',
            'find', 'found', 'finding', 'show', 'showing',
            'put', 'set', 'provide', 'providing',
            'if', 'how', 'why', 'who', 'whom',
            # Words that connect fodder to definition
            'giving', 'producing', 'causing', 'creating',
            'offering', 'providing', 'yielding', 'making',
        }

        # Positional indicators that show construction order
        self.positional_words = {'after', 'before', 'following', 'preceding',
                                 'then', 'first', 'finally', 'initially'}

    def close(self):
        self.db.close()

    def _is_enumeration(self, word: str) -> bool:
        """Check if a word is an enumeration pattern like '8', '2,5', '3-4', '2,3,4'."""
        # Remove common punctuation
        cleaned = word.strip('()[]')

        # Pure digits
        if cleaned.isdigit():
            return True

        # Comma or hyphen separated digits: "2,5" or "3-4" or "2,3,4"
        if all(c.isdigit() or c in ',.-' for c in cleaned) and any(
                c.isdigit() for c in cleaned):
            return True

        return False

    def analyze_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single case with evidence data and build complete attribution.

        IMPORTANT: We use the MATCHED CANDIDATE from brute force as our "likely answer",
        NOT the database answer. The database answer is only for comparison/verification.
        """
        clue_text = case.get('clue', '')
        clue_words = clue_text.replace('(', ' ').replace(')', ' ').split()

        # Filter out enumeration patterns (digits, comma-separated numbers)
        clue_words = [w for w in clue_words if not self._is_enumeration(w)]

        # Database answer - ONLY for display/comparison, never for solving
        db_answer = case.get('answer', '').upper().replace(' ', '')

        # Get evidence analysis results
        evidence_analysis = case.get('evidence_analysis', {})
        scored_candidates = evidence_analysis.get('scored_candidates', [])

        if not scored_candidates:
            return self._build_fallback(case, clue_words, db_answer)

        # Use top ranked candidate - THIS is our likely answer
        top_candidate = scored_candidates[0]
        evidence = top_candidate.get('evidence')

        if not evidence:
            return self._build_fallback(case, clue_words, db_answer)

        # LIKELY ANSWER comes from the matched candidate, not database
        likely_answer = top_candidate.get('candidate', '').upper().replace(' ', '')

        # Get definition from pipeline data (needed for indicator inference)
        definition_window = self._get_definition_window(case, clue_words)

        # Extract anagram component
        fodder_words = evidence.fodder_words or []
        fodder_letters = evidence.fodder_letters or ''

        # First try to get indicator from evidence (already found by evidence system)
        anagram_indicator = None
        if hasattr(evidence, 'indicator_words') and evidence.indicator_words:
            # Evidence system already found the indicator
            anagram_indicator = ' '.join(evidence.indicator_words)

        # Fallback: search for indicator ourselves
        if not anagram_indicator:
            anagram_indicator = self._find_anagram_indicator(clue_words, fodder_words,
                                                             definition_window)

        # Build word roles tracking
        word_roles = []
        accounted_words = set()

        # Account for anagram indicator FIRST (so we can exclude from definition)
        if anagram_indicator:
            word_roles.append(
                WordRole(anagram_indicator, 'anagram_indicator', '', 'evidence'))
            # Handle both single and two-word indicators
            for ind_word in anagram_indicator.split():
                accounted_words.add(ind_word.lower())

        # Account for anagram fodder
        for fw in fodder_words:
            word_roles.append(WordRole(fw, 'fodder', fodder_letters, 'evidence'))
            accounted_words.add(fw.lower())

        # Account for definition (excluding indicator and fodder)
        if definition_window:
            def_words = definition_window.split()
            for w in def_words:
                w_norm = norm_letters(w)
                # Skip if already accounted (indicator or fodder)
                if w_norm in {norm_letters(a) for a in accounted_words}:
                    continue
                word_roles.append(WordRole(w, 'definition', likely_answer, 'pipeline'))
                accounted_words.add(w.lower())

        # Account for link words
        for w in clue_words:
            w_norm = norm_letters(w)
            if w_norm in self.link_words and w_norm not in {norm_letters(a) for a in
                                                            accounted_words}:
                word_roles.append(WordRole(w, 'link', '', 'heuristic'))
                accounted_words.add(w.lower())

        # Helper to normalize word for comparison (strip punctuation)
        def normalize_word(w):
            return ''.join(c.lower() for c in w if c.isalpha())

        # Find remaining words (compare normalized forms)
        remaining_words = []
        for w in clue_words:
            w_norm = normalize_word(w)
            # Check if normalized form matches any accounted word
            if w_norm not in {normalize_word(aw) for aw in accounted_words}:
                remaining_words.append(w)

        # Analyze remaining words using database
        # Use LIKELY_ANSWER (from candidate), not db_answer
        compound_solution = None
        if remaining_words:
            compound_solution = self._analyze_remaining_words(
                remaining_words, fodder_letters, likely_answer, word_roles,
                accounted_words,
                clue_words, definition_window
            )

        # Build explanation using LIKELY_ANSWER
        explanation = self._build_explanation(
            case, word_roles, fodder_words, fodder_letters,
            anagram_indicator, definition_window, compound_solution, clue_words,
            likely_answer
        )

        return {
            'clue': clue_text,
            'likely_answer': likely_answer,  # What we solved
            'db_answer': db_answer,  # For comparison only
            'answer_matches': norm_letters(likely_answer) == norm_letters(db_answer),
            # Verification
            'word_roles': word_roles,
            'definition_window': definition_window,
            'anagram_component': {
                'fodder_words': fodder_words,
                'fodder_letters': fodder_letters,
                'indicator': anagram_indicator
            },
            'compound_solution': compound_solution,
            'explanation': explanation,
            'remaining_unresolved': [w for w in clue_words
                                     if norm_letters(w) not in {norm_letters(a) for a in
                                                                accounted_words}]
        }

    def _find_anagram_indicator(self, clue_words: List[str],
                                fodder_words: List[str],
                                definition_window: Optional[str] = None) -> Optional[str]:
        """
        Find the anagram indicator in the clue.

        CORRECTED: Now searches only remaining words (clue - definition - fodder).
        The indicator must be in the remaining words, not in the definition or fodder.

        1. First try two-word indicators in database (excluding definition and fodder)
        2. Then try single-word indicators in database (excluding definition and fodder)
        3. If not found, infer based on proximity to fodder and distance from definition
        """
        fodder_lower = {norm_letters(w) for w in fodder_words}
        def_words_lower = set()
        if definition_window:
            def_words_lower = {norm_letters(w) for w in definition_window.split()}

        # First pass: look for TWO-WORD indicators in database
        for i in range(len(clue_words) - 1):
            word1 = clue_words[i]
            word2 = clue_words[i + 1]
            # Skip if either word is fodder or definition
            if norm_letters(word1) in fodder_lower or norm_letters(word2) in fodder_lower:
                continue
            if norm_letters(word1) in def_words_lower or norm_letters(
                    word2) in def_words_lower:
                continue
            # Build two-word phrase (strip punctuation for lookup)
            two_word = f"{norm_letters(word1)} {norm_letters(word2)}"
            indicator_match = self.db.lookup_indicator(two_word)
            if indicator_match and indicator_match.wordplay_type == 'anagram':
                return f"{word1} {word2}"  # Return original form with punctuation

        # Second pass: look for SINGLE-WORD indicators in database
        for word in clue_words:
            # Skip if word is fodder or definition
            if norm_letters(word) in fodder_lower:
                continue
            if norm_letters(word) in def_words_lower:
                continue
            indicator_match = self.db.lookup_indicator(word)
            # DEBUG: uncomment to trace indicator lookup
            # print(f"  [DEBUG] Checking '{word}' -> {indicator_match}")
            if indicator_match and indicator_match.wordplay_type == 'anagram':
                return word

        # Third pass: infer indicator based on proximity
        # Find indices of fodder words
        fodder_indices = []
        for i, word in enumerate(clue_words):
            if norm_letters(word) in fodder_lower:
                fodder_indices.append(i)

        if not fodder_indices:
            return None

        # Words adjacent to fodder (immediately before first or after last fodder)
        first_fodder_idx = min(fodder_indices)
        last_fodder_idx = max(fodder_indices)

        adjacent_candidates = []

        # Check word before first fodder
        if first_fodder_idx > 0:
            candidate = clue_words[first_fodder_idx - 1]
            if (norm_letters(candidate) not in fodder_lower and
                    norm_letters(candidate) not in def_words_lower and
                    norm_letters(candidate) not in self.link_words):
                adjacent_candidates.append((candidate, 'before', first_fodder_idx - 1))

        # Check word after last fodder
        if last_fodder_idx < len(clue_words) - 1:
            candidate = clue_words[last_fodder_idx + 1]
            if (norm_letters(candidate) not in fodder_lower and
                    norm_letters(candidate) not in def_words_lower and
                    norm_letters(candidate) not in self.link_words):
                adjacent_candidates.append((candidate, 'after', last_fodder_idx + 1))

        # Also check TWO-WORD combinations adjacent to fodder for inference
        # Check two words after last fodder
        if last_fodder_idx < len(clue_words) - 2:
            word1 = clue_words[last_fodder_idx + 1]
            word2 = clue_words[last_fodder_idx + 2]
            if (norm_letters(word1) not in fodder_lower and
                    norm_letters(word2) not in fodder_lower and
                    norm_letters(word1) not in def_words_lower):
                adjacent_candidates.append(
                    (f"{word1} {word2}", 'after_two', last_fodder_idx + 1))

        if not adjacent_candidates:
            return None

        # Prefer candidate that is furthest from definition
        # If definition is at start, prefer candidate after fodder
        # If definition is at end, prefer candidate before fodder
        best_candidate = None

        if def_words_lower:
            # Find where definition is (start or end of clue)
            first_word_norm = norm_letters(clue_words[0])
            last_word_norm = norm_letters(clue_words[-1])

            def_at_start = first_word_norm in def_words_lower
            def_at_end = last_word_norm in def_words_lower

            for candidate, position, idx in adjacent_candidates:
                if def_at_start and position in ('after', 'after_two'):
                    best_candidate = candidate
                    break
                elif def_at_end and position == 'before':
                    best_candidate = candidate
                    break

            # If no preference matched, just take first adjacent
            if not best_candidate and adjacent_candidates:
                best_candidate = adjacent_candidates[0][0]
        else:
            # No definition window, just take first adjacent candidate
            best_candidate = adjacent_candidates[0][0] if adjacent_candidates else None

        if best_candidate:
            # REMOVED: Do not insert inferred indicators into database - this pollutes data
            # self._insert_inferred_indicator(best_candidate, 'anagram')
            return best_candidate

        return None

    def _insert_inferred_indicator(self, word: str, wordplay_type: str):
        """Insert an inferred indicator into the database with low confidence."""
        # For two-word indicators, clean each word but keep the space
        if ' ' in word:
            parts = word.split()
            word_clean = ' '.join(norm_letters(p) for p in parts)
        else:
            word_clean = norm_letters(word)

        if not word_clean:
            return

        # Check if already exists
        existing = self.db.lookup_indicator(word_clean)
        if existing:
            return  # Already in database

        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO indicators (word, wordplay_type, subtype, confidence)
                VALUES (?, ?, ?, ?)
            """, (word_clean, wordplay_type, 'inferred', 'low'))
            conn.commit()
            print(f"  [INFERRED] Added '{word_clean}' as {wordplay_type} indicator")
        except Exception as e:
            pass  # Silently fail - don't break analysis

    def _get_definition_window(self, case: Dict[str, Any],
                               clue_words: List[str]) -> Optional[str]:
        """Extract definition window from pipeline data."""
        window_support = case.get('window_support', {})
        answer = case.get('answer', '').upper()

        if window_support:
            for window_text, candidates in window_support.items():
                if isinstance(candidates, list):
                    normalized = [c.upper().replace(' ', '') for c in candidates]
                    if answer.replace(' ', '') in normalized:
                        return window_text

        # Fallback: use definition from case if available
        if case.get('definition'):
            return case.get('definition')

        return None

    def _analyze_remaining_words(self, remaining_words: List[str],
                                 anagram_letters: str, answer: str,
                                 word_roles: List[WordRole],
                                 accounted_words: Set[str],
                                 clue_words: List[str],
                                 definition_window: Optional[str]) -> Optional[
        Dict[str, Any]]:
        """
        Analyze remaining words by querying the database.
        Identifies substitutions and construction operations.
        Handles both ADDITIONS (answer > anagram) and DELETIONS (anagram > answer).
        """
        answer_upper = answer.upper().replace(' ', '')
        anagram_upper = anagram_letters.upper()

        # Calculate what letters we still need (for additions)
        needed_letters = ''
        temp_anagram = list(anagram_upper)
        for c in answer_upper:
            if c in temp_anagram:
                temp_anagram.remove(c)
            else:
                needed_letters += c

        # Calculate excess letters (for deletions)
        # What's left in temp_anagram after removing all answer letters
        excess_letters = ''.join(sorted(temp_anagram))

        # Handle DELETION case: anagram has MORE letters than answer
        if not needed_letters and excess_letters:
            deletion_result = self._handle_deletion_compound(
                remaining_words, anagram_letters, answer, excess_letters,
                word_roles, accounted_words, clue_words, definition_window
            )
            # If deletion succeeded (found indicator), return it
            if deletion_result is not None:
                return deletion_result

            # No deletion indicator - try reducing fodder and looking for additions instead
            # This handles cases like GREATDANE where DOG+NEAR+GATE was grabbed but
            # the real solution is NEAR+GATE + D (daughter)
            alternative = self._try_reduced_fodder(
                remaining_words, anagram_letters, answer, excess_letters,
                word_roles, accounted_words, clue_words, definition_window
            )
            if alternative:
                return alternative

            # Still nothing - return unresolved
            return {
                'operation': 'unresolved_excess',
                'excess_letters': excess_letters,
                'fully_resolved': False,
                'note': 'Excess letters but no deletion indicator or alternative found'
            }

        if not needed_letters and not excess_letters:
            # Pure anagram - no additions or deletions needed
            return self._classify_remaining_as_indicators(
                remaining_words, word_roles, accounted_words,
                clue_words, definition_window
            )

        # Handle ADDITION case: answer has MORE letters than anagram
        # We need additional letters - look for substitutions OR additional fodder
        found_substitutions = []
        additional_fodder = []  # Words that provide needed letters directly
        operation_indicators = []
        positional_indicators = []

        # Get definition words to exclude from positional check
        def_words_lower = set()
        if definition_window:
            def_words_lower = {w.lower() for w in definition_window.split()}

        # Helper to get letters from word (strip punctuation)
        def get_letters(w):
            return ''.join(c.upper() for c in w if c.isalpha())

        # Helper to check if two strings are anagrams
        def is_anagram(s1, s2):
            return sorted(s1.upper()) == sorted(s2.upper())

        for word in remaining_words:
            word_lower = word.lower()
            word_letters = get_letters(word)

            # Check if it's a positional indicator (but not in definition)
            if word_lower in self.positional_words and word_lower not in def_words_lower:
                positional_indicators.append(word)
                word_roles.append(WordRole(word, 'positional_indicator', '', 'database'))
                accounted_words.add(word_lower)
                continue

            # Check indicators table for operation type
            indicator_match = self.db.lookup_indicator(word)
            if indicator_match:
                op_type = indicator_match.wordplay_type
                if op_type in ('insertion', 'container', 'deletion', 'reversal',
                               'hidden'):
                    operation_indicators.append((word, indicator_match))
                    word_roles.append(WordRole(
                        word, f'{op_type}_indicator', '', 'database'
                    ))
                    accounted_words.add(word_lower)
                    continue

            # Check if this word's letters are contained in needed letters (partial fodder)
            # This handles cases like "ill" providing ILL when we need WILL
            if word_letters and needed_letters:
                # Check if all letters in this word are present in needed_letters
                temp_needed = list(needed_letters)
                can_use = True
                for c in word_letters:
                    if c in temp_needed:
                        temp_needed.remove(c)
                    else:
                        can_use = False
                        break

                if can_use:
                    additional_fodder.append((word, word_letters))
                    word_roles.append(WordRole(
                        word, 'fodder', word_letters, 'compound_analysis'
                    ))
                    accounted_words.add(word_lower)
                    # Update needed_letters by removing the used letters
                    for c in word_letters:
                        needed_letters = needed_letters.replace(c, '', 1)
                    continue

            # Check for substitution
            subs = self.db.lookup_substitution(word)
            for sub in subs:
                # Check if this substitution provides letters we need
                sub_letters = sub.letters.upper()
                if all(c in needed_letters for c in sub_letters):
                    found_substitutions.append((word, sub))
                    word_roles.append(WordRole(
                        word, 'substitution', sub_letters,
                        f'database ({sub.category})'
                    ))
                    accounted_words.add(word_lower)
                    # Update needed letters
                    for c in sub_letters:
                        needed_letters = needed_letters.replace(c, '', 1)
                    break  # Stop at first valid substitution for this word

        # Build compound solution
        solution = {
            'needed_letters_original': answer_upper,
            'anagram_provides': anagram_upper,
            'additional_fodder': [(w, letters) for w, letters in additional_fodder],
            'substitutions': [(w, s.letters, s.category) for w, s in found_substitutions],
            'operation_indicators': [(w, i.wordplay_type, i.subtype)
                                     for w, i in operation_indicators],
            'positional_indicators': positional_indicators,
            'letters_still_needed': needed_letters,
            'fully_resolved': len(needed_letters) == 0
        }

        # Try to solve the compound construction
        if found_substitutions:
            sub_letters = ''.join(s.letters for _, s in found_substitutions)
            op_type = operation_indicators[0][
                1].wordplay_type if operation_indicators else None

            if op_type == 'insertion':
                result = self.solver.solve_insertion(anagram_letters, sub_letters, answer)
                if result:
                    solution['construction'] = result
            elif op_type == 'container':
                result = self.solver.solve_container(anagram_letters, sub_letters, answer)
                if result:
                    solution['construction'] = result
            else:
                # Default: try concatenation (check if anagram + sub = answer)
                combined = anagram_upper + sub_letters.upper()
                if sorted(combined) == sorted(answer_upper):
                    solution['construction'] = {
                        'operation': 'concatenation',
                        'parts': [anagram_letters, sub_letters],
                        'result': answer
                    }

        return solution

    def _handle_deletion_compound(self, remaining_words: List[str],
                                  anagram_letters: str, answer: str,
                                  excess_letters: str,
                                  word_roles: List[WordRole],
                                  accounted_words: Set[str],
                                  clue_words: List[str],
                                  definition_window: Optional[str]) -> Dict[str, Any]:
        """
        Handle deletion compounds where anagram has MORE letters than answer.
        Example: LOVE + IRISH (9) - O (duck) = LIVERISH (8)

        Looks for:
        1. A word that substitutes to the excess letters (e.g., "duck" -> O)
        2. A deletion indicator (e.g., "out", "without", "missing")
        """
        def_words_lower = set()
        if definition_window:
            def_words_lower = {w.lower() for w in definition_window.split()}

        deletion_substitution = None
        deletion_indicator = None
        other_indicators = []

        for word in remaining_words:
            word_lower = word.lower()

            # Skip definition words
            if word_lower in def_words_lower:
                continue

            # Check if this word substitutes to the excess letters
            subs = self.db.lookup_substitution(word)
            for sub in subs:
                sub_letters = sub.letters.upper()
                if sorted(sub_letters) == sorted(excess_letters):
                    deletion_substitution = (word, sub)
                    word_roles.append(WordRole(
                        word, 'deletion_target', sub_letters,
                        f'database ({sub.category}) - letters to remove'
                    ))
                    accounted_words.add(word_lower)
                    break

            if deletion_substitution and word_lower == deletion_substitution[0].lower():
                continue

            # Check for deletion indicator
            indicator_match = self.db.lookup_indicator(word)
            if indicator_match:
                if indicator_match.wordplay_type == 'deletion':
                    deletion_indicator = (word, indicator_match)
                    word_roles.append(WordRole(
                        word, 'deletion_indicator', '', 'database'
                    ))
                    accounted_words.add(word_lower)
                elif indicator_match.wordplay_type == 'anagram':
                    # Already used for anagram, just note it
                    other_indicators.append((word, indicator_match))
                    word_roles.append(WordRole(
                        word, 'anagram_indicator', '', 'database'
                    ))
                    accounted_words.add(word_lower)

        # Build solution
        # CRITICAL: Only return a deletion solution if we have a deletion indicator
        # Without an indicator, we shouldn't assume deletion - the fodder selection may be wrong
        if deletion_indicator is None:
            # No deletion indicator - don't commit to deletion interpretation
            # Return None so the system can try other approaches
            return None

        fully_resolved = deletion_substitution is not None and deletion_indicator is not None

        solution = {
            'operation': 'deletion',
            'anagram_provides': anagram_letters.upper(),
            'excess_letters': excess_letters,
            'deletion_target': (deletion_substitution[0],
                                deletion_substitution[1].letters,
                                deletion_substitution[
                                    1].category) if deletion_substitution else None,
            'deletion_indicator': (deletion_indicator[0],
                                   deletion_indicator[1].wordplay_type,
                                   deletion_indicator[
                                       1].subtype) if deletion_indicator else None,
            'fully_resolved': fully_resolved,
            'construction': {
                'operation': 'deletion',
                'base': anagram_letters,
                'remove': excess_letters,
                'result': answer
            } if fully_resolved else None
        }

        return solution

    def _try_reduced_fodder(self, remaining_words: List[str],
                            anagram_letters: str,
                            answer: str,
                            excess_letters: str,
                            word_roles: List[WordRole],
                            accounted_words: Set[str],
                            clue_words: List[str],
                            definition_window: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        When we have excess letters but no deletion indicator, try reducing fodder
        and looking for substitutions that could provide the missing letters.

        Example: DOG+NEAR+GATE = 11 letters for 9-letter answer GREATDANE
        - Look for substitutions in remaining words: "daughter" → D
        - Try removing fodder: if we remove DOG, we have NEARGATE = 8 letters
        - 8 + D = 9 = GREATDANE ✓
        """
        answer_upper = answer.upper().replace(' ', '')
        answer_len = len(answer_upper)

        # Helper to get letters from word
        def get_letters(w):
            return ''.join(c.upper() for c in w if c.isalpha())

        # First, find what substitutions are available in remaining words
        available_subs = []
        for word in remaining_words:
            subs = self.db.lookup_substitution(word)
            for sub in subs:
                available_subs.append((word, sub.letters.upper(), sub.category))

        if not available_subs:
            return None

        # Get current fodder words from word_roles
        fodder_roles = [wr for wr in word_roles if wr.role == 'fodder']
        if len(fodder_roles) < 2:
            return None  # Need at least 2 fodder words to try removing one

        # Try removing each fodder word and see if remaining + substitution works
        for remove_role in fodder_roles:
            # Calculate letters without this fodder word
            remaining_fodder_letters = ''
            remaining_fodder_words = []
            for fr in fodder_roles:
                if fr.word != remove_role.word:
                    remaining_fodder_letters += get_letters(fr.word)
                    remaining_fodder_words.append(fr.word)

            remaining_len = len(remaining_fodder_letters)
            needed_len = answer_len - remaining_len

            if needed_len <= 0:
                continue  # Still too many letters

            # Check if any substitution provides exactly the needed letters
            for sub_word, sub_letters, sub_category in available_subs:
                if len(sub_letters) == needed_len:
                    # Check if remaining fodder + substitution = answer (as anagram)
                    combined = remaining_fodder_letters + sub_letters
                    if sorted(combined.upper()) == sorted(answer_upper):
                        # Found a working combination!
                        # Update word_roles: remove the excluded fodder, add substitution
                        new_word_roles = [wr for wr in word_roles if
                                          wr.word != remove_role.word]
                        new_word_roles.append(WordRole(
                            sub_word, 'substitution', sub_letters,
                            f'database ({sub_category})'
                        ))

                        # Update accounted words
                        new_accounted = set(accounted_words)
                        new_accounted.discard(remove_role.word.lower())
                        new_accounted.add(sub_word.lower())

                        # Check if removed word is in definition window - mark as definition
                        if definition_window:
                            def_words = {w.lower() for w in definition_window.split()}
                            if norm_letters(remove_role.word) in {norm_letters(w) for w in
                                                                  def_words}:
                                new_word_roles.append(WordRole(
                                    remove_role.word, 'definition', answer_upper,
                                    'reduced_fodder'
                                ))
                                new_accounted.add(remove_role.word.lower())

                        # Look for indicators (anagram, container, insertion)
                        operation_indicators = []
                        anagram_indicator = None
                        for word in remaining_words:
                            word_norm = norm_letters(word)
                            if word_norm in {norm_letters(a) for a in new_accounted}:
                                continue
                            indicator_match = self.db.lookup_indicator(word)
                            if indicator_match:
                                op_type = indicator_match.wordplay_type
                                if op_type == 'anagram' and anagram_indicator is None:
                                    anagram_indicator = (word, indicator_match)
                                    new_word_roles.append(WordRole(
                                        word, 'anagram_indicator', '', 'database'
                                    ))
                                    new_accounted.add(word.lower())
                                elif op_type in ('insertion', 'container'):
                                    operation_indicators.append((word, indicator_match))
                                    new_word_roles.append(WordRole(
                                        word, f'{op_type}_indicator', '', 'database'
                                    ))
                                    new_accounted.add(word.lower())

                        # Update the main word_roles and accounted_words
                        word_roles.clear()
                        word_roles.extend(new_word_roles)
                        accounted_words.clear()
                        accounted_words.update(new_accounted)

                        return {
                            'operation': 'reduced_fodder',
                            'original_fodder': anagram_letters,
                            'reduced_fodder': remaining_fodder_letters,
                            'removed_word': remove_role.word,
                            'substitutions': [(sub_word, sub_letters, sub_category)],
                            'additional_fodder': [],
                            'operation_indicators': [(w, i.wordplay_type, i.subtype)
                                                     for w, i in operation_indicators],
                            'anagram_indicator': (anagram_indicator[0], anagram_indicator[
                                1].wordplay_type) if anagram_indicator else None,
                            'fully_resolved': True,
                            'construction': {
                                'operation': 'insertion' if operation_indicators else 'concatenation',
                                'base': remaining_fodder_letters,
                                'add': sub_letters,
                                'result': answer_upper
                            }
                        }

        return None

    def _classify_remaining_as_indicators(self, remaining_words: List[str],
                                          word_roles: List[WordRole],
                                          accounted_words: Set[str],
                                          clue_words: List[str],
                                          definition_window: Optional[str]) -> Dict[
        str, Any]:
        """When anagram is complete, classify remaining words as indicators."""
        def_words_lower = set()
        if definition_window:
            def_words_lower = {w.lower() for w in definition_window.split()}

        classified = []

        for word in remaining_words:
            word_lower = word.lower()

            # Skip if in definition
            if word_lower in def_words_lower:
                continue

            # Check indicators table
            indicator_match = self.db.lookup_indicator(word)
            if indicator_match:
                classified.append((word, indicator_match.wordplay_type))
                word_roles.append(WordRole(
                    word, f'{indicator_match.wordplay_type}_indicator', '', 'database'
                ))
                accounted_words.add(word_lower)
            elif word_lower in self.positional_words:
                classified.append((word, 'positional'))
                word_roles.append(WordRole(word, 'positional_indicator', '', 'heuristic'))
                accounted_words.add(word_lower)

        return {
            'fully_resolved': True,
            'classified_indicators': classified
        }

    def _build_explanation(self, case: Dict[str, Any],
                           word_roles: List[WordRole],
                           fodder_words: List[str],
                           fodder_letters: str,
                           anagram_indicator: Optional[str],
                           definition_window: Optional[str],
                           compound_solution: Optional[Dict[str, Any]],
                           clue_words: List[str],
                           likely_answer: str) -> Dict[str, Any]:
        """
        Build a complete explanation following the format spec.

        Uses likely_answer (from matched candidate), NOT database answer.
        """
        # Get substitutions from compound solution
        subs = []
        if compound_solution and compound_solution.get('substitutions'):
            subs = compound_solution['substitutions']

        # Get additional fodder from compound solution
        additional_fodder = []
        if compound_solution and compound_solution.get('additional_fodder'):
            additional_fodder = compound_solution['additional_fodder']

        # Build fodder part - include both original and additional fodder
        all_fodder_words = list(fodder_words) if fodder_words else []
        for word, letters in additional_fodder:
            all_fodder_words.append(word.upper().replace("'", ""))  # Strip apostrophes

        fodder_part = ' + '.join(
            w.upper() for w in all_fodder_words) if all_fodder_words else ''

        # Check for deletion operation (only if validated with indicator)
        if compound_solution and compound_solution.get('operation') == 'deletion':
            deletion_target = compound_solution.get('deletion_target')
            excess = compound_solution.get('excess_letters', '')

            if deletion_target:
                word, letters, category = deletion_target
                formula = f"anagram({fodder_part}) - {letters} ({word}) = {likely_answer}"
            else:
                formula = f"anagram({fodder_part}) - {excess} = {likely_answer}"
        elif compound_solution and compound_solution.get('operation') == 'reduced_fodder':
            # Reduced fodder with substitution - show the corrected fodder
            reduced = compound_solution.get('reduced_fodder', '')
            subs_from_compound = compound_solution.get('substitutions', [])
            if subs_from_compound:
                sub_part = ' + '.join(
                    f"{letters} ({word})" for word, letters, _ in subs_from_compound)
                # Get the actual fodder words (not the removed one)
                actual_fodder = [wr.word for wr in word_roles if wr.role == 'fodder']
                fodder_part = ' + '.join(w.upper() for w in actual_fodder)
                formula = f"anagram({fodder_part}) + {sub_part} = {likely_answer}"
            else:
                formula = f"anagram({reduced}) = {likely_answer}"
        elif compound_solution and compound_solution.get(
                'operation') == 'unresolved_excess':
            # Excess letters but no deletion indicator - show honest formula
            formula = f"anagram({fodder_part}) = {likely_answer} [excess letters unresolved]"
        elif subs:
            # Compound with substitutions (additions)
            sub_part = ' + '.join(f"{letters} ({word})" for word, letters, _ in subs)

            construction = compound_solution.get('construction',
                                                 {}) if compound_solution else {}
            op = construction.get('operation', 'concatenation')

            if op == 'insertion':
                formula = f"anagram({fodder_part}) with {sub_part} inserted = {likely_answer}"
            elif op == 'container':
                formula = f"{sub_part} inside anagram({fodder_part}) = {likely_answer}"
            else:
                formula = f"anagram({fodder_part}) + {sub_part} = {likely_answer}"
        else:
            # Pure anagram
            formula = f"anagram({fodder_part}) = {likely_answer}"

        # Build word-by-word explanation following clue order
        explanations = []
        # Use norm_letters for lookup to handle punctuation (e.g., "gate," matches "gate")
        role_lookup = {norm_letters(wr.word): wr for wr in word_roles}
        explained_words = set()

        for word in clue_words:
            word_norm = norm_letters(word)

            if word_norm in explained_words:
                continue

            if word_norm in role_lookup:
                wr = role_lookup[word_norm]
                explained_words.add(word_norm)

                if wr.role == 'definition':
                    explanations.append(f'• "{word}" = definition for {likely_answer}')
                elif wr.role == 'fodder':
                    explanations.append(f'• "{word}" = anagram fodder')
                elif wr.role == 'anagram_indicator':
                    explanations.append(f'• "{word}" = anagram indicator')
                elif wr.role == 'substitution':
                    explanations.append(f'• "{word}" = {wr.contributes} ({wr.source})')
                elif wr.role == 'deletion_target':
                    explanations.append(f'• "{word}" = {wr.contributes} (to be removed)')
                elif wr.role == 'deletion_indicator':
                    explanations.append(f'• "{word}" = deletion indicator')
                elif wr.role == 'positional_indicator':
                    explanations.append(
                        f'• "{word}" = positional indicator (construction order)')
                elif wr.role == 'insertion_indicator':
                    explanations.append(f'• "{word}" = insertion indicator')
                elif wr.role == 'container_indicator':
                    explanations.append(f'• "{word}" = container indicator')
                elif wr.role == 'link':
                    pass  # Skip link words in explanation
                else:
                    # Generic indicator
                    if '_indicator' in wr.role:
                        ind_type = wr.role.replace('_indicator', '')
                        explanations.append(f'• "{word}" = {ind_type} indicator')

        return {
            'formula': formula,
            'breakdown': explanations,
            'quality': self._assess_quality(compound_solution, word_roles, clue_words)
        }

    def _assess_quality(self, compound_solution: Optional[Dict[str, Any]],
                        word_roles: List[WordRole],
                        clue_words: List[str]) -> str:
        """Assess explanation quality based on word coverage."""
        accounted = {norm_letters(wr.word) for wr in word_roles}
        total_words = len(
            [w for w in clue_words if norm_letters(w) not in self.link_words])
        accounted_content = len([w for w in clue_words
                                 if norm_letters(w) in accounted and norm_letters(
                w) not in self.link_words])

        # Calculate coverage ratio
        coverage = accounted_content / total_words if total_words > 0 else 0

        # Check for required elements
        has_definition = any(wr.role == 'definition' for wr in word_roles)
        has_fodder = any(wr.role == 'fodder' for wr in word_roles)
        has_indicator = any('indicator' in wr.role for wr in word_roles)

        # Compound with substitutions
        if compound_solution and compound_solution.get('fully_resolved'):
            if coverage >= 0.9:
                return 'solved'
            else:
                return 'high'

        # Pure anagram - check if fully explained
        if has_definition and has_fodder and has_indicator:
            if coverage >= 0.9:
                return 'solved'
            elif coverage >= 0.7:
                return 'high'
            else:
                return 'medium'

        # Partial explanation
        if compound_solution and compound_solution.get('substitutions'):
            return 'medium'
        elif has_fodder and has_indicator:
            return 'medium'
        elif has_fodder:
            return 'low'
        else:
            return 'none'

    def _build_fallback(self, case: Dict[str, Any],
                        clue_words: List[str], db_answer: str) -> Dict[str, Any]:
        """Fallback when no evidence available."""
        return {
            'clue': case.get('clue', ''),
            'likely_answer': '',  # No likely answer - couldn't solve
            'db_answer': db_answer,  # For comparison only
            'answer_matches': False,
            'word_roles': [],
            'definition_window': None,
            'anagram_component': None,
            'compound_solution': None,
            'explanation': {
                'formula': 'Unable to analyze',
                'breakdown': [],
                'quality': 'none'
            },
            'remaining_unresolved': clue_words
        }


class ExplanationSystemBuilder:
    """
    Wrapper class maintaining backward compatibility with pipeline.
    Processes cases through the CompoundWordplayAnalyzer.
    """

    def __init__(self):
        self.analyzer = CompoundWordplayAnalyzer()

    def close(self):
        self.analyzer.close()

    def enhance_pipeline_data(self, evidence_enhanced_cases: List[Dict[str, Any]]) -> \
            List[Dict[str, Any]]:
        """
        Process evidence-enhanced cases through compound analysis.
        """
        enhanced = []

        print("Analyzing compound wordplay with database lookups...")

        for i, case in enumerate(evidence_enhanced_cases):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} cases...")

            try:
                result = self.analyzer.analyze_case(case)
                enhanced.append(result)
            except Exception as e:
                print(f"Warning: Could not analyze case {i + 1}: {e}")
                enhanced.append({
                    'clue': case.get('clue', ''),
                    'answer': case.get('answer', ''),
                    'error': str(e)
                })

        print(f"Analyzed {len(enhanced)} cases.")
        return enhanced

    def build_explanations(self, enhanced_cases: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        Extract explanations from analyzed cases.
        """
        return [
            {
                'clue': case.get('clue', ''),
                'likely_answer': case.get('likely_answer', ''),
                'db_answer': case.get('db_answer', ''),
                'answer_matches': case.get('answer_matches', False),
                'explanation': case.get('explanation', {}),
                'quality': case.get('explanation', {}).get('quality', 'none')
            }
            for case in enhanced_cases
        ]


def test_database_lookup():
    """Test the database lookup functionality."""
    db = DatabaseLookup()

    print("=" * 60)
    print("TESTING DATABASE LOOKUPS")
    print("=" * 60)

    print("\n1. INDICATOR LOOKUPS:")
    print("-" * 40)
    test_indicators = ['drunk', 'involving', 'around', 'without', 'back', 'after',
                       'containing']
    for word in test_indicators:
        result = db.lookup_indicator(word)
        if result:
            print(
                f"  ✓ {word:15} -> {result.wordplay_type:12} / {result.subtype or 'none'}")
        else:
            print(f"  ✗ {word:15} -> NOT FOUND")

    print("\n2. SUBSTITUTION LOOKUPS:")
    print("-" * 40)
    test_subs = ['party', 'Germany', 'love', 'time', 'church', 'company', 'English']
    for word in test_subs:
        results = db.lookup_substitution(word)
        if results:
            for r in results[:2]:  # Show max 2 per word
                print(f"  ✓ {word:15} -> {r.letters:5} ({r.category})")
        else:
            print(f"  ✗ {word:15} -> NOT FOUND")

    db.close()
    print("\n" + "=" * 60)


def test_with_sample_case():
    """Test with a sample case like DOORMAN."""
    print("\n" + "=" * 60)
    print("TESTING WITH SAMPLE CASE: DOORMAN")
    print("=" * 60)

    # Simulate a case as it would come from evidence analysis
    from dataclasses import dataclass

    @dataclass
    class MockEvidence:
        candidate: str = 'DOORMAN'
        fodder_words: list = None
        fodder_letters: str = 'NORMA'
        evidence_type: str = 'partial_anagram'
        confidence: float = 0.85

        def __post_init__(self):
            if self.fodder_words is None:
                self.fodder_words = ['Norma']

    sample_case = {
        'clue': 'One who may help guest Norma drunk after party (7)',
        'answer': 'DOORMAN',
        'definition': 'One who may help guest',
        'window_support': {
            'One who may help guest': ['DOORMAN']
        },
        'evidence_analysis': {
            'scored_candidates': [
                {
                    'candidate': 'DOORMAN',
                    'evidence': MockEvidence()
                }
            ]
        }
    }

    analyzer = CompoundWordplayAnalyzer()
    result = analyzer.analyze_case(sample_case)

    print(f"\nClue: {result['clue']}")
    print(f"Answer: {result['answer']}")
    print(f"\nExplanation:")
    print(f"  Formula: {result['explanation']['formula']}")
    print(f"  Quality: {result['explanation']['quality']}")
    print(f"\nBreakdown:")
    for line in result['explanation']['breakdown']:
        print(f"  {line}")

    if result.get('compound_solution'):
        print(f"\nCompound Solution:")
        cs = result['compound_solution']
        if cs.get('substitutions'):
            print(f"  Substitutions: {cs['substitutions']}")
        if cs.get('operation_indicators'):
            print(f"  Operation indicators: {cs['operation_indicators']}")
        if cs.get('positional_indicators'):
            print(f"  Positional indicators: {cs['positional_indicators']}")
        print(f"  Fully resolved: {cs.get('fully_resolved', False)}")

    if result.get('remaining_unresolved'):
        print(f"\nRemaining unresolved: {result['remaining_unresolved']}")

    analyzer.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_database_lookup()
    test_with_sample_case()