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
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(r'C:\Users\shute\PycharmProjects\cryptic_solver')

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
        word_lower = word.lower()

        if word_lower in self._indicator_cache:
            return self._indicator_cache[word_lower]

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT word, wordplay_type, subtype, confidence
            FROM indicators
            WHERE LOWER(word) = ?
        """, (word_lower,))

        result = cursor.fetchone()

        if result:
            match = IndicatorMatch(
                word=result[0],
                wordplay_type=result[1],
                subtype=result[2],
                confidence=result[3]
            )
            self._indicator_cache[word_lower] = match
            return match

        self._indicator_cache[word_lower] = None
        return None

    def lookup_substitution(self, word: str) -> List[SubstitutionMatch]:
        """Look up a word in the wordplay table for substitutions."""
        word_lower = word.lower()

        if word_lower in self._substitution_cache:
            return self._substitution_cache[word_lower]

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT indicator, substitution, category, notes
            FROM wordplay
            WHERE LOWER(indicator) = ?
        """, (word_lower,))

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

        self._substitution_cache[word_lower] = matches
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
        self.link_words = {'of', 'in', 'the', 'a', 'an', 'to', 'for', 'with',
                           'and', 'or', 'is', 'are', 'by', 'from', 'as', 'on'}

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
        """
        clue_text = case.get('clue', '')
        clue_words = clue_text.replace('(', ' ').replace(')', ' ').split()

        # Filter out enumeration patterns (digits, comma-separated numbers)
        clue_words = [w for w in clue_words if not self._is_enumeration(w)]

        answer = case.get('answer', '').upper().replace(' ', '')

        # Get evidence analysis results
        evidence_analysis = case.get('evidence_analysis', {})
        scored_candidates = evidence_analysis.get('scored_candidates', [])

        if not scored_candidates:
            return self._build_fallback(case, clue_words)

        # Use top ranked candidate
        top_candidate = scored_candidates[0]
        evidence = top_candidate.get('evidence')

        if not evidence:
            return self._build_fallback(case, clue_words)

        # Extract anagram component
        fodder_words = evidence.fodder_words or []
        fodder_letters = evidence.fodder_letters or ''
        anagram_indicator = self._find_anagram_indicator(clue_words, fodder_words)

        # Get definition from pipeline data
        definition_window = self._get_definition_window(case, clue_words)

        # Build word roles tracking
        word_roles = []
        accounted_words = set()

        # Account for definition
        if definition_window:
            def_words = definition_window.split()
            for w in def_words:
                word_roles.append(WordRole(w, 'definition', answer, 'pipeline'))
                accounted_words.add(w.lower())

        # Account for anagram fodder
        for fw in fodder_words:
            word_roles.append(WordRole(fw, 'fodder', fodder_letters, 'evidence'))
            accounted_words.add(fw.lower())

        # Account for anagram indicator
        if anagram_indicator:
            word_roles.append(
                WordRole(anagram_indicator, 'anagram_indicator', '', 'evidence'))
            accounted_words.add(anagram_indicator.lower())

        # Account for link words
        for w in clue_words:
            if w.lower() in self.link_words and w.lower() not in accounted_words:
                word_roles.append(WordRole(w, 'link', '', 'heuristic'))
                accounted_words.add(w.lower())

        # Find remaining words
        remaining_words = [w for w in clue_words if w.lower() not in accounted_words]

        # Analyze remaining words using database
        compound_solution = None
        if remaining_words:
            compound_solution = self._analyze_remaining_words(
                remaining_words, fodder_letters, answer, word_roles, accounted_words,
                clue_words, definition_window
            )

        # Build explanation
        explanation = self._build_explanation(
            case, word_roles, fodder_words, fodder_letters,
            anagram_indicator, definition_window, compound_solution, clue_words
        )

        return {
            'clue': clue_text,
            'answer': answer,
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
                                     if w.lower() not in accounted_words]
        }

    def _find_anagram_indicator(self, clue_words: List[str],
                                fodder_words: List[str]) -> Optional[str]:
        """Find the anagram indicator in the clue."""
        fodder_lower = {w.lower() for w in fodder_words}

        for word in clue_words:
            if word.lower() in fodder_lower:
                continue
            indicator_match = self.db.lookup_indicator(word)
            if indicator_match and indicator_match.wordplay_type == 'anagram':
                return word

        return None

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
        """
        answer_upper = answer.upper().replace(' ', '')
        anagram_upper = anagram_letters.upper()

        # Calculate what letters we still need
        needed_letters = ''

        # Find letters in answer but not in anagram
        temp_anagram = list(anagram_upper)
        for c in answer_upper:
            if c in temp_anagram:
                temp_anagram.remove(c)
            else:
                needed_letters += c

        if not needed_letters:
            # Anagram is complete, remaining words are just indicators
            return self._classify_remaining_as_indicators(
                remaining_words, word_roles, accounted_words,
                clue_words, definition_window
            )

        # We need additional letters - look for substitutions
        found_substitutions = []
        operation_indicators = []
        positional_indicators = []

        # Get definition words to exclude from positional check
        def_words_lower = set()
        if definition_window:
            def_words_lower = {w.lower() for w in definition_window.split()}

        for word in remaining_words:
            word_lower = word.lower()

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
                           clue_words: List[str]) -> Dict[str, Any]:
        """
        Build a complete explanation following the format spec.
        """
        answer = case.get('answer', '').upper()

        # Get substitutions from compound solution
        subs = []
        if compound_solution and compound_solution.get('substitutions'):
            subs = compound_solution['substitutions']

        # Build formula based on construction
        fodder_part = ' + '.join(w.upper() for w in fodder_words) if fodder_words else ''

        if subs:
            # Compound with substitutions
            sub_part = ' + '.join(f"{letters} ({word})" for word, letters, _ in subs)

            construction = compound_solution.get('construction',
                                                 {}) if compound_solution else {}
            op = construction.get('operation', 'concatenation')

            if op == 'insertion':
                formula = f"anagram({fodder_part}) with {sub_part} inserted = {answer}"
            elif op == 'container':
                formula = f"{sub_part} inside anagram({fodder_part}) = {answer}"
            else:
                formula = f"anagram({fodder_part}) + {sub_part} = {answer}"
        else:
            # Pure anagram
            formula = f"anagram({fodder_part}) = {answer}"

        # Build word-by-word explanation following clue order
        explanations = []
        role_lookup = {wr.word.lower(): wr for wr in word_roles}
        explained_words = set()

        for word in clue_words:
            word_lower = word.lower()

            if word_lower in explained_words:
                continue

            if word_lower in role_lookup:
                wr = role_lookup[word_lower]
                explained_words.add(word_lower)

                if wr.role == 'definition':
                    explanations.append(f'• "{word}" = definition for {answer}')
                elif wr.role == 'fodder':
                    explanations.append(f'• "{word}" = anagram fodder')
                elif wr.role == 'anagram_indicator':
                    explanations.append(f'• "{word}" = anagram indicator')
                elif wr.role == 'substitution':
                    explanations.append(f'• "{word}" = {wr.contributes} ({wr.source})')
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
        """Assess explanation quality."""
        accounted = {wr.word.lower() for wr in word_roles}
        total_words = len([w for w in clue_words if w.lower() not in self.link_words])
        accounted_content = len([w for w in clue_words
                                 if
                                 w.lower() in accounted and w.lower() not in self.link_words])

        if compound_solution and compound_solution.get('fully_resolved'):
            if accounted_content >= total_words * 0.9:
                return 'solved'
            else:
                return 'high'
        elif compound_solution and compound_solution.get('substitutions'):
            return 'medium'
        else:
            return 'low'

    def _build_fallback(self, case: Dict[str, Any],
                        clue_words: List[str]) -> Dict[str, Any]:
        """Fallback when no evidence available."""
        return {
            'clue': case.get('clue', ''),
            'answer': case.get('answer', ''),
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
                'answer': case.get('answer', ''),
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