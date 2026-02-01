#!/usr/bin/env python3
"""
Anagram Evidence System V2 - Candidate-Centric Approach

Instead of starting from indicators and expanding to find fodder,
this version starts from each candidate and asks:
"Which clue words can explain this candidate's letters?"

Flow:
1. Get candidate + its definition window from pipeline
2. Remove definition words from clue → wordplay pool
3. Find contiguous words from pool whose letters are ALL in candidate
4. Check remaining pool words for substitutions/operators
5. Validate indicator adjacency
6. Score by % of letters explained
"""

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import os

# Database paths
PIPELINE_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\pipeline_stages.db"
CRYPTIC_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"


@dataclass
class WordplayComponent:
    """A word/phrase from the clue and its role in the wordplay."""
    word: str
    role: str  # 'fodder', 'substitution', 'operator', 'indicator', 'definition', 'link', 'unaccounted'
    wordplay_type: str  # 'anagram', 'single_letter', 'abbreviation', 'first_letter', etc.
    contributes_letters: str  # letters this component provides to the answer
    detail: str  # human-readable explanation


@dataclass
class CandidateAnalysis:
    """Complete analysis of one candidate."""
    candidate: str
    candidate_letters: str
    definition_window: str
    clue_text: str

    components: List[WordplayComponent] = field(default_factory=list)

    fodder_words: List[str] = field(default_factory=list)
    fodder_letters: str = ""

    substitutions: List[Tuple[str, str, str]] = field(
        default_factory=list)  # (word, letters, category)
    operators: List[Tuple[str, str, str, str]] = field(
        default_factory=list)  # (indicator, operand, result, type)

    indicator_word: str = ""
    indicator_type: str = ""

    letters_explained: int = 0
    letters_total: int = 0
    fully_resolved: bool = False

    unaccounted_words: List[str] = field(default_factory=list)
    unaccounted_letters: str = ""

    score: float = 0.0


class AnagramEvidenceSystemV2:
    """Candidate-centric anagram evidence detection."""

    def __init__(self, pipeline_db: str = None, cryptic_db: str = None):
        self.pipeline_db = pipeline_db or PIPELINE_DB
        self.cryptic_db = cryptic_db or CRYPTIC_DB

        # Load indicators and wordplay from database
        self.anagram_indicators: Set[str] = set()
        self.all_indicators: Dict[str, str] = {}  # word -> wordplay_type
        self.wordplay_subs: Dict[
            str, List[Tuple[str, str]]] = {}  # word -> [(substitution, category), ...]

        self._load_from_database()

    def _load_from_database(self):
        """Load indicators and wordplay substitutions from cryptic database."""
        try:
            conn = sqlite3.connect(self.cryptic_db)
            cursor = conn.cursor()

            # Load indicators
            cursor.execute("SELECT word, wordplay_type FROM indicators")
            for word, wtype in cursor.fetchall():
                word_lower = word.lower().strip()
                self.all_indicators[word_lower] = wtype
                if wtype == 'anagram':
                    self.anagram_indicators.add(word_lower)

            # Load wordplay substitutions
            cursor.execute("SELECT indicator, substitution, category FROM wordplay")
            for indicator, substitution, category in cursor.fetchall():
                ind_lower = indicator.lower().strip()
                if ind_lower not in self.wordplay_subs:
                    self.wordplay_subs[ind_lower] = []
                self.wordplay_subs[ind_lower].append((substitution, category))

            conn.close()
            print(f"Loaded {len(self.anagram_indicators)} anagram indicators")
            print(f"Loaded {len(self.all_indicators)} total indicators")
            print(f"Loaded {len(self.wordplay_subs)} wordplay substitution entries")

        except Exception as e:
            print(f"Error loading from database: {e}")

    def normalize_letters(self, text: str) -> str:
        """Extract and lowercase only alphabetic characters."""
        return ''.join(c.lower() for c in text if c.isalpha())

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens, cleaning punctuation."""
        tokens = text.split()
        cleaned = []
        for token in tokens:
            # Strip standard and curly quotes, punctuation
            clean = token.strip('.,;:!?"()[]{}\'""''`')
            if clean and not clean.replace(',', '').replace('-', '').isdigit():
                cleaned.append(clean)
        return cleaned

    def letters_subset_of(self, source: str, target: str) -> bool:
        """Check if all letters in source are contained in target."""
        source_counter = Counter(self.normalize_letters(source))
        target_counter = Counter(self.normalize_letters(target))

        for letter, count in source_counter.items():
            if target_counter[letter] < count:
                return False
        return True

    def remaining_letters(self, target: str, source: str) -> str:
        """Return letters in target not explained by source."""
        target_counter = Counter(self.normalize_letters(target))
        source_counter = Counter(self.normalize_letters(source))

        remaining = target_counter - source_counter
        return ''.join(sorted(remaining.elements()))

    def find_best_fodder(self, candidate_letters: str, wordplay_pool: List[str]) -> Tuple[
        List[str], str, int, int]:
        """
        Find the best contiguous fodder sequence from wordplay_pool.

        Returns: (fodder_words, fodder_letters, start_index, end_index)

        "Best" = longest sequence whose letters are ALL in candidate.
        """
        best_fodder = ([], "", -1, -1)
        best_letter_count = 0

        n = len(wordplay_pool)

        # Try all contiguous sequences
        for start in range(n):
            for end in range(start + 1, n + 1):
                words = wordplay_pool[start:end]
                letters = self.normalize_letters(' '.join(words))

                # Check if ALL fodder letters are in candidate
                if self.letters_subset_of(letters, candidate_letters):
                    if len(letters) > best_letter_count:
                        best_fodder = (words, letters, start, end)
                        best_letter_count = len(letters)

        return best_fodder

    def find_substitutions(self, needed_letters: str, pool_words: List[str]) -> List[
        Tuple[str, str, str]]:
        """
        Find words in pool that substitute to provide needed letters.

        Returns: [(word, substitution_letters, category), ...]
        """
        if not needed_letters:
            return []

        results = []
        needed_counter = Counter(needed_letters.lower())

        for word in pool_words:
            word_lower = word.lower().strip('.,;:!?')

            if word_lower in self.wordplay_subs:
                for substitution, category in self.wordplay_subs[word_lower]:
                    sub_letters = self.normalize_letters(substitution)
                    sub_counter = Counter(sub_letters)

                    # Check if substitution provides only letters we need
                    provides_needed = True
                    for letter, count in sub_counter.items():
                        if needed_counter[letter] < count:
                            provides_needed = False
                            break

                    if provides_needed and sub_letters:
                        results.append((word, sub_letters, category))

        # Also check two-word phrases
        for i in range(len(pool_words) - 1):
            phrase = f"{pool_words[i].lower().strip('.,;:!?')} {pool_words[i + 1].lower().strip('.,;:!?')}"

            if phrase in self.wordplay_subs:
                for substitution, category in self.wordplay_subs[phrase]:
                    sub_letters = self.normalize_letters(substitution)
                    sub_counter = Counter(sub_letters)

                    provides_needed = True
                    for letter, count in sub_counter.items():
                        if needed_counter[letter] < count:
                            provides_needed = False
                            break

                    if provides_needed and sub_letters:
                        results.append((phrase, sub_letters, category))

        return results

    def find_anagram_indicator(self, wordplay_pool: List[str], fodder_start: int,
                               fodder_end: int) -> Tuple[str, int]:
        """
        Find an anagram indicator adjacent to the fodder.

        Returns: (indicator_word, indicator_index) or ("", -1) if not found
        """
        # Check word immediately before fodder
        if fodder_start > 0:
            word = wordplay_pool[fodder_start - 1]
            if word.lower().strip('.,;:!?') in self.anagram_indicators:
                return (word, fodder_start - 1)

        # Check word immediately after fodder
        if fodder_end < len(wordplay_pool):
            word = wordplay_pool[fodder_end]
            if word.lower().strip('.,;:!?') in self.anagram_indicators:
                return (word, fodder_end)

        return ("", -1)

    def analyze_candidate(self, candidate: str, definition_window: str,
                          clue_text: str) -> CandidateAnalysis:
        """
        Analyze one candidate using candidate-centric approach.
        """
        analysis = CandidateAnalysis(
            candidate=candidate,
            candidate_letters=self.normalize_letters(candidate),
            definition_window=definition_window,
            clue_text=clue_text,
            letters_total=len(self.normalize_letters(candidate))
        )

        # Tokenize clue
        tokens = self.tokenize(clue_text)

        # Remove definition window words from pool
        def_words = set(w.lower() for w in self.tokenize(definition_window))
        wordplay_pool = [t for t in tokens if t.lower() not in def_words]

        # Find best fodder
        fodder_words, fodder_letters, fodder_start, fodder_end = self.find_best_fodder(
            analysis.candidate_letters, wordplay_pool
        )

        analysis.fodder_words = fodder_words
        analysis.fodder_letters = fodder_letters

        if not fodder_words:
            # No valid fodder found
            analysis.unaccounted_words = wordplay_pool
            analysis.unaccounted_letters = analysis.candidate_letters
            return analysis

        # Calculate remaining letters needed
        remaining = self.remaining_letters(analysis.candidate_letters, fodder_letters)

        # Get remaining pool words (not fodder, not definition)
        fodder_set = set(w.lower() for w in fodder_words)
        remaining_pool = [t for t in wordplay_pool if t.lower() not in fodder_set]

        # Find substitutions for remaining letters
        if remaining:
            subs = self.find_substitutions(remaining, remaining_pool)

            # Greedily apply substitutions
            for word, sub_letters, category in subs:
                if not remaining:
                    break

                # Check if this sub's letters are still needed
                temp_remaining = remaining
                can_use = True
                for c in sub_letters:
                    if c in temp_remaining:
                        temp_remaining = temp_remaining.replace(c, '', 1)
                    else:
                        can_use = False
                        break

                if can_use:
                    analysis.substitutions.append((word, sub_letters, category))
                    remaining = temp_remaining

        # Find anagram indicator
        indicator_word, indicator_idx = self.find_anagram_indicator(wordplay_pool,
                                                                    fodder_start,
                                                                    fodder_end)
        analysis.indicator_word = indicator_word
        analysis.indicator_type = 'anagram' if indicator_word else ''

        # Calculate results
        letters_from_fodder = len(fodder_letters)
        letters_from_subs = sum(len(sub[1]) for sub in analysis.substitutions)
        analysis.letters_explained = letters_from_fodder + letters_from_subs
        analysis.unaccounted_letters = remaining
        analysis.fully_resolved = (remaining == "" and indicator_word != "")

        # Score: % explained, bonus for indicator, bonus for fully resolved
        pct_explained = analysis.letters_explained / analysis.letters_total if analysis.letters_total > 0 else 0
        analysis.score = pct_explained * 100
        if indicator_word:
            analysis.score += 20
        if analysis.fully_resolved:
            analysis.score += 30

        # Track unaccounted words
        accounted = fodder_set | set(w.lower() for w, _, _ in analysis.substitutions)
        if indicator_word:
            accounted.add(indicator_word.lower())
        analysis.unaccounted_words = [t for t in remaining_pool if
                                      t.lower() not in accounted]

        return analysis

    def create_results_table(self):
        """Create table for v2 results in pipeline_stages database."""
        conn = sqlite3.connect(self.pipeline_db)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stage_evidence_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                clue_id INTEGER,
                clue_text TEXT,
                enumeration TEXT,
                db_answer TEXT,
                candidate TEXT,
                definition_window TEXT,
                fodder_words TEXT,
                fodder_letters TEXT,
                substitutions TEXT,
                indicator_word TEXT,
                letters_explained INTEGER,
                letters_total INTEGER,
                fully_resolved INTEGER,
                score REAL,
                unaccounted_words TEXT,
                unaccounted_letters TEXT
            )
        """)

        conn.commit()
        conn.close()
        print("Created stage_evidence_v2 table")

    def clear_results(self, run_id: int = None):
        """Clear previous results."""
        conn = sqlite3.connect(self.pipeline_db)
        cursor = conn.cursor()

        if run_id is not None:
            cursor.execute("DELETE FROM stage_evidence_v2 WHERE run_id = ?", (run_id,))
        else:
            cursor.execute("DELETE FROM stage_evidence_v2")

        conn.commit()
        conn.close()

    def save_analysis(self, run_id: int, clue_id: int, clue_text: str, enumeration: str,
                      db_answer: str, analysis: CandidateAnalysis):
        """Save one candidate analysis to database."""
        conn = sqlite3.connect(self.pipeline_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO stage_evidence_v2 
            (run_id, clue_id, clue_text, enumeration, db_answer, candidate, definition_window,
             fodder_words, fodder_letters, substitutions, indicator_word,
             letters_explained, letters_total, fully_resolved, score,
             unaccounted_words, unaccounted_letters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            clue_id,
            clue_text,
            enumeration,
            db_answer,
            analysis.candidate,
            analysis.definition_window,
            '|'.join(analysis.fodder_words),
            analysis.fodder_letters,
            '|'.join(f"{w}:{l}:{c}" for w, l, c in analysis.substitutions),
            analysis.indicator_word,
            analysis.letters_explained,
            analysis.letters_total,
            1 if analysis.fully_resolved else 0,
            analysis.score,
            '|'.join(analysis.unaccounted_words),
            analysis.unaccounted_letters
        ))

        conn.commit()
        conn.close()

    def run_on_pipeline_data(self, run_id: int = 0, limit: int = None):
        """
        Run v2 analysis on data from pipeline stages.

        Reads from stage_definition to get candidates.
        Uses support column (if available) to get definition windows per candidate.
        """
        self.create_results_table()
        self.clear_results(run_id)

        conn = sqlite3.connect(self.pipeline_db)
        cursor = conn.cursor()

        # Check if support column exists
        cursor.execute("PRAGMA table_info(stage_definition)")
        columns = [col[1] for col in cursor.fetchall()]
        has_support = 'support' in columns

        # Get definition stage data - ONLY clues where answer is in candidates
        if has_support:
            query = """
                SELECT clue_id, clue_text, answer, candidates, support
                FROM stage_definition
                WHERE run_id = ? AND answer_in_candidates = 1
            """
        else:
            query = """
                SELECT clue_id, clue_text, answer, candidates
                FROM stage_definition
                WHERE run_id = ? AND answer_in_candidates = 1
            """
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, (run_id,))
        rows = cursor.fetchall()
        conn.close()

        print(f"\nProcessing {len(rows)} clues from stage_definition...")
        print(f"Support column available: {has_support}")

        results_summary = {'total': 0, 'resolved': 0, 'with_indicator': 0,
                           'answer_resolved': 0}

        for idx, row in enumerate(rows):
            if has_support:
                clue_id, clue_text, db_answer, candidates_json, support_json = row
            else:
                clue_id, clue_text, db_answer, candidates_json = row
                support_json = None

            # Derive enumeration from answer length
            enumeration = str(len(db_answer)) if db_answer else ""

            # Parse candidates
            import json
            try:
                candidates = json.loads(candidates_json) if candidates_json else []
            except:
                candidates = candidates_json.split('|') if candidates_json else []

            # Parse support dict (candidate -> list of definition windows)
            support = {}
            if support_json:
                try:
                    support = json.loads(support_json)
                except:
                    pass

            # Track if the correct answer gets resolved
            answer_resolved_this_clue = False

            # Analyze each candidate
            for candidate in candidates:
                if not candidate:
                    continue

                # Get definition window for this candidate from support dict
                def_window = ""
                if support and candidate in support:
                    windows = support[candidate]
                    if isinstance(windows, list) and windows:
                        def_window = windows[0]  # Use first window
                    elif isinstance(windows, str):
                        def_window = windows

                analysis = self.analyze_candidate(candidate, def_window, clue_text)

                # Save to database
                self.save_analysis(run_id, clue_id, clue_text, enumeration, db_answer,
                                   analysis)

                results_summary['total'] += 1
                if analysis.fully_resolved:
                    results_summary['resolved'] += 1
                    if candidate.upper() == db_answer.upper():
                        answer_resolved_this_clue = True
                if analysis.indicator_word:
                    results_summary['with_indicator'] += 1

            if answer_resolved_this_clue:
                results_summary['answer_resolved'] += 1

            # Progress every 10 clues
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1} clues...")

        print(f"\nResults summary:")
        print(f"  Total candidates analyzed: {results_summary['total']}")
        print(f"  Fully resolved: {results_summary['resolved']}")
        print(f"  With anagram indicator: {results_summary['with_indicator']}")
        print(
            f"  Clues where correct answer resolved: {results_summary['answer_resolved']}")

        return results_summary


def main():
    """Run v2 on pipeline data."""
    print("=" * 60)
    print("ANAGRAM EVIDENCE SYSTEM V2 - Candidate-Centric Approach")
    print("=" * 60)

    v2 = AnagramEvidenceSystemV2()

    # Run on pipeline data
    v2.run_on_pipeline_data(run_id=0, limit=100)


if __name__ == "__main__":
    main()