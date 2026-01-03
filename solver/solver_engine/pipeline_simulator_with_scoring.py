# solver/solver_engine/pipeline_simulator_with_scoring.py
#
# Enhanced Pipeline simulator with word-frequency scoring:
# - runs clues through ALL stages
# - preserves evidence
# - optional wordplay_type filter
# - ADDS: word-frequency based candidate scoring using historical clue data
#   * wordplay_type="all" => no filtering
#   * otherwise case-insensitive match IF column exists

from __future__ import annotations

import re
import csv
import os
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional
from pathlib import Path

from solver.solver_engine.resources import (
    connect_db,
    load_graph,
    parse_enum,
    norm_letters,
    clean_key,
)

from solver.definition.definition_engine_edges import definition_candidates
from solver.wordplay.double_definition.dd_stage import generate_dd_hypotheses
from solver.wordplay.anagram.anagram_stage import generate_anagram_hypotheses
from solver.wordplay.lurker.lurker_stage import generate_lurker_hypotheses

# ==============================
# SIMULATOR CONFIGURATION
# ==============================

MAX_CLUES = 500
WORDPLAY_TYPE = "all"  # e.g. "all", "anagram", "lurker", "dd"
ONLY_MISSING_DEFINITION = False  # show only clues where answer NOT in def candidates
MAX_DISPLAY = 10  # max number of clues to print
SINGLE_CLUE_MATCH = ""  # normalised substring
# match on clue_text (highest priority)

# Word frequency scoring settings
ENABLE_WORD_SCORING = True  # Enable/disable word frequency scoring
WORD_FREQUENCY_WEIGHT = 0.7  # Weight for word overlap scoring
ANSWER_FREQUENCY_WEIGHT = 0.3  # Weight for historical answer frequency
MIN_WORD_SCORE_THRESHOLD = 2  # Minimum word overlap score to consider

_ENUM_RE = re.compile(r"\(\d+(?:,\d+)*\)")

# Global variables for word frequency data
_answer_word_index = {}
_answer_frequency_index = {}
_word_frequency_loaded = False


# ==============================
# WORD FREQUENCY SCORING SYSTEM
# ==============================

def load_answer_word_frequency():
    """Load the answer-word frequency CSV file into memory."""
    global _answer_word_index, _answer_frequency_index, _word_frequency_loaded

    if _word_frequency_loaded:
        return

    # Try to find the CSV file
    data_dir = Path(r"C:\Users\shute\PycharmProjects\cryptic_solver\data")
    csv_path = data_dir / "answer_word_frequency.csv"

    if not os.path.exists(csv_path):
        print(f"WARNING: Answer word frequency file not found: {csv_path}")
        print("Word frequency scoring will be disabled.")
        return

    print(f"Loading answer-word frequency data from {csv_path}...")

    _answer_word_index = defaultdict(dict)
    _answer_frequency_index = defaultdict(int)

    rows_processed = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                answer = row['answer'].upper().strip()
                word = row['word'].lower().strip()
                count = int(row['count'])

                # Build word index: answer -> {word: count}
                _answer_word_index[answer][word] = count

                # Build frequency index: answer -> total_occurrences
                _answer_frequency_index[answer] += count

                rows_processed += 1

                if rows_processed % 100000 == 0:
                    print(f"  Loaded {rows_processed:,} word-answer associations...")

        print(f"Word frequency data loaded successfully!")
        print(f"  Total word-answer pairs: {rows_processed:,}")
        print(f"  Unique answers: {len(_answer_word_index):,}")
        print(
            f"  Unique words: {len(set(word for words in _answer_word_index.values() for word in words.keys())):,}")

        _word_frequency_loaded = True

    except Exception as e:
        print(f"ERROR loading word frequency data: {e}")
        print("Word frequency scoring will be disabled.")


def parse_clue_words(clue_text: str) -> List[str]:
    """
    Parse clue text into individual words.
    Minimal processing: lowercase, remove punctuation, split on whitespace.
    Keeps ALL words including potential cryptic indicators.
    """
    if not clue_text:
        return []

    # Remove punctuation, convert to lowercase, extract words
    words = re.findall(r'\b[a-z]+\b', clue_text.lower())
    return words


def calculate_word_overlap_score(clue_words: List[str], candidate_answer: str) -> float:
    """
    Calculate word overlap score between clue and historical clues for this answer.
    Returns weighted score based on word frequency associations.
    """
    # Normalize candidate answer to uppercase for consistent lookup
    candidate_normalized = candidate_answer.upper().strip()

    if not _word_frequency_loaded or candidate_normalized not in _answer_word_index:
        return 0.0

    answer_words = _answer_word_index[candidate_normalized]
    total_score = 0.0

    for word in clue_words:
        if word in answer_words:
            # Use logarithmic scaling to prevent dominant words from overwhelming
            word_frequency = answer_words[word]
            total_score += min(word_frequency, 20)  # Cap individual word contribution

    return total_score


def get_answer_frequency_score(candidate_answer: str) -> float:
    """
    Get historical frequency score for an answer.
    More frequent answers get higher scores, with logarithmic scaling.
    """
    # Normalize candidate answer to uppercase for consistent lookup
    candidate_normalized = candidate_answer.upper().strip()

    if not _word_frequency_loaded or candidate_normalized not in _answer_frequency_index:
        return 1.0  # Default score for unknown answers

    frequency = _answer_frequency_index[candidate_normalized]

    # Logarithmic scaling to prevent super-frequent answers from dominating
    import math
    return math.log(frequency + 1)


def score_candidates(clue_text: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """
    Score candidates using word-frequency associations and answer popularity.
    Returns list of scored candidates sorted by confidence.
    """
    if not ENABLE_WORD_SCORING or not _word_frequency_loaded:
        # Return candidates with default scores
        return [{"candidate": c, "total_score": 1.0, "word_score": 0.0, "freq_score": 1.0}
                for c in candidates]

    clue_words = parse_clue_words(clue_text)
    scored_candidates = []

    for candidate in candidates:
        candidate_norm = candidate.upper().strip()

        # Calculate component scores
        word_overlap_score = calculate_word_overlap_score(clue_words, candidate_norm)
        frequency_score = get_answer_frequency_score(candidate_norm)

        # Combined weighted score
        total_score = (
                word_overlap_score * WORD_FREQUENCY_WEIGHT +
                frequency_score * ANSWER_FREQUENCY_WEIGHT
        )

        scored_candidates.append({
            "candidate": candidate,
            "total_score": total_score,
            "word_score": word_overlap_score,
            "freq_score": frequency_score,
        })

    # Sort by total score (highest first)
    scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)

    return scored_candidates


# ==============================
# ORIGINAL PIPELINE FUNCTIONS
# ==============================

def _norm_continuous(s: str) -> str:
    """Lowercase, letters only, no spaces."""
    return re.sub(r"[^a-z]", "", s.lower())


def _match_single_clue(query: str, clue_text: str) -> bool:
    """
    Match query against clue_text using continuous normalised letters,
    but tolerate extra words in the clue.
    Example:
      query: "top removed vehicle in struggle"
      clue : "Top removed vehicle in a struggle"
    should match.
    """
    q_words = [w for w in re.findall(r"[A-Za-z]+", query.lower()) if w]
    if not q_words:
        return False
    # Build a regex that matches the words in order with any letters between them.
    pattern = ".*".join(re.escape(_norm_continuous(w)) for w in q_words)
    return re.search(pattern, _norm_continuous(clue_text)) is not None


def _clean_window(w: str) -> str:
    w = _ENUM_RE.sub("", w)
    w = w.strip(" ,.;:-")
    return " ".join(w.split())


def _length_filter(cands: List[str], total_len: int) -> List[str]:
    return [c for c in cands if len(norm_letters(c)) == total_len]


def run_pipeline_probe(
        max_clues: int = MAX_CLUES,
        wordplay_type: str = WORDPLAY_TYPE,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    # Load word frequency data if enabled
    if ENABLE_WORD_SCORING:
        load_answer_word_frequency()

    wp_filter = wordplay_type.lower()

    conn = connect_db()
    cur = conn.cursor()
    graph = load_graph(conn)

    # -----------------------------
    # SOURCE SELECTION
    # -----------------------------
    if SINGLE_CLUE_MATCH:
        cur.execute(
            """
            SELECT clue_text, enumeration, answer, wordplay_type
            FROM clues
            """
        )
        rows = [
            r for r in cur.fetchall()
            if _match_single_clue(SINGLE_CLUE_MATCH, r[0])
        ]
    elif wp_filter == "all":
        cur.execute(
            """
            SELECT clue_text, enumeration, answer, wordplay_type
            FROM clues
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (max_clues,),
        )
        rows = cur.fetchall()
    else:
        cur.execute(
            """
            SELECT clue_text, enumeration, answer, wordplay_type
            FROM clues
            WHERE LOWER(wordplay_type) = ?
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (wp_filter, max_clues),
        )
        rows = cur.fetchall()

    results: List[Dict[str, Any]] = []

    overall = {
        "clues": 0,
        "clues_with_def_match": 0,
        "clues_with_anagram": 0,
        "clues_with_lurker": 0,
        "clues_with_dd": 0,
        "clues_with_improved_ranking": 0,  # NEW: tracking ranking improvements
        # Candidate funnel metrics
        "clues_with_candidates": 0,
        "top_1_hits": 0,
        "top_3_hits": 0,
        "top_5_hits": 0,
        "top_10_hits": 0,
        "top_15_hits": 0,
        "top_20_hits": 0,
        "original_ranks": [],
        "scored_ranks": [],
    }

    for clue, enum, answer_raw, wp_type in rows:
        answer = norm_letters(answer_raw)
        total_len = parse_enum(enum)

        record: Dict[str, Any] = {
            "clue": clue,
            "enumeration": enum,
            "answer": answer,
            "answer_raw": answer_raw,
            "wordplay_type": wp_type,
        }

        # ---- Double Definition ----
        dd_hits = generate_dd_hypotheses(
            clue_text=clue,
            graph=graph,
        )

        # Replicate master_solver behaviour: enforce enumeration on DD hits
        dd_hits = [
            h for h in dd_hits
            if len(norm_letters(h["answer"])) == total_len
        ]

        # ---- Definition ----
        def_result = definition_candidates(
            clue_text=clue,
            enumeration=enum,
            graph=graph,
        )

        raw_windows = [
            _clean_window(w)
            for w in def_result.get("definition_windows", [])
            if w and _clean_window(w)
        ]

        raw_candidates = def_result.get("candidates", []) or []
        flat_candidates = _length_filter(raw_candidates, total_len)

        # ---- NEW: WORD FREQUENCY SCORING ----
        scored_candidates = score_candidates(clue, flat_candidates)

        # Track if scoring improved the ranking of the correct answer
        original_rank = None
        scored_rank = None
        ranking_improved = False

        if flat_candidates and answer_raw in flat_candidates:
            original_rank = flat_candidates.index(answer_raw) + 1
            scored_list = [sc["candidate"] for sc in scored_candidates]
            if answer_raw in scored_list:
                scored_rank = scored_list.index(answer_raw) + 1
                ranking_improved = scored_rank < original_rank

        # ---- WINDOW → CANDIDATES (INVERTED SUPPORT) ----
        window_support: Dict[str, List[str]] = defaultdict(list)
        # All windows, including those with zero candidates
        window_candidates_by_window: Dict[str, List[str]] = {w: [] for w in raw_windows}

        for window in raw_windows:
            window_key = clean_key(window)

            keys = [window_key]
            for art in ("a ", "an ", "the "):
                keys.append(clean_key(art + window_key))

            for key in keys:
                if key not in graph:
                    continue
                for cand in graph[key]:
                    if cand in flat_candidates:
                        window_support[window].append(cand)
                        window_candidates_by_window[window].append(cand)

        for w in window_support:
            window_support[w] = sorted(set(window_support[w]))

        for w in window_candidates_by_window:
            window_candidates_by_window[w] = sorted(set(window_candidates_by_window[w]))

        windows_with_hits = {w: c for w, c in window_support.items() if c}

        # ---- Anagrams ----
        anag_hits = generate_anagram_hypotheses(
            clue_text=clue,
            enumeration=total_len,
            candidates=flat_candidates,
        )

        # ---- Lurkers ----
        lurk_hits = generate_lurker_hypotheses(
            clue_text=clue,
            enumeration=total_len,
            candidates=flat_candidates,
        )

        definition_answer_present = (
                answer in {norm_letters(c) for c in flat_candidates}
        )

        # ---- HIT CLASSIFICATION (REPORTING ONLY) ----
        hit_types = []
        if windows_with_hits:
            hit_types.append("definition")
        if anag_hits:
            hit_types.append("anagram")
        if lurk_hits:
            hit_types.append("lurker")
        if dd_hits:
            hit_types.append("dd")

        hit_any = bool(hit_types)

        overall["clues"] += 1
        if definition_answer_present:
            overall["clues_with_def_match"] += 1
        if anag_hits:
            overall["clues_with_anagram"] += 1
        if lurk_hits:
            overall["clues_with_lurker"] += 1
        if dd_hits:
            overall["clues_with_dd"] += 1
        if ranking_improved:
            overall["clues_with_improved_ranking"] += 1

        # Track candidate funnel metrics (only when we have candidates and the answer is present)
        if flat_candidates and answer_raw in flat_candidates:
            overall["clues_with_candidates"] += 1

            if original_rank:
                overall["original_ranks"].append(original_rank)
            if scored_rank:
                overall["scored_ranks"].append(scored_rank)

                # Track top-N hit rates for scored candidates
                if scored_rank <= 1:
                    overall["top_1_hits"] += 1
                if scored_rank <= 3:
                    overall["top_3_hits"] += 1
                if scored_rank <= 5:
                    overall["top_5_hits"] += 1
                if scored_rank <= 10:
                    overall["top_10_hits"] += 1
                if scored_rank <= 15:
                    overall["top_15_hits"] += 1
                if scored_rank <= 20:
                    overall["top_20_hits"] += 1

        # Reporting filter: show only clues where answer is NOT in definition candidates
        # (unless SINGLE_CLUE_MATCH is explicitly set)
        if ONLY_MISSING_DEFINITION and not SINGLE_CLUE_MATCH:
            if definition_answer_present:
                continue

        record["summary"] = {
            "definition_windows_with_hits": len(windows_with_hits),
            "definition_candidates": len(set(flat_candidates)),
            "anagram_hits": len(anag_hits),
            "lurker_hits": len(lurk_hits),
            "double_definition_hits": len(dd_hits),
            "hit_any": hit_any,
            "hit_types": hit_types,
            "answer_in_definition_candidates": definition_answer_present,
            "original_rank": original_rank,
            "scored_rank": scored_rank,
            "ranking_improved": ranking_improved,
        }

        record["window_support"] = windows_with_hits
        record["window_candidates_by_window"] = window_candidates_by_window
        record["definition_candidates"] = flat_candidates
        record["definition_answer_present"] = definition_answer_present
        record["scored_candidates"] = scored_candidates  # NEW: scored candidate list
        record["anagrams"] = anag_hits
        record["lurkers"] = lurk_hits
        record["double_definition"] = dd_hits

        results.append(record)

    conn.close()
    return results, overall


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    data, overall = run_pipeline_probe()

    print("\n=== ENHANCED PIPELINE SIMULATOR WITH WORD-FREQUENCY SCORING ===")
    print("POPULATION SUMMARY:")
    print(f"  clues processed           : {overall['clues']}")
    print(f"  clues w/ def answer match : {overall['clues_with_def_match']}")
    print(f"  clues w/ anagram hit      : {overall['clues_with_anagram']}")
    print(f"  clues w/ lurker hit       : {overall['clues_with_lurker']}")
    print(f"  clues w/ DD hit           : {overall['clues_with_dd']}")
    if ENABLE_WORD_SCORING:
        print(f"  clues w/ improved ranking : {overall['clues_with_improved_ranking']}")
    print()

    # Candidate funnel effectiveness metrics
    if ENABLE_WORD_SCORING and overall["clues_with_candidates"] > 0:
        candidates_total = overall["clues_with_candidates"]
        print("CANDIDATE FUNNEL EFFECTIVENESS:")
        print(f"  Clues with answer in candidates: {candidates_total}")
        print()

        # Hit rate statistics
        print("HIT RATES (answer found in top-N scored candidates):")
        if overall["top_1_hits"] > 0:
            hit_rate_1 = (overall["top_1_hits"] / candidates_total) * 100
            print(
                f"  Top 1:   {overall['top_1_hits']:3d}/{candidates_total:3d} = {hit_rate_1:5.1f}%")

        if overall["top_3_hits"] > 0:
            hit_rate_3 = (overall["top_3_hits"] / candidates_total) * 100
            print(
                f"  Top 3:   {overall['top_3_hits']:3d}/{candidates_total:3d} = {hit_rate_3:5.1f}%")

        if overall["top_5_hits"] > 0:
            hit_rate_5 = (overall["top_5_hits"] / candidates_total) * 100
            print(
                f"  Top 5:   {overall['top_5_hits']:3d}/{candidates_total:3d} = {hit_rate_5:5.1f}%")

        if overall["top_10_hits"] > 0:
            hit_rate_10 = (overall["top_10_hits"] / candidates_total) * 100
            print(
                f"  Top 10:  {overall['top_10_hits']:3d}/{candidates_total:3d} = {hit_rate_10:5.1f}%")

        if overall["top_15_hits"] > 0:
            hit_rate_15 = (overall["top_15_hits"] / candidates_total) * 100
            print(
                f"  Top 15:  {overall['top_15_hits']:3d}/{candidates_total:3d} = {hit_rate_15:5.1f}%")

        if overall["top_20_hits"] > 0:
            hit_rate_20 = (overall["top_20_hits"] / candidates_total) * 100
            print(
                f"  Top 20:  {overall['top_20_hits']:3d}/{candidates_total:3d} = {hit_rate_20:5.1f}%")

        print()

        # Ranking improvement statistics
        if overall["original_ranks"] and overall["scored_ranks"]:
            import statistics

            avg_original = statistics.mean(overall["original_ranks"])
            avg_scored = statistics.mean(overall["scored_ranks"])
            median_original = statistics.median(overall["original_ranks"])
            median_scored = statistics.median(overall["scored_ranks"])

            print("RANKING IMPROVEMENT STATISTICS:")
            print(
                f"  Average rank:    {avg_original:.1f} → {avg_scored:.1f} (improvement: {avg_original - avg_scored:+.1f})")
            print(
                f"  Median rank:     {median_original:.0f} → {median_scored:.0f} (improvement: {median_original - median_scored:+.1f})")

            # Calculate how many ranks improved on average
            improvement_rate = (overall[
                                    "clues_with_improved_ranking"] / candidates_total) * 100
            print(
                f"  Improved ranks:  {overall['clues_with_improved_ranking']:3d}/{candidates_total:3d} = {improvement_rate:.1f}%")

        print()
        print("WORDPLAY EFFICIENCY IMPACT:")
        if overall["top_5_hits"] > 0:
            eff_5 = (overall["top_5_hits"] / candidates_total) * 100
            print(
                f"  With top-5 focus:  {eff_5:.1f}% of clues would need only 5-candidate wordplay")
        if overall["top_10_hits"] > 0:
            eff_10 = (overall["top_10_hits"] / candidates_total) * 100
            print(
                f"  With top-10 focus: {eff_10:.1f}% of clues would need only 10-candidate wordplay")
        if overall["top_15_hits"] > 0:
            eff_15 = (overall["top_15_hits"] / candidates_total) * 100
            print(
                f"  With top-15 focus: {eff_15:.1f}% of clues would need only 15-candidate wordplay")
    print()

    for i, r in enumerate(data[:MAX_DISPLAY], 1):
        print(f"[{i}] CLUE: {r['clue']}")
        print(f"    TYPE: {r['wordplay_type']}")
        print(f"    ENUM: {r['enumeration']}")
        print(f"    ANSWER: {r['answer_raw']}")
        print(f"    SUMMARY: {r['summary']}")
        print(
            f"    HIT: {r['summary']['hit_any']} "
            f"({', '.join(r['summary']['hit_types']) or 'none'}) | "
            f"ANSWER IN DEF CANDIDATES: {r['summary']['answer_in_definition_candidates']}"
        )

        # ---- NEW: SCORING RESULTS ----
        if ENABLE_WORD_SCORING and r.get("scored_candidates"):
            print("    WORD-FREQUENCY SCORING:")
            if r['summary']['original_rank'] and r['summary']['scored_rank']:
                improvement = "IMPROVED" if r['summary'][
                    'ranking_improved'] else "same/worse"
                print(
                    f"      Answer ranking: {r['summary']['original_rank']} → {r['summary']['scored_rank']} ({improvement})")

            print(f"      CORRECT ANSWER: {r['answer_raw']}")

            # Show top 10 with detailed scoring
            print(f"      Top 10 scored candidates:")
            top_10 = r["scored_candidates"][:10]
            for j, sc in enumerate(top_10, 1):
                marker = "★" if sc["candidate"] == r["answer_raw"] else " "
                print(
                    f"      {marker} {j:2d}. {sc['candidate']:12s} (total: {sc['total_score']:.2f}, word: {sc['word_score']:.1f}, freq: {sc['freq_score']:.1f})")

            # Show remaining candidates (if any) without detailed scoring
            remaining = r["scored_candidates"][10:]
            if remaining:
                print(f"      Remaining {len(remaining)} candidates (unscored):")
                remaining_text = []
                for sc in remaining:
                    marker = "★" if sc["candidate"] == r["answer_raw"] else ""
                    remaining_text.append(f"{marker}{sc['candidate']}")

                # Display in rows of 6 candidates each
                for i in range(0, len(remaining_text), 6):
                    row = remaining_text[i:i + 6]
                    print(f"      {', '.join(row)}")

            # Summary of all candidates
            total_candidates = len(r["scored_candidates"])
            answer_in_candidates = r["answer_raw"] in [sc["candidate"] for sc in
                                                       r["scored_candidates"]]
            print(
                f"      Total candidates: {total_candidates}, Answer present: {answer_in_candidates}")

        else:
            # Show all candidates without scoring when scoring is disabled
            candidates = r["definition_candidates"]
            if candidates:
                print(f"    ALL DEFINITION CANDIDATES:")
                print(f"      CORRECT ANSWER: {r['answer_raw']}")

                # Mark correct answer and display candidates
                marked_candidates = []
                for cand in candidates:
                    marker = "★" if cand == r["answer_raw"] else ""
                    marked_candidates.append(f"{marker}{cand}")

                # Display in rows of 8 candidates each
                for i in range(0, len(marked_candidates), 8):
                    row = marked_candidates[i:i + 8]
                    print(f"      {', '.join(row)}")

                answer_present = r["answer_raw"] in candidates
                original_rank = candidates.index(
                    r["answer_raw"]) + 1 if answer_present else None
                print(
                    f"      Total candidates: {len(candidates)}, Answer present: {answer_present}")
                if original_rank:
                    print(f"      Answer position: {original_rank}")
            else:
                print(f"    NO DEFINITION CANDIDATES FOUND")
                print(f"      CORRECT ANSWER: {r['answer_raw']}")

        # ---- WORDPLAY HIT DETAILS (REPORTING ONLY) ----
        if r["anagrams"]:
            print("    ANAGRAM HITS:")
            for h in r["anagrams"]:
                print(f"      {h}")

        if r["lurkers"]:
            print("    LURKER HITS:")
            for h in r["lurkers"]:
                print(f"      {h}")

        if r["double_definition"]:
            print("    DOUBLE-DEFINITION HITS:")
            for h in r["double_definition"]:
                print(f"      {h}")
        print(f"    WINDOW → CANDIDATES: {r['window_support']}")

        print("    ALL WINDOWS → CANDIDATES:")
        for w, cands in sorted(
                r["window_candidates_by_window"].items(),
                key=lambda x: (len(x[0].split()), x[0].lower())
        ):
            if cands:
                print(f"      {w} → {', '.join(cands)}")
        print("-" * 80)