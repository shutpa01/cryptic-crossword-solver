# solver/solver_engine/pipeline_simulator.py
#
# Pipeline simulator:
# - runs clues through ALL stages
# - preserves evidence
# - optional wordplay_type filter
#   * wordplay_type="all" => no filtering
#   * otherwise case-insensitive match IF column exists

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Dict, Any

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
WORDPLAY_TYPE = "anagram"              # e.g. "all", "anagram", "lurker", "dd"
ONLY_MISSING_DEFINITION = False   # show only clues where answer NOT in def candidates
MAX_DISPLAY = 10                   # max number of clues to print
SINGLE_CLUE_MATCH = ""           # normalised substring
# match on clue_text (highest priority)


_ENUM_RE = re.compile(r"\(\d+(?:,\d+)*\)")


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
) -> List[Dict[str, Any]]:

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
        }

        record["window_support"] = windows_with_hits
        record["window_candidates_by_window"] = window_candidates_by_window
        record["definition_candidates"] = flat_candidates
        record["definition_answer_present"] = definition_answer_present
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

    print("\n=== PIPELINE SIMULATOR ===")
    print("POPULATION SUMMARY:")
    print(f"  clues processed           : {overall['clues']}")
    print(f"  clues w/ def answer match : {overall['clues_with_def_match']}")
    print(f"  clues w/ anagram hit      : {overall['clues_with_anagram']}")
    print(f"  clues w/ lurker hit       : {overall['clues_with_lurker']}")
    print(f"  clues w/ DD hit           : {overall['clues_with_dd']}")
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
        print("-" * 60)
