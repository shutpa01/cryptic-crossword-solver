# solver/wordplay/anagram/anagram_probe.py
#
# Anagram probe:
# - mirrors master_solver behaviour up to and including anagrams
# - runs on a small rolling cohort
# - preserves full anagram evidence
#
# Population solver is untouched.

from __future__ import annotations

from typing import List, Dict, Any

from solver.selection.presets import CURRENT_CRITERIA
from solver.selection.criteria import get_cohort

from solver.solver_engine.resources import (
    connect_db,
    load_graph,
    parse_enum,
    norm_letters,
)

from solver.definition.definition_engine_edges import definition_candidates
from solver.wordplay.anagram.anagram_stage import generate_anagram_hypotheses
from solver.wordplay.double_definition.dd_stage import generate_dd_hypotheses


def _length_filter(cands: List[str], total_len: int) -> List[str]:
    return [c for c in cands if len(norm_letters(c)) == total_len]


def run_anagram_probe(
    max_clues: int = 50,
) -> List[Dict[str, Any]]:
    """
    Run until `max_clues` reach the anagram stage.
    Returns a list of evidence dicts (one per clue).
    """

    conn = connect_db()
    graph = load_graph(conn)
    rows = get_cohort(conn, CURRENT_CRITERIA)

    results: List[Dict[str, Any]] = []

    for row in rows:
        if len(results) >= max_clues:
            break

        clue = row["clue_text"]
        enum = row["enumeration"]
        answer = norm_letters(row["answer"])
        total_len = parse_enum(enum)

        # ---- Stage 1: Double Definition (same as master_solver) ----
        dd_hits = generate_dd_hypotheses(clue_text=clue, graph=graph)
        if dd_hits:
            continue  # solved earlier; never reaches anagrams

        # ---- Stage 2: Definition gate (same as master_solver) ----
        def_result = definition_candidates(
            clue_text=clue,
            enumeration=enum,
            graph=graph,
        )

        raw_candidates = def_result.get("candidates", []) or []
        flat_candidates = _length_filter(raw_candidates, total_len)

        cand_norm = {norm_letters(c) for c in flat_candidates}
        if answer not in cand_norm:
            continue  # dropped by definition gate

        # ---- Stage 3: Anagrams (instrumented) ----
        anagram_hits = generate_anagram_hypotheses(
            clue_text=clue,
            enumeration=total_len,
            candidates=flat_candidates,
        )

        results.append(
            {
                "clue": clue,
                "enumeration": enum,
                "answer": answer,
                "candidates_sent_to_anagrams": flat_candidates,
                "anagram_hits": anagram_hits,
            }
        )

    conn.close()
    return results


if __name__ == "__main__":
    evidence = run_anagram_probe(max_clues=50)

    print("\n=== ANAGRAM PROBE RESULTS ===\n")
    for i, item in enumerate(evidence, 1):
        print(f"[{i}] CLUE: {item['clue']}")
        print(f"    ENUM: {item['enumeration']}")
        print(f"    ANSWER: {item['answer']}")
        print(f"    CANDIDATES: {item['candidates_sent_to_anagrams']}")
        if item["anagram_hits"]:
            print(f"    ANAGRAM HITS: {item['anagram_hits']}")
        else:
            print("    ANAGRAM HITS: NONE")
        print("-" * 60)
