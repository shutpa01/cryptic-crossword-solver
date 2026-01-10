# solver/orchestration/master_solver.py

from __future__ import annotations

from typing import List

from solver.selection.presets import DB_PATH, CURRENT_CRITERIA
from solver.selection.criteria import get_cohort

from solver.solver_engine.resources import (
    connect_db,
    load_graph,
    parse_enum,
    norm_letters,
)

from solver.definition.definition_engine_edges import definition_candidates
from solver.wordplay.anagram.anagram_stage import generate_anagram_hypotheses
from solver.wordplay.lurker.lurker_stage import generate_lurker_hypotheses
from solver.wordplay.double_definition.dd_stage import generate_dd_hypotheses

from solver.orchestration.cascade_analyser import CascadeAnalyser
from solver.orchestration.output_presets import active_output


def _length_filter(cands: List[str], total_len: int) -> List[str]:
    """
    Keep candidates whose normalised letter-length matches enumeration.
    """
    out: List[str] = []
    for c in cands:
        if len(norm_letters(c)) == total_len:
            out.append(c)
    return out


def main():
    conn = connect_db()
    graph = load_graph(conn)

    analyser = CascadeAnalyser(max_examples_per_bucket=5)

    rows = get_cohort(conn, CURRENT_CRITERIA)

    # ------------------------------------------------------------
    # Cascade counters (pure orchestration - no calculation logic)
    # ------------------------------------------------------------
    n0 = len(rows)

    dd_entered = n0
    dd_solved = 0

    gate_entered = 0
    gate_passed = 0
    gate_failed = 0

    anagram_entered = 0
    anagram_solved = 0
    all_anagram_hits = []  # NEW: Collect all hypotheses for analyser

    lurker_entered = 0
    lurker_solved = 0

    # ------------------------------------------------------------
    # Run cascade
    # ------------------------------------------------------------
    carried_after_dd = 0
    carried_after_gate = 0
    carried_after_anagram = 0
    carried_after_lurker = 0

    for row in rows:
        clue = row["clue_text"]
        enum = row["enumeration"]
        db_answer = norm_letters(row["answer"])
        total_len = parse_enum(enum)

        # ---- Stage 1: Double Definition (DD) ----
        dd_hits = generate_dd_hypotheses(clue_text=clue, graph=graph)

        # Enforce enumeration on DD hits (match pipeline behaviour)
        dd_hits = [
            h for h in dd_hits
            if len(norm_letters(h["answer"])) == total_len
        ]

        if dd_hits:
            dd_solved += 1
            continue

        carried_after_dd += 1

        # ---- Stage 2: Definition Gate (diagnostic + hard filter) ----
        gate_entered += 1

        def_result = definition_candidates(
            clue_text=clue,
            enumeration=enum,
            graph=graph,
        )

        # Definition windows are context only in this prototype.
        # Candidates ARE used (as a pool) for later stages, but the DB answer is used only as a gate.
        raw_candidates = def_result.get("candidates", []) or []
        flat_candidates = _length_filter(raw_candidates, total_len)

        # Gate check uses normalised strings for safety.
        # DB answer is NEVER used for solvingâ€"only for this diagnostic filter.
        candidate_norm = {norm_letters(c) for c in flat_candidates}
        if db_answer not in candidate_norm:
            gate_failed += 1
            continue

        gate_passed += 1
        carried_after_gate += 1

        # ---- Stage 3: Anagrams ----
        anagram_entered += 1
        anagram_hits = generate_anagram_hypotheses(
            clue_text=clue,
            enumeration=total_len,
            candidates=flat_candidates,
        )

        if anagram_hits:
            anagram_solved += 1
            # Pure orchestration: collect hypotheses for analyser (no calculation here)
            all_anagram_hits.extend(anagram_hits)
            continue

        carried_after_anagram += 1

        # ---- Stage 4: Lurkers ----
        lurker_entered += 1
        lurker_hits = generate_lurker_hypotheses(
            clue_text=clue,
            enumeration=total_len,
            candidates=flat_candidates,
        )
        if lurker_hits:
            lurker_solved += 1
            continue

        carried_after_lurker += 1

    # ------------------------------------------------------------
    # Record population stats (pure orchestration - calculation done by analyser)
    # ------------------------------------------------------------
    analyser.record_simple(
        stage_name="Double Definitions",
        entered=dd_entered,
        solved=dd_solved,
        forwarded=carried_after_dd,
    )

    analyser.record_simple(
        stage_name="Definition Gate (answer present as candidate)",
        entered=gate_entered,
        solved=gate_passed,
        forwarded=carried_after_gate,
    )

    # CLEAN: Let analyser calculate breakdown from hypotheses automatically
    analyser.record_simple(
        stage_name="Anagrams",
        entered=anagram_entered,
        solved=anagram_solved,
        forwarded=carried_after_anagram,
        hypotheses=all_anagram_hits,  # Analyser will auto-calculate breakdown
    )

    analyser.record_simple(
        stage_name="Lurkers",
        entered=lurker_entered,
        solved=lurker_solved,
        forwarded=carried_after_lurker,
    )

    active_output(analyser)
    conn.close()


if __name__ == "__main__":
    main()