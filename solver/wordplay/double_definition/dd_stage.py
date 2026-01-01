# solver/wordplay/double_definition/dd_stage.py

from collections import defaultdict
from solver.definition.definition_engine import generate_definition_windows
from solver.solver_engine.resources import clean_key, norm_letters


def generate_dd_hypotheses(
    *,
    clue_text: str,
    graph,
    total_len=None,
):

    """
    Validation-only DD detection.

    A DD fires if the SAME candidate answer is supported by
    TWO DIFFERENT clue windows, independently.

    Existential:
      - returns at most ONE hit
      - stops as soon as a collision is found
    """

    if not clue_text:
        return []

    # window -> candidates mapping (implicit)
    candidate_to_windows = defaultdict(set)

    windows = generate_definition_windows(clue_text)

    for window in windows:
        key = clean_key(window)
        if not key:
            continue

        candidates = graph.get(key)
        if not candidates:
            continue

        for cand in candidates:
            cand_norm = norm_letters(cand)

            if total_len is not None and len(cand_norm) != total_len:
                continue

            candidate_to_windows[cand_norm].add(key)

            # ---- DD COLLISION FOUND ----
            if len(candidate_to_windows[cand_norm]) >= 2:
                return [{
                    "answer": cand,
                    "windows": list(candidate_to_windows[cand_norm]),
                }]

    return []
