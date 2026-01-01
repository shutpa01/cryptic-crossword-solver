"""
Full-cohort lurker stage for clues with no definition hit.

This stage is deliberately conservative:
- It runs ONLY when definition evidence is absent
- It reuses the existing lurker engine unchanged
- It allows wordlist fallback
- It emits lurker hypotheses tagged as weak evidence

No pruning, no ranking, no heuristics.
"""

from typing import List, Dict, Any

from solver.wordplay.lurker.lurker_stage import generate_lurker_hypotheses


def run_lurker_no_definition(
    clue_text: str,
    enumeration,
    wordlist,
    answer_in_definition_candidates: bool,
) -> List[Dict[str, Any]]:
    """
    Run a full-cohort lurker sweep ONLY if the clue has no definition hit.

    Parameters
    ----------
    clue_text : str
        Raw clue text
    enumeration : Any
        Enumeration object (passed through to lurker engine)
    wordlist : set | list
        Global wordlist (admissible words)
    answer_in_definition_candidates : bool
        Whether the correct answer appeared in definition candidates

    Returns
    -------
    List[Dict[str, Any]]
        Lurker hypotheses, tagged as weak evidence.
    """

    # Gate: do nothing if definition evidence exists
    if answer_in_definition_candidates:
        return []

    # Delegate to existing lurker engine
    lurker_hits = generate_lurker_hypotheses(
        clue_text=clue_text,
        enumeration=enumeration,
        candidates=None,          # force full wordlist search
        wordlist=wordlist,
        allow_wordlist_fallback=True,
    )

    # Tag hits as weak, without altering structure
    for hit in lurker_hits:
        hit["solve_type"] = "lurker"
        hit["confidence"] = "low"
        hit["source"] = "no_definition_lurker"

    return lurker_hits
