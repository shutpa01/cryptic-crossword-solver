"""
Stage-3 wordplay pipeline.

Owns:
- ordering
- gating
- short-circuiting

Master solver calls ONE function from here and stays pure forever.
"""

from solver.wordplay.lurker.lurker_stage import generate_lurker_hypotheses
from solver.wordplay.double_definition.dd_stage import generate_dd_hypotheses


def run_stage3_pipeline(
    *,
    clue_text: str,
    enumeration_str: str,
    total_len: int,
    candidates: list[str],
    graph,
    wordlist,
):
    """
    Apply Stage-3 handlers in order.
    Short-circuits on first successful handler.

    Returns a dict:
      {
        "lurker_hypotheses": [...],
        "dd_hypotheses": [...],
        "stage3_hit": bool,
        "stage3_type": str | None,
      }
    """

    # --- Handler 1: Lurkers ---
    lurker_hypotheses = generate_lurker_hypotheses(
        clue_text=clue_text,
        enumeration=total_len,
        candidates=candidates,
        wordlist=wordlist,
    )
    if lurker_hypotheses:
        return {
            "lurker_hypotheses": lurker_hypotheses,
            "dd_hypotheses": [],
            "stage3_hit": True,
            "stage3_type": "lurker",
        }

    # --- Handler 2: Double Definitions ---
    dd_hypotheses = generate_dd_hypotheses(
        clue_text=clue_text,
        enumeration_str=enumeration_str,
        total_len=total_len,
        candidates=candidates,
        graph=graph,
        wordlist=wordlist,
    )
    if dd_hypotheses:
        return {
            "lurker_hypotheses": [],
            "dd_hypotheses": dd_hypotheses,
            "stage3_hit": True,
            "stage3_type": "double_definition",
        }

    # --- No Stage-3 hit ---
    return {
        "lurker_hypotheses": [],
        "dd_hypotheses": [],
        "stage3_hit": False,
        "stage3_type": None,
    }
