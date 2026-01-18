# solver/wordplay/double_definition/dd_stage.py

"""
Double Definition Detection

A true double definition has TWO SEPARATE parts of the clue,
each independently defining the same answer.

Algorithm:
1. Split the clue into LEFT and RIGHT halves at each possible point
2. Generate definition windows from each half independently
3. Look up candidates for each half
4. If both halves produce the SAME candidate → true DD

Example:
  "Run fast (6)" → SPRINT
  Split at word 1: LEFT="Run" | RIGHT="fast"
  LEFT candidates: [SPRINT, JOG, ...]
  RIGHT candidates: [SPRINT, QUICK, ...]
  Overlap: SPRINT → TRUE DD
"""

import re
from collections import defaultdict
from solver.definition.definition_engine import generate_definition_windows
from solver.solver_engine.resources import clean_key, norm_letters


def _extract_words(clue_text: str) -> list:
    """Extract words from clue, removing enumeration."""
    # Remove enumeration like (9) or (3,5)
    text = re.sub(r'\(\d+(?:,\d+)*\)\s*$', '', clue_text).strip()
    # Split into words
    words = text.split()
    return words


def _get_candidates_for_phrase(phrase: str, graph, total_len: int = None) -> dict:
    """
    Get all candidates for windows generated from a phrase.
    Returns dict: {normalized_candidate: [windows that produced it]}
    """
    candidates = defaultdict(list)

    windows = generate_definition_windows(phrase)

    for window in windows:
        key = clean_key(window)
        if not key:
            continue

        graph_candidates = graph.get(key)
        if not graph_candidates:
            continue

        for cand in graph_candidates:
            cand_norm = norm_letters(cand)

            # Apply length filter if specified
            if total_len is not None and len(cand_norm) != total_len:
                continue

            candidates[cand_norm].append((window, cand))

    return candidates


def generate_dd_hypotheses(
        *,
        clue_text: str,
        graph,
        total_len=None,
):
    """
    True Double Definition detection.

    A DD fires only if:
    1. The clue can be split into LEFT and RIGHT parts
    2. LEFT part independently produces candidate X
    3. RIGHT part independently produces the SAME candidate X

    Returns at most ONE hit (existential check).
    """

    if not clue_text:
        return []

    words = _extract_words(clue_text)

    if len(words) < 2:
        return []

    # Try each possible split point
    for split_point in range(1, len(words)):
        left_phrase = ' '.join(words[:split_point])
        right_phrase = ' '.join(words[split_point:])

        # Get candidates from each half independently
        left_candidates = _get_candidates_for_phrase(left_phrase, graph, total_len)
        right_candidates = _get_candidates_for_phrase(right_phrase, graph, total_len)

        # Find overlap - candidates that appear in BOTH halves
        overlap = set(left_candidates.keys()) & set(right_candidates.keys())

        if overlap:
            # Take first overlapping candidate
            cand_norm = next(iter(overlap))

            # Get the actual answer form and windows from each side
            left_window, left_answer = left_candidates[cand_norm][0]
            right_window, right_answer = right_candidates[cand_norm][0]

            return [{
                "answer": left_answer,  # Use actual form from graph
                "windows": [left_window, right_window],
                "left_definition": left_phrase,
                "right_definition": right_phrase,
                "split_point": split_point,
            }]

    return []