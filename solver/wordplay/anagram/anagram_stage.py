from itertools import combinations
from collections import Counter

from solver.solver_engine.resources import norm_letters


def generate_anagram_hypotheses(clue_text, enumeration, candidates):
    """
    Stage A: Free anagram hypothesis generation (provisional).
    Includes Stage-B hygiene: reject trivial self-anagrams.
    """

    clue_lc = clue_text.lower()

    # ---- normalise candidates to letter counters ----
    candidate_counters = {}
    for cand in candidates:
        norm = norm_letters(cand)
        if len(norm) == enumeration:
            candidate_counters[cand] = Counter(norm)

    if not candidate_counters:
        return []

    # ---- tokenise clue ----
    words = [
        w for w in clue_text.split()
        if w.isalpha()
    ]

    word_counters = [(w, Counter(norm_letters(w))) for w in words]

    hypotheses = []

    for r in range(1, len(word_counters) + 1):
        for idxs in combinations(range(len(word_counters)), r):
            chosen = [word_counters[i] for i in idxs]

            combined = Counter()
            for _, ctr in chosen:
                combined += ctr

            if sum(combined.values()) != enumeration:
                continue

            for candidate, cand_ctr in candidate_counters.items():
                if combined != cand_ctr:
                    continue

                # ---- STAGE B HYGIENE: reject self-anagrams ----
                # If the candidate appears verbatim in the clue, reject
                if candidate.lower() in clue_lc:
                    continue

                used_words = [w for w, _ in chosen]
                unused_words = [
                    w for i, (w, _) in enumerate(word_counters)
                    if i not in idxs
                ]

                hypotheses.append({
                    "answer": candidate,
                    "fodder_words": used_words,
                    "fodder_letters": "".join(sorted(combined.elements())),
                    "unused_words": unused_words,
                    "candidate_source": candidate,
                    "solve_type": "anagram_provisional",
                    "confidence": "provisional",
                })

    return hypotheses
