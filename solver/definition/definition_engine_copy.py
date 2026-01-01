from solver.solver_engine.resources import (
    clean_key,
    norm_letters,
    build_wordlist,
)

ARTICLES = ("a ", "an ", "the ")


def definition_candidates(clue_text, enumeration, graph):
    # wordlist = build_wordlist()

    definition_windows = generate_definition_windows(clue_text)

    answer_norm = None
    candidates = set()
    support = {}

    for dp in definition_windows:
        key = clean_key(dp)
        if key in graph:
            for cand in graph[key]:
                candidates.add(cand)
                support.setdefault(cand, set()).add(dp)

    # Article-variant matches
    for dp in definition_windows:
        dp_clean = clean_key(dp)
        for art in ARTICLES:
            key = clean_key(art + dp_clean)
            if key in graph:
                for cand in graph[key]:
                    candidates.add(cand)
                    support.setdefault(cand, set()).add(dp)

    return {
        "definition_windows": definition_windows,
        "candidates": list(candidates),
        "support": support,
    }


def generate_definition_windows(clue_text):
    # Same logic as always — scan start and end of clue
    words = clue_text.split()
    windows = set()

    for i in range(len(words)):
        window1 = " ".join(words[: i + 1])
        window2 = " ".join(words[-(i + 1):])
        if window1:
            windows.add(window1.strip())
        if window2:
            windows.add(window2.strip())

    return list(windows)
