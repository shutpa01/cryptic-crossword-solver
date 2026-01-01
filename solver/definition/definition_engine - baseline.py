import re
from solver.solver_engine.resources import (
    clean_key,
    clean_val,
    norm_letters,
    tokenize,
    matches_enumeration,
)

MAX_DEF_WORDS = 6

STOP_DEFS = {
    "a","an","the","it","its","this","that","these","those",
    "he","she","they","them","his","her","their",
    "one","ones","some","any","each","all","both"
}

# ------------------------- ENUM -------------------------

def parse_enum(en):
    return sum(map(int, re.findall(r"\d+", en or "")))

def enumeration_candidates(enum, wordlist):
    return [w for w in wordlist if matches_enumeration(w, enum)]

# ------------------------- DEF ROOTS -------------------------

def normalise_definition_root(dp):
    out = {dp}
    if dp.endswith("'s") or dp.endswith("’s"):
        out.add(dp[:-2])
    if dp.endswith("s") and len(dp) > 3:
        out.add(dp[:-1])
    return list(out)

def get_candidates(def_phrase, graph):
    key = clean_key(def_phrase)
    if not key:
        return []
    return graph.get(key, [])

# ------------------------- MAIN ENTRY -------------------------

def definition_candidates(clue_text, enumeration, graph, wordlist):
    answer_len = parse_enum(enumeration)

    body = re.sub(r"\s*\([^)]*\)\s*$", "", clue_text or "").strip()
    tokens = tokenize(body)
    if not tokens:
        return {
            "definition_windows": [],
            "candidates": [],
            "support": {}
        }

    # --- definition windows (exact baseline behaviour) ---
    def_windows = []

    for k in range(1, min(MAX_DEF_WORDS, len(tokens)) + 1):
        dp = clean_key(" ".join(tokens[:k]))
        if dp and dp not in STOP_DEFS:
            def_windows.append(dp)

    for k in range(1, min(MAX_DEF_WORDS, len(tokens)) + 1):
        dp = clean_key(" ".join(tokens[-k:]))
        if dp and dp not in STOP_DEFS:
            def_windows.append(dp)

    # --- candidate admission (OR logic, not AND) ---
    all_candidates = []

    for dp in def_windows:
        for root in normalise_definition_root(dp):
            all_candidates.extend(get_candidates(root, graph))

    all_candidates.extend(enumeration_candidates(enumeration, wordlist))

    # --- length filter ---
    length_ok = [
        c for c in all_candidates
        if len(norm_letters(c)) == answer_len
    ]

    # --- support tracking (annotation only) ---
    support = {}
    for dp in def_windows:
        for root in normalise_definition_root(dp):
            for cand in get_candidates(root, graph):
                if len(norm_letters(cand)) != answer_len:
                    continue
                support.setdefault(cand, set()).add(dp)

    return {
        "definition_windows": def_windows,
        "candidates": length_ok,
        "support": support
    }
