import sqlite3
import re
import json
from collections import defaultdict
from typing import List, Tuple

DB_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic.db"

# ============================================================
#               LOAD MULTI-WORD INDICATORS
# ============================================================

def load_multiword_indicators(conn):
    cur = conn.cursor()
    rows = cur.execute("SELECT word FROM indicators").fetchall()
    multi = []
    for (w,) in rows:
        if not w:
            continue
        w = w.strip()
        if " " in w:
            multi.append(w.lower())
    # Sort by descending length so longer phrases match first
    multi.sort(key=lambda x: -len(x.split()))
    return multi


def clue_tokens(text: str) -> List[str]:
    """Lowercase + basic tokenisation"""
    text = re.sub(r"\([^)]*\)", "", text)  # strip enumeration safely
    return re.findall(r"[A-Za-z]+", text.lower())


def tokens_start_with(tokens, phrase_tokens):
    n = len(phrase_tokens)
    return tokens[:n] == phrase_tokens


def tokens_end_with(tokens, phrase_tokens):
    n = len(phrase_tokens)
    return tokens[-n:] == phrase_tokens


# ============================================================
#       BUILD DEFINITION WINDOWS (BASELINE BEHAVIOUR)
# ============================================================

def build_windows_baseline(tokens, max_window=7):
    """Standard prefix + suffix windows."""
    windows = []

    # prefix windows
    for i in range(1, min(max_window, len(tokens)) + 1):
        windows.append(("prefix", tokens[:i]))

    # suffix windows
    for i in range(1, min(max_window, len(tokens)) + 1):
        windows.append(("suffix", tokens[-i:]))

    return windows


def build_windows_prefix_only(tokens, max_window=7):
    return [("prefix", tokens[:i])
            for i in range(1, min(max_window, len(tokens)) + 1)]


def build_windows_suffix_only(tokens, max_window=7):
    return [("suffix", tokens[-i:])
            for i in range(1, min(max_window, len(tokens)) + 1)]


# ============================================================
#       PLACEHOLDER FOR YOUR EXISTING CANDIDATE ENGINE
# ============================================================

def generate_candidates(def_tokens, clue_text, enumeration):
    """
    This function MUST call your existing graph + enumeration +
    embedding ranking logic.

    This placeholder simply returns [].
    The pass/fail logic is correct — you just plug in your engine.
    """
    return []


# ============================================================
#                     MAIN PROCESS LOGIC
# ============================================================

def process_clue(clue_text, answer, enumeration, multi):
    """
    Runs Pass 1, then (only if needed) Pass 2 with window shrinking.
    """

    # For tokenisation: drop enumeration
    body = re.sub(r"\([^)]*\)", "", clue_text)
    tokens = clue_tokens(body)

    # ======================================================
    #                PASS 1 — BASELINE
    # ======================================================

    windows = build_windows_baseline(tokens)
    for _, win_tokens in windows:
        cands = generate_candidates(win_tokens, clue_text, enumeration)
        if answer in cands:
            return True, "pass1"

    # ======================================================
    #        PASS 2 — SHRINK WINDOWS BASED ON POSITIONAL
    #                  MULTI-WORD INDICATOR
    # ======================================================

    # Build tokenised versions of each indicator
    for phrase in multi:
        p_tokens = phrase.split()

        # -------- start match → definition at end only --------
        if tokens_start_with(tokens, p_tokens):
            windows2 = build_windows_suffix_only(tokens)
            for _, win_tokens in windows2:
                cands = generate_candidates(win_tokens, clue_text, enumeration)
                if answer in cands:
                    return True, "pass2_start_indicator"
            return False, "fail_after_start_indicator"

        # -------- end match → definition at start only --------
        if tokens_end_with(tokens, p_tokens):
            windows2 = build_windows_prefix_only(tokens)
            for _, win_tokens in windows2:
                cands = generate_candidates(win_tokens, clue_text, enumeration)
                if answer in cands:
                    return True, "pass2_end_indicator"
            return False, "fail_after_end_indicator"

    # No start/end multi-word indicator → nothing to do
    return False, "fail_no_indicator"


# ============================================================
#                     DRIVER (EXAMPLE)
# ============================================================

def main():
    conn = sqlite3.connect(DB_PATH)
    multi = load_multiword_indicators(conn)

    # Replace this with your real clue loop
    test_clue = "Something one might say at heart (9)"
    test_answer = "EXAMPLE"  # placeholder
    enumeration = "9"

    ok, mode = process_clue(test_clue, test_answer, enumeration, multi)
    print(ok, mode)


if __name__ == "__main__":
    main()
