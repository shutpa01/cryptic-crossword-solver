from solver.lexicon import LEXICON
from solver.wordplay.anagram_engine import solve_anagram

ANAGRAM_INDICATORS = set(LEXICON.indicators.get("anagram", []))

def handle_anagram(fodder, target_length):
    """
    Delegate to the real anagram engine.
    """
    return solve_anagram(fodder, target_length)
