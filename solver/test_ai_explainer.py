from solver.wordplay.anagram_solver import solve_anagram_clue
from solver.lexicon import LEXICON

clue = "Lord held up by dangerous special character"
tokens = clue.split()

# Suppose the definition model gave this:
def_tokens = ["dangerous", "special"]

# Indicators come from lexicon
anagram_inds = set(LEXICON.indicators.get("anagram", []))

# Suppose we know slot length = 8
length = 8

results = solve_anagram_clue(tokens, def_tokens, anagram_inds, length)
print(results)
