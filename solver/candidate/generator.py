from solver.lexicon import LEXICON

def candidates_from_definition(def_tokens, length=None):
    """
    Generate preliminary candidate answers from definition tokens.
    Uses synonyms, direct matches, substitutions.
    """
    def_tokens = [t.lower() for t in def_tokens]
    candidates = set()

    # Direct matches + synonyms + substitutions
    for tok in def_tokens:
        # direct token
        if length is None or len(tok) == length:
            candidates.add(tok)

        # synonyms
        for syn in LEXICON.get_synonyms(tok):
            if length is None or len(syn) == length:
                candidates.add(syn)

        # known cryptic short substitutions (doctor = dr, etc.)
        for sub in LEXICON.get_substitutions(tok):
            if length is None or len(sub) == length:
                candidates.add(sub)

    # synonym → definition overlap search
    for word, syns in LEXICON.synonyms.items():
        if any(t in syns for t in def_tokens):
            if length is None or len(word) == length:
                candidates.add(word)

    return list(candidates)
