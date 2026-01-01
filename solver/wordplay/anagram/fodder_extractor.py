def extract_fodder(tokens, def_tokens, indicators):
    """
    Extract raw anagram fodder from tokens.
    - Remove definition tokens
    - Remove indicator tokens
    - Return the leftover as fodder text
    """

    def_set = set(t.lower() for t in def_tokens)
    ind_set = set(i.lower() for i in indicators)

    cleaned = []
    for tok in tokens:
        low = tok.lower()
        if low in def_set:
            continue
        if low in ind_set:
            continue
        cleaned.append(tok)

    # Join into a single fodder string
    return " ".join(cleaned)
