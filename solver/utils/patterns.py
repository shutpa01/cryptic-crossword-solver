import re

def word_fits_pattern(word: str, pattern: str) -> bool:
    """
    Pattern may contain:
        ?   = unknown letter
        A-Z = fixed letter
    Example: A?E?I?
    """

    if len(word) != len(pattern):
        return False

    for wc, pc in zip(word.lower(), pattern.lower()):
        if pc == '?':
            continue
        if wc != pc:
            return False
    return True


def filter_by_pattern(candidates, pattern):
    if not pattern:
        return candidates
    return [w for w in candidates if word_fits_pattern(w, pattern)]
