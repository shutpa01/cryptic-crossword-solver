from solver.lexicon import LEXICON

# Pull real container indicators from lexicon
CONTAINER_INDICATORS = set(LEXICON.indicators.get("container", []))

def handle_container(outer, inner):
    """
    Minimal placeholder.
    Real implementation will properly wrap inner inside outer.
    """
    if not outer:
        return inner
    if len(outer) == 1:
        return outer + inner
    return outer[0] + inner + outer[1:]
