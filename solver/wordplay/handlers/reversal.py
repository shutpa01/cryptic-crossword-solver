from solver.lexicon import LEXICON

# Pull real reversal indicators from lexicon
REVERSAL_INDICATORS = set(LEXICON.indicators.get("reversal", []))

def handle_reversal(source):
    """
    Simple placeholder: reverse the string.
    """
    return source[::-1]
