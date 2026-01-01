import csv
import os
import re
from difflib import SequenceMatcher

# Path to export.csv
DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "export.csv",
)


def _normalise(text: str) -> str:
    """Lowercase + collapse spaces for simpler matching."""
    text = text.lower().strip()
    return " ".join(text.split())


def _parse_enumeration(raw: str):
    """
    Convert enumeration string like '7', '3,5', '2,3,4' into a list of ints.
    Returns None if no valid numbers.
    """
    if not raw:
        return None

    parts = [p.strip() for p in raw.split(",")]
    nums = []

    for p in parts:
        if p.isdigit():
            nums.append(int(p))

    return nums if nums else None


def _extract_enumeration_from_clue(clue: str):
    """
    Extract enumeration from the actual text the user typed, e.g. '(3,5)' or '(7)'.
    Returns list of ints or None.
    """
    m = re.search(r"\(([\d,\s]+)\)", clue)
    if not m:
        return None

    inner = m.group(1)
    parts = [p.strip() for p in inner.split(",")]
    nums = []

    for p in parts:
        if p.isdigit():
            nums.append(int(p))

    return nums if nums else None


def _load_dataset():
    """
    Load clue, answer, explanation, and enumeration list from CSV.
    Rows stored as:
    (normalised_clue, answer, explanation, original_clue, enum_list)
    """
    rows = []
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clue = row["clue_text"]
            answer = row["answer"]
            explanation = row.get("explanation", "") or ""

            enum_list = _parse_enumeration(row.get("enumeration", ""))

            norm = _normalise(clue)

            rows.append((norm, answer, explanation, clue, enum_list))

    return rows


# Load dataset ONCE
_DATA = _load_dataset()


def lookup_clue(clue: str, min_ratio: float = 0.90):
    """
    Fuzzy-match a clue against the dataset,
    enforcing enumeration structure if provided.

    Returns:
        (answer, explanation, original_clue, similarity_score)
    or None.
    """

    # Extract enumeration pattern from the user's clue
    enum_required = _extract_enumeration_from_clue(clue)

    norm = _normalise(clue)

    best = None
    best_score = 0.0

    for stored_norm, answer, explanation, original_clue, stored_enum in _DATA:

        # If the user gave (3,5), only match clues with EXACT same pattern
        if enum_required and stored_enum != enum_required:
            continue

        score = SequenceMatcher(None, norm, stored_norm).ratio()

        if score > best_score:
            best_score = score
            best = (answer, explanation, original_clue)

    if best and best_score >= min_ratio:
        answer, explanation, original_clue = best
        return answer, explanation, original_clue, best_score

    return None
