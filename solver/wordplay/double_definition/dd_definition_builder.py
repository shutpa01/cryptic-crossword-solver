import sqlite3
import csv
import re
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------

DB_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
DD_CSV_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\raw\double_definitions.csv"
SOURCE_TAG = "double_definition"

STOPWORDS = {
    "and", "or", "with", "for", "to", "of", "in", "on", "at", "by", "from"
}

MAX_SPLITS_PER_CLUE = 6


# ----------------------------
# UTILITIES
# ----------------------------

ENUM_RE = re.compile(r"\s*\([^)]*\)\s*$")


def strip_enumeration(text: str) -> str:
    return ENUM_RE.sub("", text)


def normalise(text: str) -> str:
    text = strip_enumeration(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text: str):
    return [t for t in text.split() if t]


def has_content(tokens_):
    return any(t not in STOPWORDS for t in tokens_)


# ----------------------------
# DD SPLITTING LOGIC
# ----------------------------

def derive_definitions(clue: str):
    toks = tokens(clue)
    n = len(toks)

    if n == 2:
        return toks

    if n <= 4:
        mid = n // 2
        return [" ".join(toks[:mid]), " ".join(toks[mid:])]

    defs = []
    for i in range(1, n):
        left = toks[:i]
        right = toks[i:]

        if abs(len(left) - len(right)) > n // 2:
            continue
        if not has_content(left) or not has_content(right):
            continue

        defs.append(" ".join(left))
        defs.append(" ".join(right))

        if len(defs) >= MAX_SPLITS_PER_CLUE * 2:
            break

    return defs


# ----------------------------
# MAIN
# ----------------------------

def main():
    if not Path(DD_CSV_PATH).exists():
        raise FileNotFoundError(DD_CSV_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(synonyms_pairs)")
    cols = [row[1] for row in cur.fetchall()]

    if "source" not in cols:
        cur.execute("ALTER TABLE synonyms_pairs ADD COLUMN source TEXT")
        conn.commit()

    cur.execute("SELECT word, synonym FROM synonyms_pairs")
    existing = set((w, s) for w, s in cur.fetchall())

    inserts = 0

    with open(DD_CSV_PATH, newline="", encoding="cp1252") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clue = normalise(row["Clue"])
            answer = normalise(row["answer"])

            if not clue or not answer:
                continue

            for d in derive_definitions(clue):
                d = normalise(d)
                if not d:
                    continue

                pair = (answer, d)
                if pair in existing:
                    continue

                cur.execute(
                    "INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, ?)",
                    (answer, d, SOURCE_TAG),
                )
                existing.add(pair)
                inserts += 1

    conn.commit()
    conn.close()

    print(f"Inserted {inserts} DD-derived synonym pairs.")


if __name__ == "__main__":
    main()
