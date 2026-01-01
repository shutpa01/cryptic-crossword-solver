import sqlite3
from pathlib import Path


def get_db_path() -> Path:
    # Project root = parent of 'scripts' directory
    return Path(__file__).resolve().parents[1] / "data" / "cryptic.db"


def main() -> None:
    db_path = get_db_path()
    print(f"Using DB at: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1) Create the new table (if it doesn't already exist)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS synonyms_pairs (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            word     TEXT NOT NULL,
            synonym  TEXT NOT NULL
        )
        """
    )

    # Optional: clear it so the script is repeatable
    cur.execute("DELETE FROM synonyms_pairs")
    conn.commit()

    # 2) Read from the existing synonyms table
    cur.execute("SELECT word, synonyms FROM synonyms")
    rows = cur.fetchall()
    print(f"Loaded {len(rows)} rows from synonyms")

    pairs = set()  # to avoid duplicates

    for word, syn_blob in rows:
        if not word:
            continue

        base = word.strip().lower()
        if not base:
            continue

        if syn_blob is None:
            continue

        # split the synonyms field on commas
        for raw in str(syn_blob).split(","):
            s = raw.strip().lower()
            if not s:
                continue
            if s == base:
                continue

            # crude filters to drop obvious junk like "we can do with it (7)"
            if any(ch.isdigit() for ch in s):
                continue
            if "(" in s or ")" in s:
                continue

            # add both directions
            pairs.add((base, s))
            pairs.add((s, base))

    print(f"Prepared {len(pairs)} unique pairs")

    # 3) Insert into synonyms_pairs
    cur.executemany(
        "INSERT INTO synonyms_pairs (word, synonym) VALUES (?, ?)",
        list(pairs),
    )
    conn.commit()

    print("Insert complete")
    print(
        "Row count in synonyms_pairs:",
        cur.execute("SELECT COUNT(*) FROM synonyms_pairs").fetchone()[0],
    )

    conn.close()


if __name__ == "__main__":
    main()
