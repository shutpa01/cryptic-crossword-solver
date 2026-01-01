import sqlite3
import re
import string

DB_PATH = r"C:\\Users\\shute\\PycharmProjects\\cryptic_solver\\data\\cryptic_new.db"
SOURCE_TAG = "double_definition"

# Filtering rules
def is_clean(text):
    if not text:
        return False
    if len(text) > 25:
        return False
    if len(text.split()) > 3:
        return False
    if not all(ord(c) < 128 for c in text):  # ASCII only
        return False
    if any(c in text for c in string.punctuation):
        return False
    return True

def clean(text):
    return text.strip().lower()

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("Fetching original DD entries...")
    cur.execute(f"SELECT word, synonym FROM synonyms_pairs WHERE source = ?", (SOURCE_TAG,))
    rows = cur.fetchall()
    print(f"  Retrieved {len(rows)} rows")

    # Optional: backup raw
    with open("backup_dd_pairs.txt", "w", encoding="utf-8") as f:
        for word, syn in rows:
            f.write(f"{word}\t{syn}\n")

    print("Filtering...")
    cleaned = [
        (clean(w), clean(s), SOURCE_TAG)
        for w, s in rows
        if is_clean(w) and is_clean(s)
    ]
    print(f"  {len(cleaned)} rows remain after filtering")

    print("Deleting existing DD rows from DB...")
    cur.execute(f"DELETE FROM synonyms_pairs WHERE source = ?", (SOURCE_TAG,))
    print(f"  Deleted {cur.rowcount} rows")

    print("Reinserting cleaned rows...")
    cur.executemany(
        "INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, ?)",
        cleaned
    )
    print(f"  Inserted {cur.rowcount} rows")

    conn.commit()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()